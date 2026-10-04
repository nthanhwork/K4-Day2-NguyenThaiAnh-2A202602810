"""Final protocol integration on synthetic CPU images; never DeepWeeds test."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_implementation import TinyModel, create_fixture
import backbone_notebook as nbtrain
import dataset
import final_notebook as final
import inference
import train


class TestFinalProtocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        images,labels = create_fixture(self.root)
        self.base = train.Config(exp_id='B03',backbone='convnext_tiny',device='cpu',amp=False,
            epochs=2,warmup_epochs=0,batch_size=9,img_size=16,num_workers=0,
            images_dir=str(images),labels_dir=str(labels),out_dir=str(self.root/'runs'),
            pred_dir=str(self.root/'predictions'),curves_dir=str(self.root/'curves'))
        self.recipe = replace(self.base,exp_id='T04',mix='cutmix')
        def build(*args,**kwargs):
            model = TinyModel()
            model.lab_pretrained = kwargs.get('pretrained',True) and kwargs.get('init') != 'scratch'
            return model
        original_check = dataset.check_split
        patches = [patch.object(train.dataset,'check_split',side_effect=lambda *a,**kw:original_check(*a,expected_total=45,**kw)),
                   patch.object(train,'_verify_provenance',return_value=None),
                   patch.object(train.model_utils,'build_model',side_effect=build),
                   patch.object(final,'FINAL_ROOT',self.root/'finalization')]
        for item in patches:
            item.start(); self.addCleanup(item.stop)
        train.run(self.base); train.run(self.recipe)
        metadata = json.loads((train.run_dir(self.recipe)/'model.json').read_text())
        latency = {'p95':3.5,'batch':1,'dtype':'fp32','n':100,'warmup':10,'scope':'synthetic fixture latency placeholder'}
        context = patch.object(final,'source_context',return_value=(self.base,self.recipe,metadata,{},latency))
        context.start(); self.addCleanup(context.stop)

    def test_sealed_recipes_alias_seed0_and_independent_training_predictions(self):
        manifest = final.lock_configuration()
        self.assertEqual(manifest['spec']['seeds'],[0,1,2])
        self.assertTrue(manifest['configuration_locked_before_test'])
        entries = manifest['spec']['entries']
        self.assertEqual(len(entries),6)
        for entry in entries:
            cfg = train.Config(**entry['source_cfg'])
            expected = self.base if entry['role'] == 'T00' else self.recipe
            self.assertEqual(cfg.mix,expected.mix)
            self.assertFalse(cfg.save_test_predictions)
            self.assertEqual(cfg.seed,entry['seed'])
            if entry['seed'] == 0:
                self.assertEqual(cfg.exp_id,expected.exp_id)
                self.assertTrue(entry['seed0_reuse'])
            else:
                self.assertEqual(Path(cfg.pred_dir).name,'training')
        final.train_entry('T00',0,subprocess_training=False)
        final.train_entry('F01',0,subprocess_training=False)
        with self.assertRaises(FileNotFoundError): final.seal_checkpoint_set()
        with self.assertRaises(FileNotFoundError): final.test_once('F01',0,device='cpu')
        self.assertFalse(list(final.FINAL_ROOT.glob('*/seed*/test_started.json')))
        # Tampering with the manifest is detected before training or evaluation.
        path = final.FINAL_ROOT/'manifest.json'
        path.write_text(path.read_text().replace('"dtype": "FP32"','"dtype": "FP16"'))
        with self.assertRaisesRegex(ValueError,'manifest was modified'): final.load_manifest()

    def test_six_checkpoints_one_test_forward_cached_reexport_and_official_evaluator(self):
        manifest = final.lock_configuration()
        for entry in manifest['spec']['entries']:
            final.train_entry(entry['role'],entry['seed'],subprocess_training=False)
        final.seal_checkpoint_set()
        for entry in manifest['spec']['entries']:
            role,seed = entry['role'],entry['seed']
            cfg = train.Config(**entry['source_cfg'])
            # Calibration must not overwrite the trainer's raw val CSV.
            nbtrain.verify_outputs(cfg)
            with patch.object(final.inference,'predict_method',wraps=inference.predict_method) as forward:
                result = final.test_once(role,seed,device='cpu')
                self.assertEqual(forward.call_count,1)
                self.assertEqual(result['test']['n'],9)
            with patch.object(final,'load_model',side_effect=AssertionError('Must not load/forward model again')):
                cached = final.test_once(role,seed,device='cpu')
            self.assertEqual(cached,result)
        # Recover an export interrupted after raw logits completed: no second forward.
        target = final.product_path(manifest,'F01uncal',0,'test')
        target.unlink()
        (final.entry_root(final.FINAL_ROOT,'F01',0)/'test_summary.json').unlink()
        with patch.object(final,'load_model',side_effect=AssertionError('Cached export must not load model')):
            final.test_once('F01',0,device='cpu')
        self.assertTrue(target.exists())
        for cfg in (self.base,self.recipe):
            self.assertFalse((train.run_dir(cfg)/'test_started.json').exists())
        frame = final.collect_progress()
        self.assertTrue((frame.status == 'test_complete').all())
        output = final.score_and_grade()
        self.assertTrue((output/'F01_summary.json').exists())
        self.assertTrue((output/'T00_summary.json').exists())
        self.assertTrue((output/'grade_I.json').exists())
        complete = json.loads((final.FINAL_ROOT/'completion.json').read_text())
        self.assertTrue(complete['section10_complete'])
        self.assertEqual(complete['std_ddof'],1)
        # Source checkpoint/model replacement after test is blocked.
        prepared = final.entry_root(final.FINAL_ROOT,'F01',1)/'prepared.json'
        value = json.loads(prepared.read_text()); value['temperature'] *= 1.1
        prepared.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'temperature differs'):
            final.seal_checkpoint_set()

    def test_failed_test_forward_leaves_marker_and_cannot_retry_test(self):
        manifest = final.lock_configuration()
        for entry in manifest['spec']['entries']:
            final.train_entry(entry['role'],entry['seed'],subprocess_training=False)
        final.seal_checkpoint_set()
        with patch.object(final.inference,'predict_method',side_effect=RuntimeError('simulated forward interruption')):
            with self.assertRaisesRegex(RuntimeError,'interruption'):
                final.test_once('F01',0,device='cpu')
        marker = final.entry_root(final.FINAL_ROOT,'F01',0)/'test_started.json'
        self.assertTrue(marker.exists())
        with patch.object(final,'load_model',side_effect=AssertionError('Must not retry interrupted test')):
            with self.assertRaisesRegex(RuntimeError,'raw cache is incomplete'):
                final.test_once('F01',0,device='cpu')


if __name__ == '__main__': unittest.main()
