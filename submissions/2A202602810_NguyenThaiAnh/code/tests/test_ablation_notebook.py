"""Controlled ablation and selection audits, using only synthetic CPU fixtures."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
import torch

from test_implementation import TinyModel, create_fixture
import ablation_notebook as ab
import backbone_notebook as notebooks
import dataset
import model
import train


class TestAblationNotebooks(unittest.TestCase):
    def test_controlled_plan_and_independent_combination_factors(self):
        base = notebooks.t00_config('B03', 'convnext_tiny')
        with patch.object(ab, '_baseline_config', return_value=base):
            configs = ab.configs_for()
            for spec in ab.default_plan():
                differences = {k:v for k,v in asdict(configs[spec.exp_id]).items()
                               if v != asdict(base)[k] and k != 'exp_id'}
                expected = {k:v for k,v in spec.changes.items() if v != asdict(base)[k]}
                self.assertEqual(differences, expected)
            self.assertEqual(configs['T00'].exp_id, 'B03')
            self.assertTrue(all(not cfg.save_test_predictions for cfg in configs.values()))
            self.assertEqual({s.axis for s in ab.default_plan()[1:]}, {'init','augmentation','loss'})
            with self.assertRaisesRegex(ValueError, 'declared controlled'):
                ab.configs_for([ab.default_plan()[0], ab.AblationSpec('T01', 'init', '', {'lr_head':.1})])
        for changes in ({'mix':'cutmix','mix_alpha':1.0}, {'loss':'ls','label_smoothing':.1},
                        {'aug':'basic','loss':'ls','label_smoothing':.1}):
            with self.assertRaisesRegex(ValueError, 'two effective factors'):
                ab.make_combination(changes)
        with self.assertRaisesRegex(ValueError, 'requires an active'):
            ab.make_combination({'aug':'color','loss':'focal','label_smoothing':.1})
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            ab.make_combination({'aug':'color','mix':'cutmix','epochs':20})
        ab.make_combination({'aug':'color','loss':'ls','label_smoothing':.1})

    def test_proposal_uses_two_factors_and_labels_exploration(self):
        frame = pd.DataFrame([{'exp_id':f'T{i:02}', 'status':'complete',
                               'delta_macro_f1': [0,.025,.02,.013,.006,.019,.015,-.002][i]}
                              for i in range(8)])
        proposed = ab.propose_combination(frame)
        # Two init choices cannot be combined; next best comes from the loss axis.
        self.assertEqual(proposed['source_ids'], ['T01','T05'])
        self.assertEqual(proposed['mode'], 'validation_improvements')
        for values, expected in [([0,-.8,-.2,-.01,-.02,-.03,-.04,-.05], ['T03','T04']),
                                 ([0,-.8,-.2,.0001,-.02,-.03,-.04,-.05], ['T03','T04'])]:
            frame['delta_macro_f1'] = values
            proposal = ab.propose_combination(frame)
            self.assertEqual(proposal['source_ids'], expected)
            self.assertEqual(proposal['mode'], 'exploratory')
        frame.loc[4, 'status'] = 'resume'
        with self.assertRaisesRegex(ValueError, 'Complete T00-T07'):
            ab.propose_combination(frame)

    def test_full_matrix_resume_audit_selection_lock_and_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = create_fixture(root)
            base = train.Config(exp_id='B03', backbone='convnext_tiny', device='cpu', amp=False,
                epochs=2, warmup_epochs=0, batch_size=9, img_size=16, num_workers=0,
                images_dir=str(images), labels_dir=str(labels), out_dir=str(root/'runs'),
                pred_dir=str(root/'predictions'), curves_dir=str(root/'curves'))
            def build(*args, **kwargs):
                tiny = TinyModel()
                tiny.lab_pretrained = kwargs['init'] != 'scratch'
                if kwargs['init'] == 'frozen':
                    model.freeze_backbone(tiny)
                return tiny
            check_split = dataset.check_split
            with patch.object(train.dataset, 'check_split', side_effect=lambda *a, **k:check_split(*a, expected_total=45, **k)), \
                 patch.object(train.model_utils, 'build_model', side_effect=build), \
                 patch.object(train, '_verify_provenance', return_value=None), \
                 patch.object(ab, '_baseline_config', return_value=base), \
                 patch.object(ab, 'ABLATION_OUTPUT', root/'ablation'):
                train.run(base)
                configs = ab.configs_for()
                with self.assertRaisesRegex(ValueError, 'incomplete run'):
                    ab.collect_results()
                # Partial summaries must not be presented as completed ablation metrics.
                train.run(configs['T04'], stop_after_epoch=1)
                partial = ab.collect_results(require_complete=False)
                cutmix = partial.set_index('exp_id').loc['T04']
                self.assertEqual(cutmix['status'], 'resume')
                self.assertTrue(np.isnan(cutmix['macro_f1_val']))
                with self.assertRaisesRegex(ValueError, 'incomplete run'):
                    ab.lock_combination({'changes':{'aug':'color','mix':'cutmix'}, 'description':'fixture'})
                # The orchestration is real; replace only its process boundary for synthetic CPU training.
                with patch.object(notebooks, 'run_in_subprocess', side_effect=notebooks.execute_training):
                    frame = ab.run_ablation()
                self.assertEqual(len(frame), 8)
                self.assertTrue((frame.status == 'complete').all())
                self.assertEqual(len(pd.read_csv(root/'ablation/training_per_class_val.csv')), 8*9)
                self.assertEqual(json.loads((root/'ablation/baseline_alias.json').read_text())['source_exp_id'], 'B03')
                self.assertFalse((root/'runs/T00').exists())
                for cfg in configs.values():
                    self.assertFalse(train.pred_path(cfg,'test').exists())
                proposal = ab.propose_combination(frame)
                locked = ab.lock_combination(proposal)
                self.assertEqual(ab.lock_combination(proposal), locked)
                changed = dict(proposal, changes={'aug':'color','mix':'cutmix'} if proposal['changes'] != {'aug':'color','mix':'cutmix'}
                               else {'aug':'color','loss':'ls','label_smoothing':.1})
                with self.assertRaisesRegex(ValueError, 'already locked'):
                    ab.lock_combination(changed)
                combo = ab.configs_for(ab.active_plan())['T08']
                notebooks.execute_training(combo, 4)
                complete = ab.collect_results()
                self.assertEqual(len(complete), 9)
                self.assertTrue(json.loads((root/'ablation/screening_selection.json').read_text())['section8_complete'])
                self.assertIn('T08 so với thành phần tốt nhất', (root/'ablation/ablation_report.md').read_text())
                # Complete run reuse must avoid retraining.
                with patch.object(train, 'run', side_effect=AssertionError('Must reuse complete runs')):
                    with patch.object(notebooks, 'run_in_subprocess', side_effect=notebooks.execute_training):
                        ab.run_ablation(run_ids=['T01'])
                # A corrupted summary with intact predictions must fail the evaluator audit.
                path = train.run_dir(configs['T03'])/'summary.json'
                original = path.read_text()
                corrupt = json.loads(original)
                corrupt['val']['macro_f1'] += .05
                path.write_text(json.dumps(corrupt))
                with self.assertRaisesRegex(ValueError, 'Summary metric differs'):
                    ab.collect_results()
                path.write_text(original)
                # Detect fold/preprocessing drift rather than mixing incomparable results.
                meta_path = train.run_dir(configs['T03'])/'model.json'
                meta = json.loads(meta_path.read_text())
                meta['preprocessing']['resize_size'] += 1
                meta_path.write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, 'preprocessing differs'):
                    ab.verify_ablation(configs['T03'], base)


if __name__ == '__main__':
    unittest.main()
