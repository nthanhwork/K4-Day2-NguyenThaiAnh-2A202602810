"""Numerical, topology, data-order and timing-protocol tests on CPU fixtures."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from test_implementation import TinyModel, create_fixture
import benchmark
import dataset
import inference
import inference_notebook as runner
import train
from eval import compute_metrics, save_predictions


class TestInference(unittest.TestCase):
    def test_temperature_nonincreasing_nll_and_unchanged_ranking(self):
        rng = np.random.default_rng(3)
        logits = rng.normal(size=(240,9))*8
        labels = rng.integers(0,9,240)
        temp = inference.fit_temperature(logits,labels)
        before = inference.apply_temperature(logits,1.)
        after = inference.apply_temperature(logits,temp)
        self.assertGreater(temp,1.)
        self.assertLessEqual(compute_metrics(labels,after.argmax(1),after)['nll'],
                             compute_metrics(labels,before.argmax(1),before)['nll'])
        np.testing.assert_array_equal(before.argmax(1),after.argmax(1))
        np.testing.assert_allclose(after.sum(1),1.)
        for invalid in (0,-1,np.inf,np.nan):
            with self.assertRaises(ValueError): inference.apply_temperature(logits,invalid)
        with self.assertRaises(ValueError): inference.fit_temperature(logits,labels[:-1])

    def test_spatial_views_aggregation_and_ensemble_order(self):
        x = torch.arange(2*3*20*20).reshape(2,3,20,20).float()
        views = inference.views_multicrop(x,16)
        self.assertEqual(len(views),5)
        self.assertEqual(len({v[0,0,0,0].item() for v in views}),5)
        torch.testing.assert_close(views[-1],x[:,:,2:18,2:18])
        torch.testing.assert_close(inference.view_hflip(x),x.flip(-1))
        self.assertEqual(inference.views_multiscale(x,[12,24])[1].shape,(2,3,24,24))
        with self.assertRaises(ValueError): inference.views_multicrop(x,20)
        a = np.tile(np.array([3.,1.,0.,0.,0.,0.,0.,0.,0.]),(2,1))
        b = np.tile(np.array([0.,5.,1.,1.,1.,1.,1.,1.,1.]),(2,1))
        prob = inference.aggregate_views([a,b],'prob')
        logit = inference.aggregate_views([a,b],'logit')
        self.assertFalse(np.allclose(prob,logit))
        np.testing.assert_allclose(prob,(inference.apply_temperature(a,1)+inference.apply_temperature(b,1))/2)
        with tempfile.TemporaryDirectory() as tmp:
            first,second = Path(tmp)/'A_seed0_val.csv',Path(tmp)/'B_seed0_val.csv'
            save_predictions(first,['a.jpg','b.jpg'],[0,1],prob)
            save_predictions(second,['b.jpg','a.jpg'],[1,0],prob[::-1])
            with self.assertRaisesRegex(ValueError,'order mismatch'):
                inference.ensemble_prediction_files([first,second])

    def test_predict_eval_no_grad_full_pipeline_matches_explicit_views(self):
        torch.manual_seed(0)
        model = TinyModel().train()
        x = torch.randn(4,3,20,20)
        loader = [(x,torch.tensor([0,1,2,3]),['a','b','c','d'])]
        state = copy.deepcopy(model.state_dict())
        names,y,logits = inference.predict_logits(model,loader,'cpu',view=lambda b:inference.center_crop(b,16))
        self.assertEqual(names,['a','b','c','d'])
        self.assertFalse(model.training)
        self.assertEqual(logits.shape,(4,9))
        for key in state: torch.testing.assert_close(state[key],model.state_dict()[key])
        with torch.inference_mode():
            expected = [model(v).numpy() for v in inference.views_multicrop(x,16)]
        _,_,actual = inference.predict_method(model,loader,'cpu','I02',img_size=16)
        np.testing.assert_allclose(inference.apply_temperature(actual,1),inference.aggregate_views(expected),atol=1e-7)
        torch.testing.assert_close(inference.method_logits(model,x,'I07',img_size=16,temperature=2.),
                                   inference.method_logits(model,x,'I00',img_size=16)/2)
        with self.assertRaises(ValueError): inference.method_logits(model,x,'I08',img_size=16)

    def test_fusion_copy_and_proven_forward_topology_only(self):
        model = TinyModel().eval()
        x = torch.randn(8,3,16,16)
        fused = inference.fuse_conv_bn(model)
        self.assertEqual(len(fused.lab_fused_pairs),1)
        self.assertIsInstance(model.features[1],nn.BatchNorm2d)
        self.assertIsInstance(fused.features[1],nn.Identity)
        with torch.inference_mode(): torch.testing.assert_close(model(x),fused(x),atol=1e-5,rtol=1e-5)
        class NonSequential(nn.Module):
            def __init__(self):
                super().__init__(); self.conv=nn.Conv2d(3,3,1); self.bn=nn.BatchNorm2d(3)
            def forward(self,x): return self.conv(x)+self.bn(x)
        arbitrary = NonSequential().eval()
        self.assertEqual(inference.fuse_conv_bn(arbitrary).lab_fused_pairs,[])


class TestBenchmark(unittest.TestCase):
    def test_warmup_sync_and_measured_samples(self):
        calls = []
        def fn(): calls.append('fn')
        def sync(): calls.append('sync')
        with patch.object(benchmark.time,'perf_counter',side_effect=[i*.002 for i in range(100)]):
            result = benchmark.bench(fn,10,50,sync)
        self.assertEqual(calls[:10],['fn']*10)
        self.assertEqual(calls[10:],['sync','fn','sync']*50)
        self.assertEqual(result['n'],50)
        np.testing.assert_allclose(result['samples_ms'],2.)
        for kwargs in ({'warmup':9,'iters':50},{'warmup':10,'iters':49}):
            with self.assertRaises(ValueError): benchmark.bench(fn,**kwargs)

    def test_real_cpu_tta_benchmark_counts_forward_views(self):
        model = TinyModel()
        counter = []
        # deepcopy retains this callable hook, so the benchmark copy increments it.
        def hook(module,args,output):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(module.training)
            counter.append(1)
        model.register_forward_hook(hook)
        report = benchmark.tta_latency(model,5,batch_size=2,img_size=16,input_size=20,device='cpu',warmup=10,iters=50)
        self.assertEqual(len(counter),5*60)
        self.assertEqual(report['k_views'],5)
        self.assertEqual(report['n'],50)
        self.assertIn('aggregation',report['scope'])
        self.assertGreater(report['images_per_s'],0)
        self.assertTrue(model.training)

    def test_cached_quality_audit_rejects_order_and_metric_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path,pred = root/'I00',root/'I00_seed0_val.csv'
            names = ['a.jpg','b.jpg']
            labels = np.array([0,1])
            logits = np.eye(9)[:2]*3
            runner._save_quality(path,pred,names,labels,logits,'I00')
            reference = pd.DataFrame({'Filename':names,'Label':labels})
            runner._verify_quality(path,pred,reference)
            with self.assertRaisesRegex(ValueError,'order mismatch'):
                runner._verify_quality(path,pred,reference.iloc[::-1])
            summary = json.loads((path/'summary.json').read_text())
            summary['macro_f1'] += .01
            (path/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'metric differs'):
                runner._verify_quality(path,pred,reference)


if __name__ == '__main__': unittest.main()
