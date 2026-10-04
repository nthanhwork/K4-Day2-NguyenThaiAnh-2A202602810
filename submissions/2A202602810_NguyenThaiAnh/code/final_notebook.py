"""Section 10: immutable recipes, three seeds and one test forward per checkpoint.

F01 = T04 training + I07; T00 = B03 training + I00. Seed 0 aliases the
completed source runs. Other seeds train independently with test disabled.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import ablation_notebook as ab
import backbone_notebook as nbtrain
import dataset
import inference
import inference_notebook as inftrain
import train
from eval import CLASS_NAMES, SCALARS, compute_metrics, read_pred, save_predictions

FINAL_ROOT = train.ROOT / 'runs/finalization/ConvNeXt_Final_v1'
EVALUATOR = Path(__file__).resolve().parents[3] / 'eval.py'
SEEDS = (0,1,2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_path(root=None):
    return FINAL_ROOT if root is None else Path(root).resolve()


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime_versions():
    return {'python':platform.python_version(),'torch':str(torch.__version__),
            'numpy':str(np.__version__),'timm':importlib.metadata.version('timm'),
            'torchvision':importlib.metadata.version('torchvision')}


@contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process owns {path}') from exc
        yield


def source_context():
    frame = inftrain.collect_session()
    selection = json.loads((inftrain.session_paths('I_T04_seed0')/'selection.json').read_text())
    if not selection['section9_complete'] or selection['quality_candidate'] != 'I07' or selection['realtime_candidate'] != 'I07':
        raise ValueError('Expected completed section 9 selecting T04 + I07')
    final_cfg, _, metadata = inftrain.source_context()
    baseline_cfg = ab._baseline_config()
    checkpoint_paths = [train.run_dir(c)/name for c in (baseline_cfg,final_cfg)
                        for name in ('best.pt','config.json','model.json','summary.json')]
    session = inftrain.session_paths('I_T04_seed0')
    evidence_paths = checkpoint_paths + [session/name for name in
        ('protocol.json','selection.json','temperature.json','I07/summary.json','I07/latency.json')]
    latency = json.loads((session/'I07/latency.json').read_text())
    batch1 = next(item for item in latency if item['batch'] == 1)
    evidence = {str(p):sha(p) for p in evidence_paths}
    return baseline_cfg,final_cfg,metadata,evidence,batch1


def build_spec():
    baseline,final,metadata,evidence,latency = source_context()
    split_hashes = json.loads((train.run_dir(final)/'split_hashes.json').read_text())
    entries = []
    for role,base,method in (('T00',baseline,'I00'),('F01',final,'I07')):
        for seed in SEEDS:
            cfg = replace(base,resume=False,save_test_predictions=False) if seed == 0 else replace(
                base,exp_id=role,seed=seed,pred_dir=str(Path(final.pred_dir)/'training'),
                resume=False,save_test_predictions=False)
            entries.append({'role':role,'seed':seed,'source_cfg':asdict(cfg),
                            'seed0_reuse':seed == 0,'source_id':cfg.exp_id,'method':method})
    code_paths = [Path(__file__),Path(train.__file__),Path(dataset.__file__),Path(inference.__file__),
                  Path(train.model_utils.__file__),Path(train.losses.__file__),Path(nbtrain.__file__),EVALUATOR]
    return {'schema':1,'seeds':list(SEEDS),'entries':entries,'baseline':'B03 recipe + I00, same ConvNeXt backbone',
            'final':'T04 recipe + I07','selection_split':'val','preprocessing':metadata['preprocessing'],
            'pretrained_tag':metadata['pretrained_cfg'].get('tag'),'class_names':CLASS_NAMES,
            'labels_dir':final.labels_dir,'split_hashes':split_hashes,'prediction_dir':final.pred_dir,
            'split_sizes':{split:len(pd.read_csv(Path(final.labels_dir)/f'{split}_subset{final.fold}.csv'))
                           for split in ('train','val','test')},'fold':final.fold,
            'optimizer':'AdamW; backbone/head LR; bias/norm no decay',
            'scheduler':f'warmup {final.warmup_epochs} epoch(s) then cosine; advance only on successful updates',
            'checkpoint_selection':'highest validation macro-F1; earliest epoch on ties',
            'inference':{'final':'I07','baseline':'I00','k_views':1,'aggregation':'single logits',
                         'dtype':'FP32','fusion':False,'ensemble':False,
                         'temperature':'per final seed, validation NLL; log(T) in [-6,6]; include T=1',
                         'test_policy':'one raw single-view forward per checkpoint; calibrated/uncalibrated share logits'},
            'representative_latency':latency,'latency_checkpoint':'T04/seed0, same F01/I07 pipeline',
            'evidence_sha256':evidence,'implementation_sha256':{str(p):sha(p) for p in code_paths},
            'runtime_versions':runtime_versions()}


def load_manifest(root=None):
    root = root_path(root)
    path = root/'manifest.json'
    manifest = json.loads(path.read_text())
    if sha(path) != (root/'manifest.sha256').read_text().strip():
        raise ValueError('Locked manifest was modified')
    if manifest['spec']['runtime_versions'] != runtime_versions():
        raise ValueError('Python/package versions differ from the locked runtime')
    for collection in ('implementation_sha256','evidence_sha256'):
        for filename,digest in manifest['spec'][collection].items():
            if sha(filename) != digest:
                raise ValueError(f'Locked code or validation evidence changed: {filename}')
    for name,digest in manifest['spec']['split_hashes'].items():
        if sha(Path(manifest['spec']['labels_dir'])/name) != digest:
            raise ValueError(f'Locked CSV changed: {name}')
    return manifest


def lock_configuration(root=None):
    root = root_path(root)
    with file_lock(root/'manifest.lock'):
        if (root/'manifest.json').exists():
            return load_manifest(root)
        pred_dir = Path(source_context()[1].pred_dir)
        for role in ('F01','T00','F01uncal'):
            # Existing test output may not be relabelled as a new pre-test lock.
            if list(pred_dir.glob(f'{role}_seed*_test.csv')):
                raise ValueError('Test output already exists before configuration lock')
        spec = json.loads(json.dumps(build_spec()))
        manifest = {'locked_at_utc':now(),'configuration_locked_before_test':True,'spec':spec}
        train.write_json(root/'manifest.json',manifest)
        (root/'manifest.sha256').write_text(sha(root/'manifest.json')+'\n')
        return manifest


def entry_for(manifest,role,seed):
    matches = [e for e in manifest['spec']['entries'] if e['role'] == role and e['seed'] == seed]
    if len(matches) != 1:
        raise ValueError('Role/seed is not in the locked plan')
    return matches[0]


def product_path(manifest,role,seed,split):
    return Path(manifest['spec']['prediction_dir'])/f'{role}_seed{seed}_{split}.csv'


def entry_root(root,role,seed):
    return root/role/f'seed{seed}'


def metrics(labels,probs):
    result = compute_metrics(labels,probs.argmax(1),probs)
    return {key:float(result[key]) for key in SCALARS} | {'n':int(result['n'])}


def check_prediction(path,reference,probs):
    pred = read_pred(str(path))
    if pred.filenames.tolist() != reference.Filename.tolist() or not np.array_equal(pred.y_true,reference.Label.to_numpy()):
        raise ValueError('Final filename/label order differs from original split')
    np.testing.assert_allclose(pred.probs,probs,atol=1e-6,rtol=1e-5)
    return pred


def source_record(cfg):
    source = train.run_dir(cfg)
    paths = [source/name for name in ('best.pt','config.json','model.json','summary.json',
                                      'val_logits.npy','val_labels.npy','val_filenames.npy')]
    return {str(p):sha(p) for p in paths}


def verify_prepared(root,manifest,role,seed):
    entry = entry_for(manifest,role,seed)
    cfg = train.Config(**entry['source_cfg'])
    nbtrain.verify_outputs(cfg)  # Checks training-only raw predictions in a separate directory.
    path = entry_root(root,role,seed)
    prepared = json.loads((path/'prepared.json').read_text())
    if prepared['manifest_sha256'] != sha(root/'manifest.json') or prepared['source_sha256'] != source_record(cfg):
        raise ValueError('Prepared checkpoint or manifest changed')
    logits = np.load(train.run_dir(cfg)/'val_logits.npy')
    labels = np.load(train.run_dir(cfg)/'val_labels.npy')
    reference = pd.read_csv(Path(cfg.labels_dir)/f'val_subset{cfg.fold}.csv')
    temp = prepared['temperature']
    if role == 'T00' and temp != 1.:
        raise ValueError('Baseline must remain uncalibrated I00')
    if role == 'F01' and not np.isclose(temp,inference.fit_temperature(logits,labels),atol=1e-12,rtol=1e-12):
        raise ValueError('Final temperature differs from the locked validation fitting protocol')
    probs = inference.apply_temperature(logits,temp)
    check_prediction(product_path(manifest,role,seed,'val'),reference,probs)
    actual = metrics(labels,probs)
    for key in SCALARS:
        if not np.isclose(actual[key],prepared['val'][key],atol=1e-7):
            raise ValueError('Prepared validation metrics differ from stored logits')
    return prepared


def prepare_entry(role,seed,root=None):
    root = root_path(root)
    manifest = load_manifest(root)
    entry = entry_for(manifest,role,seed)
    cfg = train.Config(**entry['source_cfg'])
    path = entry_root(root,role,seed)
    with file_lock(path/'prepare.lock'):
        if (path/'prepared.json').exists():
            return verify_prepared(root,manifest,role,seed)
        summary,_,metadata = nbtrain.verify_outputs(cfg)
        if metadata['pretrained_cfg'].get('tag') != manifest['spec']['pretrained_tag']:
            raise ValueError('Pretrained tag differs from locked recipe')
        if metadata['preprocessing'] != manifest['spec']['preprocessing']:
            raise ValueError('Preprocessing differs from locked recipe')
        split = json.loads((train.run_dir(cfg)/'split_hashes.json').read_text())
        if split != manifest['spec']['split_hashes']:
            raise ValueError('Source training split differs from the manifest')
        logits = np.load(train.run_dir(cfg)/'val_logits.npy')
        labels = np.load(train.run_dir(cfg)/'val_labels.npy')
        names = np.load(train.run_dir(cfg)/'val_filenames.npy').tolist()
        temperature = inference.fit_temperature(logits,labels) if role == 'F01' else 1.
        uncal,cal = inference.apply_temperature(logits,1.),inference.apply_temperature(logits,temperature)
        np.testing.assert_array_equal(uncal.argmax(1),cal.argmax(1))
        before,after = metrics(labels,uncal),metrics(labels,cal)
        if after['nll'] > before['nll']+1e-9:
            raise ValueError('Validation NLL increased during calibration')
        save_predictions(product_path(manifest,role,seed,'val'),names,labels,cal)
        if role == 'F01':
            save_predictions(product_path(manifest,'F01uncal',seed,'val'),names,labels,uncal)
        prepared = {'role':role,'seed':seed,'source_id':cfg.exp_id,'seed0_reuse':entry['seed0_reuse'],
                    'best_epoch':summary['best_epoch'],'temperature':temperature,'temperature_fit_split':'val',
                    'manifest_sha256':sha(root/'manifest.json'),'source_sha256':source_record(cfg),
                    'val':after,'val_uncalibrated':before,'test_evaluated':False,'prepared_at_utc':now()}
        train.write_json(path/'prepared.json',prepared)
        return verify_prepared(root,manifest,role,seed)


def train_entry(role,seed,root=None,*,subprocess_training=True):
    root = root_path(root)
    manifest = load_manifest(root)
    entry = entry_for(manifest,role,seed)
    cfg = train.Config(**entry['source_cfg'])
    if entry['seed0_reuse']:
        nbtrain.verify_outputs(cfg)
        print(f'{role}/seed0: reuse {cfg.exp_id}/seed0',flush=True)
    else:
        if subprocess_training:
            nbtrain.run_in_subprocess(cfg,planned_group=10)
        else:
            nbtrain.execute_training(cfg,planned_group=10)
    return prepare_entry(role,seed,root)


def seal_checkpoint_set(root=None):
    root = root_path(root)
    manifest = load_manifest(root)
    with file_lock(root/'seal.lock'):
        records = {}
        for entry in manifest['spec']['entries']:
            role,seed = entry['role'],entry['seed']
            prepared = verify_prepared(root,manifest,role,seed)
            path = entry_root(root,role,seed)/'prepared.json'
            records[f'{role}/seed{seed}'] = {'prepared_sha256':sha(path),'source_sha256':prepared['source_sha256'],
                                          'temperature':prepared['temperature']}
        seal = {'manifest_sha256':sha(root/'manifest.json'),'checkpoints':records,
                'all_seeds_trained_and_calibrated_before_test':True}
        path = root/'checkpoint_set.json'
        if path.exists():
            if json.loads(path.read_text()) != seal:
                raise ValueError('Checkpoint set already sealed; do not replace models after test')
        else:
            if any(root.glob('*/seed*/test_started.json')):
                raise ValueError('Test marker exists before checkpoint-set seal')
            train.write_json(path,seal)
        return seal


def atomic_numpy(path,value):
    tmp = path.with_suffix('.npy.tmp')
    with tmp.open('wb') as stream: np.save(stream,value)
    tmp.replace(path)


def test_raw_loader(cfg,manifest):
    """Only called by test_once after all checkpoints/temperatures are sealed."""
    prep = manifest['spec']['preprocessing']
    mode = {'bicubic':InterpolationMode.BICUBIC,'bilinear':InterpolationMode.BILINEAR}[prep['interpolation']]
    transform = transforms.Compose([transforms.Resize(prep['resize_size'],interpolation=mode),
                    transforms.CenterCrop(prep['resize_size']),transforms.ToTensor(),
                    transforms.Normalize(prep['mean'],prep['std'])])
    frame = pd.read_csv(Path(cfg.labels_dir)/f'test_subset{cfg.fold}.csv')
    return frame,dataset.make_loader(frame,cfg.images_dir,transform,cfg.batch_size,False,
                                     num_workers=cfg.num_workers,seed=cfg.seed+1)


def load_model(cfg,device,manifest):
    metadata = json.loads((train.run_dir(cfg)/'model.json').read_text())
    model = train.model_utils.build_model(metadata['resolved_name'],pretrained=False,init='scratch',
                                           num_classes=9,drop_rate=cfg.drop_rate,cache_dir=cfg.cache_dir)
    state = torch.load(train.run_dir(cfg)/'best.pt',map_location='cpu',weights_only=True)
    summary = json.loads((train.run_dir(cfg)/'summary.json').read_text())
    if state['epoch'] != summary['best_epoch'] or state['weights'] != summary['weights']:
        raise ValueError('Selected checkpoint epoch/weights mismatch')
    model.load_state_dict(state['model'],strict=True)
    return model.to(device).float().eval()


def read_raw_test(path,manifest):
    complete = json.loads((path/'raw_complete.json').read_text())
    if complete['forward_passes'] != 1 or complete['n'] != manifest['spec']['split_sizes']['test']:
        raise ValueError('Raw test cache violates the one-forward/full-split protocol')
    for name,digest in complete['sha256'].items():
        if sha(path/name) != digest:
            raise ValueError('Cached raw test output changed')
    logits = np.load(path/'test_logits.npy')
    labels = np.load(path/'test_labels.npy')
    names = np.load(path/'test_filenames.npy').tolist()
    reference = pd.read_csv(Path(manifest['spec']['labels_dir'])/f'test_subset{manifest["spec"]["fold"]}.csv')
    if names != reference.Filename.tolist() or not np.array_equal(labels,reference.Label.to_numpy()):
        raise ValueError('Raw test filename/label order mismatch')
    if logits.shape != (len(reference),9) or not np.isfinite(logits).all():
        raise ValueError('Invalid raw test logits')
    return names,labels,logits,reference


def export_test_from_cache(root,manifest,role,seed):
    path = entry_root(root,role,seed)
    marker = json.loads((path/'test_started.json').read_text())
    if marker['manifest_sha256'] != sha(root/'manifest.json') or marker['checkpoint_set_sha256'] != sha(root/'checkpoint_set.json'):
        raise ValueError('Test start marker refers to a different locked configuration')
    if marker['role'] != role or marker['seed'] != seed or marker['method'] != entry_for(manifest,role,seed)['method']:
        raise ValueError('Test start marker role/seed/method mismatch')
    names,labels,logits,reference = read_raw_test(path,manifest)
    prepared = verify_prepared(root,manifest,role,seed)
    cal = inference.apply_temperature(logits,prepared['temperature'])
    uncal = inference.apply_temperature(logits,1.)
    np.testing.assert_array_equal(cal.argmax(1),uncal.argmax(1))
    for target,probs in [(role,cal)]+([('F01uncal',uncal)] if role == 'F01' else []):
        prediction = product_path(manifest,target,seed,'test')
        if prediction.exists():
            check_prediction(prediction,reference,probs)
        else:
            save_predictions(prediction,names,labels,probs)
    result = {'role':role,'seed':seed,'test':metrics(labels,cal),'test_uncalibrated':metrics(labels,uncal),
              'val':prepared['val'],'temperature':prepared['temperature'],
              'test_forward_passes':1,'manifest_sha256':sha(root/'manifest.json'),
              'checkpoint_set_sha256':sha(root/'checkpoint_set.json'),
              'raw_complete_sha256':sha(path/'raw_complete.json'),'completed_at_utc':now()}
    if (path/'test_summary.json').exists():
        saved = json.loads((path/'test_summary.json').read_text())
        for key in ('manifest_sha256','checkpoint_set_sha256','raw_complete_sha256','test','test_uncalibrated'):
            if saved[key] != result[key]:
                raise ValueError('Completed test summary differs from locked cached results')
        return saved
    train.write_json(path/'test_summary.json',result)
    return result


def test_once(role,seed,root=None,*,device='cuda:0'):
    root = root_path(root)
    manifest = load_manifest(root)
    seal_checkpoint_set(root)  # Requires all six prepared entries before any test loader/forward.
    entry = entry_for(manifest,role,seed)
    cfg = train.Config(**entry['source_cfg'])
    path = entry_root(root,role,seed)
    with file_lock(path/'test.lock'):
        marker = path/'test_started.json'
        if marker.exists():
            if not (path/'raw_complete.json').exists():
                raise RuntimeError('Test started but raw cache is incomplete; do not forward test again or remove its marker')
            print(f'{role}/seed{seed}: reuse cached test logits; no new test forward',flush=True)
            return export_test_from_cache(root,manifest,role,seed)
        if (path/'raw_complete.json').exists() or any(product_path(manifest,target,seed,'test').exists()
                                                     for target in (role,'F01uncal') if role == 'F01' or target == role):
            raise ValueError('Test artifacts exist without a start marker')
        device = torch.device(device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no test marker created')
        train.set_seed(cfg.seed,cfg.deterministic)
        torch.set_num_threads(4)
        model = load_model(cfg,device,manifest)
        reference,loader = test_raw_loader(cfg,manifest)
        with marker.open('x') as stream:
            json.dump({'started_at_utc':now(),'role':role,'seed':seed,'method':entry['method'],
                       'manifest_sha256':sha(root/'manifest.json'),'checkpoint_set_sha256':sha(root/'checkpoint_set.json'),
                       'raw_forward':'I00 FP32 once; apply pre-fitted scalar T to same logits'},stream,indent=2)
        print(f'{role}/seed{seed}: one locked test forward on {len(reference)} images',flush=True)
        names,labels,logits = inference.predict_method(model,loader,device,'I00',img_size=cfg.img_size)
        if names != reference.Filename.tolist() or not np.array_equal(labels,reference.Label.to_numpy()):
            raise ValueError('Test output differs from original split order')
        for name,value in (('test_logits.npy',logits),('test_labels.npy',labels),
                           ('test_filenames.npy',np.asarray(names,dtype=str))):
            atomic_numpy(path/name,value)
        train.write_json(path/'raw_complete.json',{'n':len(names),'forward_passes':1,
                'sha256':{name:sha(path/name) for name in ('test_logits.npy','test_labels.npy','test_filenames.npy')}})
        return export_test_from_cache(root,manifest,role,seed)


def collect_progress(root=None):
    root = root_path(root)
    if not (root/'manifest.json').exists():
        return pd.DataFrame([{'role':r,'seed':s,'status':'configuration_not_locked'} for r in ('T00','F01') for s in SEEDS])
    manifest = load_manifest(root)
    rows = []
    for entry in manifest['spec']['entries']:
        role,seed = entry['role'],entry['seed']
        cfg = train.Config(**entry['source_cfg'])
        path = entry_root(root,role,seed)
        row = {'role':role,'seed':seed,'source_id':cfg.exp_id,'seed0_reuse':entry['seed0_reuse'],
               'training_status':nbtrain.training_status(cfg),'status':'training_needed'}
        if row['training_status'] == 'complete': row['status'] = 'validation_preparation_needed'
        if (path/'prepared.json').exists():
            prepared = verify_prepared(root,manifest,role,seed)
            row.update(status='ready_for_test',temperature=prepared['temperature'],val_macro_f1=prepared['val']['macro_f1'],
                       val_top1=prepared['val']['top1'],val_ece=prepared['val']['ece'])
        if (path/'test_started.json').exists(): row['status'] = 'test_started'
        if (path/'test_summary.json').exists():
            summary = export_test_from_cache(root,manifest,role,seed)
            row.update(status='test_complete',**{f'test_{key}':summary['test'][key] for key in SCALARS})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(root/'progress.csv',index=False)
    return frame


def score_and_grade(root=None):
    root = root_path(root)
    manifest = load_manifest(root)
    seal_checkpoint_set(root)
    frame = collect_progress(root)
    if not (frame.status == 'test_complete').all():
        raise ValueError('Complete all six locked test predictions before score/grade')
    spec = manifest['spec']
    pred = Path(spec['prediction_dir'])
    for role in ('F01','T00','F01uncal'):
        expected = {str(product_path(manifest,role,seed,'test')) for seed in spec['seeds']}
        if {str(p) for p in pred.glob(f'{role}_seed*_test.csv')} != expected:
            raise ValueError('Evaluator glob contains extra/missing seeds')
    if {str(p) for p in pred.glob('F01_seed*_val.csv')} != {
            str(product_path(manifest,'F01',seed,'val')) for seed in spec['seeds']}:
        raise ValueError('Final validation glob contains extra/missing seeds')
    output = root/'eval_out'
    output.mkdir(parents=True,exist_ok=True)
    labels = Path(spec['labels_dir'])
    common = ['--test-csv',str(labels/f'test_subset{spec["fold"]}.csv'),'--labels',str(labels/'labels.csv'),'--out',str(output)]
    commands = [(role+'_score',[sys.executable,str(EVALUATOR),'score','--pred',str(pred/f'{role}_seed*_test.csv'),
                                '--tag',role,*common]) for role in ('F01','T00')]
    commands.append(('grade',[sys.executable,str(EVALUATOR),'grade','--final',str(pred/'F01_seed*_test.csv'),
          '--baseline',str(pred/'T00_seed*_test.csv'),'--uncal',str(pred/'F01uncal_seed*_test.csv'),
          '--final-val',str(pred/'F01_seed*_val.csv'),'--val-csv',str(labels/f'val_subset{spec["fold"]}.csv'),
          '--latency-p95-ms',str(spec['representative_latency']['p95']),'--latency-method','proper',*common]))
    for name,command in commands:
        result = subprocess.run(command,capture_output=True,text=True,cwd=EVALUATOR.parent)
        (output/(name+'.log')).write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'Official evaluator failed: {output/(name+".log")}')
    for role in ('F01','T00'):
        per_seed = pd.read_csv(output/f'{role}_per_seed.csv')
        if len(per_seed) != len(spec['seeds']): raise ValueError('Evaluator seed count mismatch')
    train.write_json(root/'completion.json',{'section10_complete':True,'n_seeds':len(spec['seeds']),
          'test_n_per_seed':spec['split_sizes']['test'],'manifest_sha256':sha(root/'manifest.json'),
          'checkpoint_set_sha256':sha(root/'checkpoint_set.json'),'evaluator_sha256':sha(EVALUATOR),
          'std_ddof':1,'eval_out':str(output),'completed_at_utc':now()})
    return output


def run_pipeline(root=None,*,run_training=True,run_test=True,run_evaluator=True):
    root = root_path(root)
    with file_lock(root/'pipeline.lock'):
        manifest = lock_configuration(root)
        if run_training:
            for entry in manifest['spec']['entries']:
                train_entry(entry['role'],entry['seed'],root)
        if run_test:
            seal_checkpoint_set(root)
            for entry in manifest['spec']['entries']:
                test_once(entry['role'],entry['seed'],root)
        if run_evaluator:
            score_and_grade(root)
        return collect_progress(root)


def run_in_subprocess(**options):
    command = [sys.executable,str(Path(__file__).resolve()),'--options-json',json.dumps(options)]
    env = dict(os.environ,HF_HUB_DISABLE_IMPLICIT_TOKEN='1',PYTHONUNBUFFERED='1')
    log_path = train.ROOT/'runs/notebook_logs'/f'final10_{time.time_ns()}.log'
    log_path.parent.mkdir(parents=True,exist_ok=True)
    print('Log:',log_path,flush=True)
    with log_path.open('w') as log:
        process = subprocess.Popen(command,cwd=train.ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
            result = process.wait()
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
            raise
        finally:
            process.stdout.close()
    if result: raise RuntimeError(f'Final pipeline failed ({result}); inspect {log_path}')
    return collect_progress(options.get('root'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--options-json',default='{}')
    parser.add_argument('--read-only',action='store_true')
    parser.add_argument('--lock-only',action='store_true')
    parser.add_argument('--train-entry',nargs=2,metavar=('ROLE','SEED'))
    args = parser.parse_args()
    options = json.loads(args.options_json)
    if args.read_only:
        frame = collect_progress(options.get('root'))
    elif args.lock_only:
        lock_configuration(options.get('root')); frame = collect_progress(options.get('root'))
    elif args.train_entry:
        lock_configuration(options.get('root'))
        train_entry(args.train_entry[0],int(args.train_entry[1]),options.get('root'))
        frame = collect_progress(options.get('root'))
    else:
        frame = run_pipeline(**options)
    print(frame.to_string(index=False),flush=True)


if __name__ == '__main__': main()
