"""Section 9: reproducible validation-only inference and isolated GPU benchmarks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
import json
import os
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
import benchmark
import dataset
import inference
import train
from eval import CLASS_NAMES, compute_metrics, read_pred, save_predictions

METHODS = {'I00':('single view',1,'fp32'), 'I01':('original + horizontal flip, mean probability',2,'fp32'),
           'I02':('five spatial crops, mean probability',5,'fp32'),
           'I07':('scalar temperature fitted on I00 validation',1,'fp32'),
           'I08':('single view, CUDA FP16 autocast',1,'amp-fp16')}
SCOPE = 'resident normalized 256x256 tensor -> GPU crop/views -> model -> aggregation/calibration -> softmax; excludes disk/CPU preprocessing/H2D and offline temperature fitting'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_context():
    configs = ab.configs_for(ab.active_plan())
    cfg = configs['T04']
    summary, _, _ = ab.verify_ablation(cfg, configs['T00'])
    metadata = json.loads((train.run_dir(cfg)/'model.json').read_text())
    return cfg, summary, metadata


def gpu_processes():
    result = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],
                            capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError('Cannot audit GPU compute processes: '+result.stderr)
    others = []
    for line in result.stdout.splitlines():
        fields = [v.strip() for v in line.split(',')]
        if fields and fields[0].isdigit() and int(fields[0]) != os.getpid():
            others.append(line)
    return others


def session_paths(session_id):
    if not session_id or Path(session_id).name != session_id or session_id in {'.','..'}:
        raise ValueError('session_id must be a basename')
    return train.ROOT/'runs/inference'/session_id


def _verify_quality(path, prediction_path, reference):
    summary = json.loads((path/'summary.json').read_text())
    pred = read_pred(str(prediction_path))
    if pred.filenames.tolist() != reference.Filename.tolist() or not np.array_equal(pred.y_true,reference.Label.to_numpy()):
        raise ValueError('Inference validation filename/label order mismatch')
    np.testing.assert_array_equal(np.load(path/'val_filenames.npy'),pred.filenames)
    np.testing.assert_array_equal(np.load(path/'val_labels.npy'),pred.y_true)
    logits = np.load(path/'final_logits.npy')
    np.testing.assert_allclose(inference.apply_temperature(logits,1.),pred.probs,atol=1e-6,rtol=1e-5)
    metrics = compute_metrics(pred.y_true,pred.y_pred,pred.probs)
    for key in ('macro_f1','top1','ece','nll','balanced_acc'):
        if not np.isclose(metrics[key],summary[key],atol=1e-7,rtol=1e-6):
            raise ValueError(f'Cached metric differs from prediction CSV: {key}')
    if not summary['quality_complete'] or summary['test_evaluated'] or summary['n'] != len(reference):
        raise ValueError('Incomplete or invalid validation result')
    return summary, metrics


def _save_quality(path, prediction_path, names, labels, logits, method):
    path.mkdir(parents=True,exist_ok=True)
    probs = inference.apply_temperature(logits,1.)
    save_predictions(prediction_path,names,labels,probs)
    np.save(path/'final_logits.npy',logits)
    np.save(path/'val_filenames.npy',np.asarray(names,dtype=str))
    np.save(path/'val_labels.npy',labels)
    m = compute_metrics(labels,probs.argmax(1),probs)
    summary = {key:float(m[key]) for key in ('macro_f1','top1','ece','nll','balanced_acc')}
    summary.update({'method':method,'n':len(names),'quality_complete':True,'test_evaluated':False})
    train.write_json(path/'summary.json',summary)


def collect_session(session_id='I_T04_seed0', *, cfg=None):
    cfg = source_context()[0] if cfg is None else cfg
    root = session_paths(session_id)
    if not (root/'protocol.json').is_file():
        return pd.DataFrame([{'method':m,'status':'not_started'} for m in METHODS])
    protocol = json.loads((root/'protocol.json').read_text())
    for path, digest in protocol['source_hashes'].items():
        if sha(path) != digest:
            raise ValueError('Inference source checkpoint/config changed since this session')
    reference = pd.read_csv(Path(cfg.labels_dir)/f'val_subset{cfg.fold}.csv')
    rows, class_rows, latency_rows = [], [], []
    for method,(name,k,dtype) in METHODS.items():
        path = root/method
        row = {'method':method,'description':name,'k_views':k,'dtype':dtype,'source_id':cfg.exp_id,
               'seed':cfg.seed,'aggregation':'mean probability' if k > 1 else 'single logits',
               'status':'pending','test_evaluated':False}
        prediction = Path(cfg.pred_dir)/f'{method}_{session_id}_seed{cfg.seed}_val.csv'
        if (path/'summary.json').is_file():
            summary, metrics = _verify_quality(path,prediction,reference)
            if summary['method'] != method:
                raise ValueError('Cached method ID mismatch')
            row.update(summary)
            row['status'] = 'quality_complete'
            row['prediction_path'] = str(prediction)
            for label,name in enumerate(CLASS_NAMES):
                class_rows.append({'method':method,'class_id':label,'class_name':name,
                                   **{key:float(metrics[key][label]) for key in ('precision','recall','f1')},
                                   'support':int(metrics['support'][label])})
        latency_path = path/'latency.json'
        if latency_path.is_file():
            measured = json.loads(latency_path.read_text())
            for item in measured:
                if item['n'] < 50 or item['warmup'] < 10 or len(item['samples_ms']) != item['n']:
                    raise ValueError('Invalid cached latency protocol')
                values = np.asarray(item['samples_ms'])
                if not np.isfinite(values).all() or (values <= 0).any():
                    raise ValueError('Invalid latency samples')
                for percentile in (50,95,99):
                    if not np.isclose(item[f'p{percentile}'],np.percentile(values,percentile)):
                        raise ValueError('Latency percentile disagrees with samples')
                latency_rows.append({'method':method,**{k:v for k,v in item.items() if k != 'samples_ms'}})
                if item['batch'] == 1:
                    row.update({f'latency_{key}_ms':item[key] for key in ('p50','p95','p99')})
                    row['realtime_p95_le_100ms'] = item['p95'] <= 100
                else:
                    row['throughput_images_per_s'] = item['images_per_s']
            if row['status'] == 'quality_complete' and {x['batch'] for x in measured} == {1,protocol['throughput_batch']}:
                row['status'] = 'complete'
        rows.append(row)
    frame = pd.DataFrame(rows)
    if (root/'I07/summary.json').is_file():
        calibration = json.loads((root/'temperature.json').read_text())
        uncal = np.load(root/'I00/final_logits.npy')
        calibrated = np.load(root/'I07/final_logits.npy')
        np.testing.assert_allclose(calibrated,uncal.astype(np.float64)/calibration['T'],atol=1e-9)
        np.testing.assert_array_equal(uncal.argmax(1),calibrated.argmax(1))
        if calibration['fit_split'] != 'val' or calibration['n'] != len(reference):
            raise ValueError('Calibration must fit the full validation split')
    base = frame[frame.method == 'I00'].iloc[0]
    if 'macro_f1' in frame:
        frame['delta_macro_f1_pp'] = (frame.macro_f1-base.get('macro_f1',np.nan))*100
    if 'latency_p50_ms' in frame:
        frame['relative_p50_cost'] = frame.latency_p50_ms/base.get('latency_p50_ms',np.nan)
    frame.to_csv(root/'inference_results.csv',index=False)
    pd.DataFrame(class_rows).to_csv(root/'per_class_val.csv',index=False)
    pd.DataFrame(latency_rows).to_csv(root/'latency_results.csv',index=False)
    all_complete = bool((frame.status == 'complete').all())
    selection = {'section9_complete':all_complete,'selection_split':'val','source_id':cfg.exp_id,
                 'seed':cfg.seed,'final_configuration_locked':False,'test_evaluated':False,
                 'single_seed_only':True,'latency_budget_p95_ms':100}
    if all_complete:
        ranking = frame.sort_values(['macro_f1','nll','latency_p95_ms','method'],ascending=[False,True,True,True])
        realtime = ranking[ranking.realtime_p95_le_100ms]
        selection.update({'quality_candidate':ranking.iloc[0].method,
                          'realtime_candidate':None if realtime.empty else realtime.iloc[0].method,
                          'tie_breaking':'macro-F1 descending, NLL ascending, p95 ascending, method ID'})
    train.write_json(root/'selection.json',selection)
    lines = ['# Phần 9 — ConvNeXt-Tiny T04, validation', '',
             f'Nguồn: {cfg.exp_id}/seed{cfg.seed}. Toàn bộ {len(reference)} ảnh val; test chưa đánh giá.',
             'Nhiệt độ fit trên chính validation: kết quả hiệu chuẩn val chưa phải ước lượng độc lập trên test.',
             'Latency: '+SCOPE, '',
             '| ID | Trạng thái | Macro-F1 (%) | ECE | NLL | p95 batch 1 (ms) |',
             '|---|---|---:|---:|---:|---:|']
    for row in frame.to_dict('records'):
        def show(key,scale=1):
            value = row.get(key,np.nan)
            return '—' if pd.isna(value) else f'{value*scale:.5f}'
        lines.append(f'| {row["method"]} | {row["status"]} | {show("macro_f1",100)} | {show("ece")} | {show("nll")} | {show("latency_p95_ms")} |')
    lines += ['', 'ConvNeXt dùng LayerNorm: Conv–BN fusion không áp dụng; I08 đo AMP/FP16 thật.',
              'Ứng viên phần 9 chưa khóa cấu hình chung kết; final/baseline nhiều seed thuộc phần 10.']
    if all_complete:
        lines.append(f'Ứng viên chất lượng: {selection["quality_candidate"]}; ứng viên p95≤100 ms: {selection["realtime_candidate"]}.')
    (root/'inference_report.md').write_text('\n'.join(lines)+'\n')
    return frame


def run_session(session_id='I_T04_seed0', *, run_quality=True, run_latency=True, device='cuda:0',
                val_batch=64, throughput_batch=32, num_workers=2, warmup=10, iters=100):
    cfg, source_summary, metadata = source_context()
    if min(val_batch,throughput_batch) < 1 or throughput_batch == 1 or num_workers < 0:
        raise ValueError('Invalid validation/throughput batch or workers')
    if warmup < 10 or iters < 50:
        raise ValueError('Require >=10 warmup and >=50 measured samples')
    root = session_paths(session_id)
    if not run_quality and not run_latency:
        return collect_session(session_id,cfg=cfg)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'session.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This inference session is already running') from exc
        device = torch.device(device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Full section 9 requires CUDA for I08 AMP and GPU latency')
        source = train.run_dir(cfg)
        sources = [source/n for n in ('best.pt','config.json','model.json','summary.json','split_hashes.json')]
        protocol = {'session_id':session_id,'source_id':cfg.exp_id,'seed':cfg.seed,
                    'source_hashes':{str(p):sha(p) for p in sources},'config':asdict(cfg),
                    'methods':METHODS,'val_batch':val_batch,'throughput_batch':throughput_batch,
                    'num_workers':num_workers,'warmup':warmup,'iters':iters,
                    'temperature_fit':'validation NLL, golden-section log(T) in [-6,6], include T=1',
                    'device':str(device),'scope':SCOPE,'test_evaluated':False,
                    'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(device)}
        protocol['timm'] = importlib.metadata.version('timm')
        protocol['implementation_sha256'] = {p.name:sha(p) for p in
            (Path(__file__),Path(inference.__file__),Path(benchmark.__file__))}
        # JSON round-trip normalizes method tuples for immutable comparisons.
        protocol = json.loads(json.dumps(protocol))
        protocol_path = root/'protocol.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Session source/protocol changed; retain it or use a new SESSION_ID')
        train.write_json(protocol_path,protocol)
        train.write_json(root/'environment.json',train.environment_info(device))
        train.set_seed(cfg.seed,cfg.deterministic)
        torch.set_num_threads(4)
        model = train.model_utils.build_model(metadata['resolved_name'],pretrained=False,init='scratch',
                                              num_classes=9,drop_rate=cfg.drop_rate,cache_dir=cfg.cache_dir)
        state = torch.load(source/'best.pt',map_location='cpu',weights_only=True)
        if state['epoch'] != source_summary['best_epoch']:
            raise ValueError('Loaded checkpoint epoch differs from the selected validation epoch')
        model.load_state_dict(state['model'],strict=True)
        model = model.to(device).float().eval()
        prep = metadata['preprocessing']
        interpolation = {'bicubic':InterpolationMode.BICUBIC,'bilinear':InterpolationMode.BILINEAR}[prep['interpolation']]
        transform = transforms.Compose([transforms.Resize(prep['resize_size'],interpolation=interpolation),
                        transforms.CenterCrop(prep['resize_size']),transforms.ToTensor(),
                        transforms.Normalize(prep['mean'],prep['std'])])
        reference = pd.read_csv(Path(cfg.labels_dir)/f'val_subset{cfg.fold}.csv')
        loader = dataset.make_loader(reference,cfg.images_dir,transform,val_batch,False,
                                     num_workers=num_workers,seed=cfg.seed+1)
        for method in METHODS:
            path = root/method
            prediction = Path(cfg.pred_dir)/f'{method}_{session_id}_seed{cfg.seed}_val.csv'
            if run_quality and not (path/'summary.json').is_file():
                print(f'{method}: evaluating all {len(reference)} validation images',flush=True)
                started = time.perf_counter()
                if method == 'I07':
                    logits = np.load(root/'I00/final_logits.npy')
                    labels = np.load(root/'I00/val_labels.npy')
                    names = np.load(root/'I00/val_filenames.npy').tolist()
                    temperature = inference.fit_temperature(logits,labels)
                    calibrated = logits.astype(np.float64)/temperature
                    if not np.array_equal(calibrated.argmax(1),logits.argmax(1)):
                        raise ValueError('Scalar temperature changed class ranking')
                    before = compute_metrics(labels,logits.argmax(1),inference.apply_temperature(logits,1.))
                    after = compute_metrics(labels,calibrated.argmax(1),inference.apply_temperature(calibrated,1.))
                    if after['nll'] > before['nll'] + 1e-9:
                        raise ValueError('Calibration increased validation NLL')
                    train.write_json(root/'temperature.json',{'T':temperature,'fit_split':'val','n':len(labels),
                        'nll_before':before['nll'],'nll_after':after['nll'],
                        'ece_before':before['ece'],'ece_after':after['ece'],'accuracy_unchanged':True,
                        'fit_method':protocol['temperature_fit']})
                    logits = calibrated
                else:
                    names,labels,logits = inference.predict_method(model,loader,device,method,img_size=cfg.img_size)
                if names != reference.Filename.tolist() or not np.array_equal(labels,reference.Label.to_numpy()):
                    raise ValueError('Loader order differs from original validation split')
                if method == 'I00':
                    original = np.load(source/'val_logits.npy')
                    np.testing.assert_allclose(inference.apply_temperature(logits,1.),
                                               inference.apply_temperature(original,1.),atol=2e-6,rtol=1e-4)
                    train.write_json(root/'baseline_equivalence.json',{
                        'max_abs_logit_difference':float(np.max(np.abs(logits-original))),
                        'source_macro_f1':source_summary['val']['macro_f1'],'probabilities_match':True})
                _save_quality(path,prediction,names,labels,logits,method)
                print(f'{method}: validation complete, {time.perf_counter()-started:.1f}s',flush=True)
            if (path/'summary.json').exists():
                _verify_quality(path,prediction,reference)
        if run_latency:
            others = gpu_processes()
            if others:
                raise RuntimeError('Latency requires an isolated GPU. Other compute processes: '+ '; '.join(others))
            train.write_json(root/'gpu_isolation.json',{'other_compute_processes':others,
                                                      'checked_pid':os.getpid(),'scope':'nvidia-smi compute processes'})
            temperature = json.loads((root/'temperature.json').read_text())['T']
            for method,(_,k,dtype) in METHODS.items():
                path = root/method
                if (path/'latency.json').exists():
                    continue
                reports = []
                for batch in (1,throughput_batch):
                    if gpu_processes():
                        raise RuntimeError('Another GPU compute process appeared; stop latency measurement')
                    x = torch.randn(batch,3,prep['resize_size'],prep['resize_size'],device=device)
                    def pipeline():
                        return inference.method_logits(model,x,method,img_size=cfg.img_size,temperature=temperature).softmax(1)
                    measured = benchmark.pipeline_latency(pipeline,device=device,batch_size=batch,img_size=cfg.img_size,
                         input_size=prep['resize_size'],dtype=dtype,k_views=k,warmup=warmup,iters=iters,scope=SCOPE)
                    reports.append(measured)
                    print(f'{method}: batch {batch}, p95 {measured["p95"]:.3f} ms, {measured["images_per_s"]:.1f} images/s',flush=True)
                if gpu_processes():
                    raise RuntimeError('Other compute processes detected; discard this incomplete benchmark')
                train.write_json(path/'latency.json',reports)
        result = collect_session(session_id,cfg=cfg)
        print(result[['method','status','macro_f1','ece','nll']].to_string(index=False),flush=True)
        return result


def plot_session(session_id='I_T04_seed0', *, cfg=None):
    import matplotlib.pyplot as plt
    cfg = source_context()[0] if cfg is None else cfg
    root = session_paths(session_id)
    frame = collect_session(session_id,cfg=cfg)
    output = Path(cfg.curves_dir)/'inference'/session_id
    output.mkdir(parents=True,exist_ok=True)
    if 'latency_p95_ms' in frame:
        completed = frame[frame.status == 'complete']
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(completed.latency_p95_ms,completed.macro_f1*100)
        for row in completed.itertuples():
            ax.annotate(row.method,(row.latency_p95_ms,row.macro_f1*100),xytext=(4,4),textcoords='offset points')
        ax.axvline(100,color='red',linestyle='--',label='p95 budget 100 ms')
        ax.set(xlabel='Latency p95, batch 1 (ms)',ylabel='Validation macro-F1 (%)')
        ax.legend(); fig.tight_layout()
        fig.savefig(output/'f1_vs_latency_p95.png',dpi=180)
        plt.close(fig)
    if (root/'temperature.json').exists():
        fig, axes = plt.subplots(1,2,figsize=(10,4))
        for method,ax in zip(('I00','I07'),axes):
            pred = read_pred(str(Path(cfg.pred_dir)/f'{method}_{session_id}_seed{cfg.seed}_val.csv'))
            confidence = pred.probs.max(1)
            correct = pred.y_pred == pred.y_true
            xs, ys = [], []
            for left,right in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
                mask = (confidence > left) & (confidence <= right)
                if mask.any():
                    xs.append(confidence[mask].mean()); ys.append(correct[mask].mean())
            ax.plot([0,1],[0,1],'--',color='gray'); ax.plot(xs,ys,'o-')
            ax.set(title=method,xlabel='Mean confidence per bin',ylabel='Accuracy per bin',xlim=(0,1),ylim=(0,1))
        fig.tight_layout(); fig.savefig(output/'reliability_before_after.png',dpi=180); plt.close(fig)
    return output


def run_in_subprocess(session_id='I_T04_seed0', **options):
    command = [sys.executable,str(Path(__file__).resolve()),'--session',session_id,'--options-json',json.dumps(options)]
    env = dict(os.environ,HF_HUB_DISABLE_IMPLICIT_TOKEN='1',PYTHONUNBUFFERED='1')
    log_path = train.ROOT/'runs/notebook_logs'/f'{session_id}_{time.time_ns()}.log'
    log_path.parent.mkdir(parents=True,exist_ok=True)
    print('Log:',log_path)
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
    if result:
        raise RuntimeError(f'Inference subprocess failed ({result}); inspect {log_path}')
    return collect_session(session_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',default='I_T04_seed0')
    parser.add_argument('--options-json',default='{}')
    parser.add_argument('--read-only',action='store_true')
    args = parser.parse_args()
    options = json.loads(args.options_json)
    if args.read_only:
        options.update(run_quality=False,run_latency=False)
    frame = run_session(args.session,**options)
    if (session_paths(args.session)/'protocol.json').exists():
        plot_session(args.session)
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
