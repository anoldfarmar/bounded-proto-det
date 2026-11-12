"""Re-evaluate exactly the three reported scores using externally referenced A5000 weights."""
import argparse
from argparse import Namespace
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo
from common import ROOT,experiment as ex,config,verify_dataset,references,read_csv,sha
from paths import load_paths, checkpoint_for
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

VARIANTS=['full_top1','ncut_seed_fixed_k','ncut_seed_mix_0p5']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path',type=Path)
    parser.add_argument('--paths',type=Path)
    parser.add_argument('--baseline-run',type=Path,help='Use a freshly completed formal A5000 run')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--smoke',action='store_true',help='10-image execution check only, not formal performance')
    parser.add_argument('--recalibrate',action='store_true',help='Recompute the normal-only graph threshold instead of reusing recorded calibration')
    args=parser.parse_args();stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output=args.output or Path(load_paths(args.paths)['output_root'])/'ncut'/f'eval_{stamp}_{os.getpid()}'
    output=output.resolve();cfg=config(args.data_path,output.parent,args.paths);dataset_info=verify_dataset(cfg)
    checkpoint=checkpoint_for(load_paths(args.paths),args.baseline_run)
    output.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(output/'log.txt'),logging.StreamHandler()])
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu for slow execution')
    ex.setup_seed(cfg['seed']);torch.set_num_threads(2)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);model_cfg=state['config']
    if state['step']!=5000 or model_cfg['input_view']!=ex.SPEC or model_cfg['categories']!=cfg['categories']:raise ValueError('Wrong training protocol')
    model_cfg['weights']=cfg['weights'];model=ex.build_model(Namespace(**model_cfg),'cpu')
    model.bottleneck.load_state_dict(state['bottleneck'],strict=True);model.decoder.load_state_dict(state['decoder'],strict=True)
    model.requires_grad_(False);model=model.to(device).eval();del state
    calibration=json.loads((ROOT/'provenance/ncut/calibration.json').read_text())
    if args.recalibrate:_,calibration=ex.calibration(cfg,references(cfg),model,device)
    tau=calibration['threshold'];ex.dump(output/'calibration.json',calibration)
    reference={r['relative_path']:r for r in read_csv(ROOT/'results/recorded/ncut_scores.csv')}
    captured={};hook=model.encoder.blocks[cfg['feature_layer_index']].register_forward_hook(lambda _m,_x,y:captured.update(tokens=y.detach()))
    rows=[];max_error=0.;started=time.monotonic()
    try:
        with torch.inference_mode():
            for category in cfg['categories']:
                data=ex.ImageAnomalyDataset(Path(cfg['data_path'])/category,ex.transform(),smoke=args.smoke)
                loader=DataLoader(data,batch_size=32,num_workers=cfg['workers'],shuffle=False)
                kernel=ex.get_gaussian_kernel(kernel_size=5,sigma=4).to(device)
                for batch,labels,paths in loader:
                    batch=batch.to(device);en,de=model(batch);amap,_=ex.cal_anomaly_maps(en,de,392)
                    amap=kernel(F.interpolate(amap,size=256,mode='bilinear',align_corners=False))[:,0]
                    tokens=ex.features(model,captured.pop('tokens'))
                    full_scores=torch.sort(amap.flatten(1),dim=1,descending=True).values[:,:655].mean(1)
                    for i,path in enumerate(paths):
                        seed,_,info,_=ex.ncut_regions(tokens[i],cfg['grid'],cfg['ncut'],tau)
                        mask=F.interpolate(torch.as_tensor(seed,dtype=torch.float32,device=device)[None,None],size=256,mode='nearest')[0,0].bool()
                        region,used=ex.fixed_topk(amap[i],mask);full=float(full_scores[i]);relative=Path(path).relative_to(cfg['data_path']).as_posix()
                        row=dict(relative_path=relative,category=category,label=int(labels[i]),defect_type=Path(path).parent.name,
                            full_top1=full,ncut_seed_fixed_k=region,ncut_seed_mix_0p5=.5*(full+region),
                            ncut_seed_area_fraction=float(seed.mean()),ncut_seed_k_used=used,ncut_seed_fallback=info['main']['fallback'])
                        expected=reference[relative]
                        if int(expected['label'])!=row['label']:raise ValueError('Reference label mismatch')
                        max_error=max(max_error,max(abs(row[k]-float(expected[k])) for k in VARIANTS));rows.append(row)
                logging.info('%s complete; total=%d elapsed %.1fs',category,len(rows),time.monotonic()-started)
    finally:hook.remove()
    if len(rows)!=(10 if args.smoke else 2656):raise ValueError('Incomplete evaluation')
    variants={};metrics=[]
    for variant in VARIANTS:
        values=[]
        for category in cfg['categories']:
            m=ex.metrics([r for r in rows if r['category']==category],variant);values.append(m)
            metrics.append(dict(variant=variant,category=category,**m))
        variants[variant]=dict(mean={k:float(np.mean([m[k] for m in values])) for k in ('I-AUROC','I-AP','I-F1','FPR@95TPR')})
    summary=dict(smoke=args.smoke,test_image_count=len(rows),checkpoint_step=5000,ncut_training_steps=0,variants=variants,
        max_recorded_score_error=max_error,tau=tau,input_view=ex.SPEC,dataset=dataset_info,checkpoint_sha256=sha(checkpoint),
        note='Smoke uses different batch composition; small floating-point differences are expected. Full reference comparison is reported, not used for tuning.')
    summary['reference_comparison_scope']='Original recorded scores; differences are reported, never used for selection.'
    if not args.smoke and not args.baseline_run and max_error>1e-6:raise ValueError(f'Recorded score reproduction mismatch: {max_error}')
    ex.dump(output/'summary.json',summary);ex.write_csv(output/'image_scores.csv',rows,list(rows[0]));ex.write_csv(output/'metrics.csv',metrics,list(metrics[0]))
    logging.info('完成：%s',output);print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
