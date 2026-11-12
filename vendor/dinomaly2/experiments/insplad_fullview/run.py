"""Fresh full-view A5000 training followed by optional NCut scoring, five classes only."""
import argparse
from argparse import Namespace
from collections import Counter
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT))
import numpy as np
from PIL import Image
import scipy
from sklearn.metrics import roc_curve
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset,DataLoader,Dataset
from dinomaly_2D import build_model,setup_seed
from insplad_adapter import FAULT5_COUNTS,ImageAnomalyDataset,validate_insplad,write_data_manifest,write_csv,evaluate_insplad,metric_values
from optimizers import StableAdamW
from reproduce_mvtec_b import digest,save_checkpoint,save_eval_weights
from utils import WarmupCosineScheduler,global_cosine_hm_percent,cal_anomaly_maps,get_gaussian_kernel
from view import transform,SPEC
from ncut import cosine_graph,ncut_regions

SHARED=['reproduce_mvtec_b.py','dinomaly_2D.py','dataset.py','utils.py','models/vit_encoder.py',
        'models/uad.py','models/vision_transformer.py','optimizers/StableAdamW.py',
        'dinov2/models/vision_transformer.py','dinov2/layers/attention.py','dinov2/layers/block.py','insplad_adapter.py']
VARIANTS=['full_top1','ncut_seed_fixed_k','ncut_seed_mix_0p5','ncut_center_fixed_k','ncut_center_mix_0p5']


def now():return datetime.now(ZoneInfo('Asia/Shanghai'))


def dump(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def read_csv(path):
    import csv
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def status(job,state,**extra):dump(job/'run_status.json',dict(status=state,updated_at=now().isoformat(),**extra))


def configuration():
    cfg=json.loads((HERE/'config.json').read_text())
    if cfg['categories']!=list(FAULT5_COUNTS):raise ValueError('Only the five requested categories are allowed')
    if (cfg['training_steps'],cfg['scheduler_steps'],cfg['seed'],cfg['batch_size'],cfg['workers'],cfg['eval_every'])!=(5000,15000,1,32,4,2500):
        raise ValueError('Keep the matched A5000 training protocol; change requires an explicit new experiment')
    if (cfg['image_size'],cfg['crop_size'],cfg['lr'],cfg['weight_decay'])!=(392,392,.002,.01):raise ValueError('Unexpected input or optimizer recipe')
    frozen=json.loads((HERE/'core_sha256.json').read_text())
    for name in SHARED:
        if digest(PROJECT/name)!=frozen[name]:raise ValueError('Shared baseline source changed: '+name)
    if digest(cfg['weights'])!=cfg['weights_sha256']:raise ValueError('Local encoder weights changed')
    refs=read_csv(HERE/'normal_references.csv')
    if len(refs)!=50 or len({r['path'] for r in refs})!=50 or Counter(r['category'] for r in refs)!={c:10 for c in FAULT5_COUNTS}:raise ValueError('Normal references changed')
    for r in refs:
        if r['split']!='train' or r['label']!='0' or '/train/good/' not in r['path']:raise ValueError('Only normal TRAIN references can calibrate NCut')
        if digest(r['path'])!=r['image_sha256']:raise ValueError('Normal reference content changed')
    return cfg,refs


def snapshot(job):
    names=SHARED+[str(p.relative_to(PROJECT)) for p in HERE.iterdir() if p.is_file()]
    for name in names:
        out=job/'source'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/name,out)
    dump(job/'source_sha256.json',{n:digest(PROJECT/n) for n in names})


def model_args(cfg,output,smoke):
    return Namespace(**cfg,mode='smoke' if smoke else 'train',dataset='insplad',insplad_subset='fault5',
        total_iters=cfg['training_steps'],stop_after=2 if smoke else cfg['training_steps'],
        output=str(output),input_view=SPEC,initialization='fresh_bottleneck_and_decoder_no_training_checkpoint',
        foreground_loss_weighting=False,mixed_precision=False)


class IndexedTrain(Dataset):
    def __init__(self,data):self.data=data
    def __len__(self):return len(self.data)
    def __getitem__(self,index):
        image,label=self.data[index];return image,label,index


class ReferenceImages(Dataset):
    def __init__(self,refs):self.refs=refs;self.transform=transform()
    def __len__(self):return len(self.refs)
    def __getitem__(self,index):
        with Image.open(self.refs[index]['path']) as image:return self.transform(image.convert('RGB'))


def train(cfg,job,device,smoke):
    # All validation has finished; resetting here also isolates training from smoke/check RNG use.
    setup_seed(cfg['seed']);output=job/'results';output.mkdir()
    args=model_args(cfg,output,smoke);model=build_model(args,'cpu');model.encoder.requires_grad_(False);model=model.to(device)
    dump(output/'config.json',vars(args));dump(output/'input_view.json',SPEC)
    train_data=ConcatDataset([ImageAnomalyDataset(Path(cfg['data_path'])/c,transform(),phase='train') for c in cfg['categories']])
    if len(train_data)!=9006:raise ValueError('Expected 9006 normal training images')
    test=[(c,ImageAnomalyDataset(Path(cfg['data_path'])/c,transform(),smoke=smoke)) for c in cfg['categories']]
    loader=DataLoader(IndexedTrain(train_data),batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['workers'],drop_last=True)
    optimizer=StableAdamW([{'params':model.bottleneck[0].parameters(),'lr':cfg['nb_lr']},
        {'params':model.bottleneck[1].parameters()},{'params':model.decoder.parameters()}],
        lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'],eps=cfg['optimizer_eps'])
    scheduler=WarmupCosineScheduler(optimizer,warmup_epochs=cfg['warmup_steps'],total_epochs=cfg['scheduler_steps'],final_ratio=cfg['lr_decay_ratio'])
    trainable=list(model.bottleneck.parameters())+list(model.decoder.parameters())
    initial_rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    reference=json.loads((HERE/'initialization_reference.json').read_text())
    if initial_rng!=reference['cpu_rng_before_loader_iteration_sha256']:raise ValueError('Initialization/loader RNG no longer matches original A recipe')
    encoder_probe=next(model.encoder.parameters()).detach().clone() if smoke else None
    torch.cuda.reset_peak_memory_stats(device);step=0;started=time.monotonic();model.train()
    logging.info('从头训练Bottleneck/Decoder；完整视野训练图9006张；训练停止步数%d；LR曲线尺度%d。',args.stop_after,cfg['scheduler_steps'])
    while step<args.stop_after:
        for image,label,indices in loader:
            if label.any():raise ValueError('Training must only contain normal images')
            if step==0:
                if indices.tolist()!=reference['first_batch_indices']:raise ValueError('First shuffled batch differs from matched A')
                dump(job/'initialization_check.json',dict(first_batch_indices=indices.tolist(),
                    cpu_rng_before_loader_iteration_sha256=initial_rng,matched_A_sample_order=True,
                    trainable_weights_fresh=True,encoder_frozen=True,input_shape=list(image.shape),input_view=SPEC))
            before=model.bottleneck[0][0].weight.detach().clone() if smoke and step==1 else None
            en,de=model(image.to(device));loss=global_cosine_hm_percent(en,de,p=min(cfg['ll_ratio']*step/1000,cfg['ll_ratio']),factor=cfg['ll_factor'])
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training loss at step {step}')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(trainable,cfg['clip_grad_norm'],error_if_nonfinite=True)
            optimizer.step();scheduler.step();step+=1
            if before is not None and torch.equal(before,model.bottleneck[0][0].weight):raise RuntimeError('Second update did not change bottleneck weights')
            if step==1 or step%100==0 or step==args.stop_after:
                logging.info('step %d/%d loss %.6f grad_norm %.6f lr %s elapsed %.1fs peak_VRAM %.2fGiB',step,args.stop_after,float(loss.detach()),float(norm),scheduler.get_last_lr(),time.monotonic()-started,torch.cuda.max_memory_allocated(device)/2**30)
                status(job,'running',phase='training',step=step,total_steps=args.stop_after)
            if step%cfg['backup_every']==0 or step%cfg['eval_every']==0 or step==args.stop_after:
                save_checkpoint(output/'last.pt',model,optimizer,scheduler,step,args)
            if step%cfg['eval_every']==0 or step==args.stop_after:
                save_eval_weights(output,model,step,args)
                status(job,'running',phase='A_evaluation',step=step)
                report=evaluate_insplad(model,test,args,output,step,device)
                if report['test_image_count']!=(10 if smoke else 2656):raise ValueError('Unexpected A test scope')
                model.train()
            if step>=args.stop_after:break
    if smoke:
        if not torch.equal(encoder_probe,next(model.encoder.parameters())) or any(p.grad is not None for p in model.encoder.parameters()):raise RuntimeError('Frozen encoder was updated')
        state=torch.load(output/'checkpoints/step_000002.pt',map_location='cpu',weights_only=True)
        model.bottleneck.load_state_dict(state['bottleneck'],strict=True);model.decoder.load_state_dict(state['decoder'],strict=True)
        dump(job/'smoke_training_check.json',dict(passed=True,steps=2,encoder_frozen=True,bottleneck_updated=True,checkpoint_reloaded=True,
            matched_first_batch=True,peak_VRAM_GiB=torch.cuda.max_memory_allocated(device)/2**30,elapsed_seconds=time.monotonic()-started))
    dump(job/'training_complete.json',dict(completed=True,step=step,smoke=smoke,checkpoint=str(output/f'checkpoints/step_{step:06d}.pt'),input_view=SPEC))
    del model,optimizer,scheduler,loader,en,de,loss,image;torch.cuda.empty_cache()
    return step


def load_trained(cfg,job,step,device,smoke):
    path=job/f'results/checkpoints/step_{step:06d}.pt'
    state=torch.load(path,map_location='cpu',weights_only=True);a=state['config']
    if state['step']!=step or state['weights_sha256']!=cfg['weights_sha256']:raise ValueError('Checkpoint identity mismatch')
    if a['experiment_id']!=cfg['experiment_id'] or a['input_view']!=SPEC or a['categories']!=cfg['categories']:raise ValueError('Require this full-view-trained five-class model, not old crop/17-class weights')
    if (a['mode']=='smoke')!=smoke or (not smoke and step!=5000):raise ValueError('Do not treat smoke checkpoint as A5000')
    setup_seed(cfg['seed']);model=build_model(Namespace(**a),'cpu')
    model.bottleneck.load_state_dict(state['bottleneck'],strict=True);model.decoder.load_state_dict(state['decoder'],strict=True)
    model.requires_grad_(False);del state
    return model.to(device).eval(),path


def features(model,tokens):return model.encoder.norm(tokens)[:,1+model.encoder.num_register_tokens:].cpu().numpy()


def calibration(cfg,refs,model,device):
    key_data=dict(encoder=cfg['weights_sha256'],view=SPEC,refs=refs,ncut=cfg['ncut'],feature_layer=cfg['feature_layer_index'],
        batch_size=cfg['batch_size'],torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
        sources={n:digest(HERE/n) for n in ('view.py','ncut.py','run.py')},core={n:digest(PROJECT/n) for n in SHARED})
    key=hashlib.sha256(json.dumps(key_data,sort_keys=True).encode()).hexdigest();folder=Path(cfg['results_root'])/'references'/key[:20]
    if (folder/'calibration.json').exists():
        ref=json.loads((folder/'calibration.json').read_text())
        if ref['key']!=key:raise ValueError('Calibration cache mismatch')
        return folder,ref
    folder.mkdir(parents=True,exist_ok=True);captured={}
    handle=model.encoder.blocks[cfg['feature_layer_index']].register_forward_hook(lambda _m,_x,y:captured.update(tokens=y.detach()))
    pairs=[];upper=np.triu_indices(784,1)
    try:
        with torch.inference_mode():
            for batch in DataLoader(ReferenceImages(refs),batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['workers']):
                model(batch.to(device));tokens=features(model,captured.pop('tokens'))
                for token in tokens:pairs.append(cosine_graph(token)[upper])
    finally:handle.remove()
    pool=np.concatenate(pairs)
    if np.ptp(pool)<1e-8:raise ValueError('Collapsed reference graph')
    tau=float(np.quantile(pool,cfg['ncut']['similarity_quantile']))
    ref=dict(key=key,key_data=key_data,threshold=tau,reference_count=50,reference_split='train/good',test_used_for_calibration=False,input_view=SPEC)
    dump(folder/'calibration.json',ref);shutil.copy2(HERE/'normal_references.csv',folder/'normal_references.csv')
    return folder,ref


def fixed_topk(amap,mask,k=655):
    ids=torch.where(mask.flatten())[0]
    if not len(ids):raise ValueError('Empty scoring mask')
    values=torch.sort(amap.flatten()[ids],descending=True).values[:min(k,len(ids))]
    return float(values.mean()),len(values)


def metrics(rows,key):
    result=metric_values([dict(label=r['label'],score=r[key]) for r in rows]);labels=np.array([r['label'] for r in rows]);scores=np.array([r[key] for r in rows])
    fpr,tpr,thresholds=roc_curve(labels,scores,drop_intermediate=False);i=int(np.where(tpr>=.95)[0][0]);pred=scores>=thresholds[i]
    result.update({'FPR@95TPR':float(fpr[i]),'TPR@95TPR':float(tpr[i]),'threshold@95TPR':float(thresholds[i]),
        'TP@95TPR':int((pred&(labels==1)).sum()),'FP@95TPR':int((pred&(labels==0)).sum())})
    return result


def score_ncut(cfg,refs,job,step,device,smoke):
    model,checkpoint=load_trained(cfg,job,step,device,smoke);ref_dir,ref=calibration(cfg,refs,model,device)
    branch=job/'ncut';branch.mkdir(exist_ok=True);output=branch/f'step_{step:06d}_{now():%Y%m%d_%H%M%S}_{os.getpid()}'
    output.mkdir(exist_ok=False);dump(output/'input_view.json',SPEC);dump(output/'reference_used.json',dict(path=str(ref_dir),key=ref['key'],tau=ref['threshold']))
    logging.info('A%d同一检查点评估全图/NCut；NCut训练0步，tau=%.8f。输出：%s',step,ref['threshold'],output)
    baseline={r['path']:r for r in read_csv(job/f'results/image_scores_{step:06d}.csv')}
    captured={};handle=model.encoder.blocks[cfg['feature_layer_index']].register_forward_hook(lambda _m,_x,y:captured.update(tokens=y.detach()))
    rows=[];started=time.monotonic();max_error=0.
    try:
        with torch.inference_mode():
            for category in cfg['categories']:
                data=ImageAnomalyDataset(Path(cfg['data_path'])/category,transform(),smoke=smoke)
                loader=DataLoader(data,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['workers'])
                kernel=get_gaussian_kernel(kernel_size=5,sigma=4).to(device);done=0
                for batch,labels,paths in loader:
                    batch=batch.to(device);en,de=model(batch);amap,_=cal_anomaly_maps(en,de,batch.shape[-1])
                    amap=kernel(F.interpolate(amap,size=256,mode='bilinear',align_corners=False))[:,0];tokens=features(model,captured.pop('tokens'))
                    full_scores=torch.sort(amap.flatten(1),descending=True,dim=1).values[:,:655].mean(1)
                    for i,path in enumerate(paths):
                        if path not in baseline or int(baseline[path]['label'])!=int(labels[i]):raise ValueError('A/NCut test samples differ')
                        full=float(full_scores[i]);error=abs(full-float(baseline[path]['score']));max_error=max(max_error,error)
                        if error>1e-6:raise ValueError('Cannot reproduce same-checkpoint A score: '+path)
                        seed,center,info,_=ncut_regions(tokens[i],cfg['grid'],cfg['ncut'],ref['threshold'])
                        row=dict(category=category,path=path,label=int(labels[i]),defect_type=Path(path).parent.name,full_top1=full,graph_density=info['graph_density'])
                        for key,mask,detail in [('seed',seed,info['main']),('center',center,info['center'])]:
                            region=F.interpolate(torch.as_tensor(mask,dtype=torch.float32,device=device)[None,None],size=256,mode='nearest')[0,0].bool()
                            value,used=fixed_topk(amap[i],region)
                            row['ncut_'+key+'_fixed_k']=value;row['ncut_'+key+'_mix_0p5']=.5*(full+value)
                            row['ncut_'+key+'_fallback']=detail['fallback'];row['ncut_'+key+'_area_fraction']=float(mask.mean());row['ncut_'+key+'_k_used']=used
                        rows.append(row)
                    done+=len(paths)
                    if done==len(data) or done%(32*10)==0:logging.info('NCut %s %d/%d；总计%d elapsed %.1fs',category,done,len(data),len(rows),time.monotonic()-started)
                    status(job,'running',phase='ncut_evaluation',step=step,completed_images=len(rows))
    finally:handle.remove()
    if len(rows)!=(10 if smoke else 2656):raise ValueError('Incomplete NCut test')
    class_rows=[];defect_rows=[];variants={}
    for variant in VARIANTS:
        values=[]
        for category in cfg['categories']:
            part=[r for r in rows if r['category']==category];m=metrics(part,variant);values.append(m);class_rows.append(dict(variant=variant,category=category,**m))
            for defect in sorted({r['defect_type'] for r in part if r['label']}):
                subset=[r for r in part if not r['label'] or r['defect_type']==defect]
                defect_rows.append(dict(variant=variant,category=category,defect_type=defect,**metrics(subset,variant)))
        variants[variant]=dict(mean={k:float(np.mean([v[k] for v in values])) for k in ('I-AUROC','I-AP','I-F1','FPR@95TPR')})
    summary=dict(experiment_id=cfg['experiment_id'],checkpoint_step=step,smoke=smoke,categories=cfg['categories'],test_image_count=len(rows),
        training_view=SPEC,evaluation_view=SPEC,ncut_training_steps=0,variants=variants,main_variant='ncut_seed_mix_0p5',diagnostic_variant='ncut_center_mix_0p5',
        tau=ref['threshold'],max_A_score_error=max_error,checkpoint_sha256=digest(checkpoint),inference_seconds=time.monotonic()-started,
        delta_vs_A_percentage_points=100*(variants['ncut_seed_mix_0p5']['mean']['I-AUROC']-variants['full_top1']['mean']['I-AUROC']),
        limitations=['TokenCut-style adaptation, not original TokenCut reproduction.','Same exploratory test split; no test-driven parameter selection.','Foreground candidates are not semantic ground truth.'])
    dump(output/'summary.json',summary);write_csv(output/'image_scores.csv',rows,list(rows[0]))
    write_csv(output/'metrics.csv',class_rows,list(class_rows[0]));write_csv(output/'defect_metrics.csv',defect_rows,list(defect_rows[0]))
    dump(output/'quality.json',{c:{key:dict(Counter(r['ncut_'+key+'_fallback'] or 'none' for r in rows if r['category']==c)) for key in ('seed','center')} for c in cfg['categories']})
    dump(job/'A_vs_NCut.json',summary);(job/'NCUT_RESULTS.txt').write_text(str(output)+'\n')
    if not smoke:(Path(cfg['results_root'])/'LATEST_NCUT.txt').write_text(str(output)+'\n')
    return summary,output


def report(job,cfg,step,smoke,score=None):
    a=json.loads((job/f'results/summary_{step:06d}.json').read_text())
    lines=['# 完整视野 Dinomaly2-B：A 与 NCut','',
        '仅2步/10张smoke检查，不能作为正式性能。' if smoke else '五类正常训练9006张，整图缩放392×392，训练5000步；测试2656张。',
        '训练与测试均无裁剪、无补边。冻结DINOv2；Bottleneck/Decoder从头训练；无前景加权训练。',
        '学习率保留旧A15000步曲线前缀，训练实际停止于5000步；不是运行15000步。','',
        f'原始全图五类宏平均 I-AUROC：{a["mean"]["I-AUROC"]:.4%}。','']
    if score:
        lines+=['| 评分 | 宏平均I-AUROC | 宏平均FPR95 |','|---|---:|---:|']
        for key in VARIANTS:
            m=score['variants'][key]['mean'];lines.append(f'| {key} | {m["I-AUROC"]:.4%} | {m["FPR@95TPR"]:.4%} |')
        lines+=['','NCut主结果为seed_mix_0p5，中心选区仅作诊断；不按测试成绩切换方法。',
            'NCut没有训练参数或额外5000步；本表各评分使用同一个已训练模型和同一异常图。']
    else:lines+=['本次只训练A，NCut未运行。可用ncut模式对本目录最终检查点补充评分。']
    (job/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','check','smoke','train','train-only','ncut'])
    parser.add_argument('--run',type=Path,help='ncut only: existing completed full-view training run')
    cli=parser.parse_args();cfg,refs=configuration();root=Path(cfg['results_root'])
    if cli.mode=='plan':print(json.dumps(dict(config=cfg,input_view=SPEC,train_mode='train A5000 then NCut; train-only skips NCut'),ensure_ascii=False,indent=2));return
    if cli.run is not None and cli.mode!='ncut':parser.error('--run is only for ncut; training always starts fresh')
    if cli.mode=='ncut':
        job=(cli.run or Path((root/'LATEST_TRAIN.txt').read_text().strip())).resolve(strict=True)
        recorded=json.loads((job/'config.json').read_text())
        if recorded!=cfg:raise ValueError('Run configuration differs from this protocol')
        complete=json.loads((job/'training_complete.json').read_text())
        if complete['smoke'] or complete['step']!=5000:raise ValueError('A5000 training has not completed')
    else:
        branch={'check':'checks','smoke':'smoke','train':'runs','train-only':'runs'}[cli.mode]
        job=root/branch/f'{cli.mode}_seed1_{now():%Y%m%d_%H%M%S}_{os.getpid()}'
        job.mkdir(parents=True,exist_ok=False);snapshot(job);dump(job/'config.json',cfg);dump(job/'input_view.json',SPEC)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(job/'log.txt'),logging.StreamHandler()])
    logging.info('完整视野独立实验；模式%s；结果：%s',cli.mode,job)
    status(job,'running',phase='validation',mode=cli.mode)
    try:
        counts=validate_insplad(cfg['data_path'],'fault5');dump(job/'dataset_counts.json',counts)
        write_data_manifest(cfg['data_path'],cfg['categories'],job/'dataset_manifest.csv')
        if digest(job/'dataset_manifest.csv')!=cfg['dataset_manifest_sha256']:raise ValueError('Dataset manifest changed')
        dump(job/'provenance.json',dict(input_view=SPEC,encoder_sha256=cfg['weights_sha256'],
            normal_reference_csv_sha256=digest(HERE/'normal_references.csv'),torch=torch.__version__,python=platform.python_version(),
            model_initialization='fresh trainable modules; only pretrained encoder loaded',training_steps=5000,scheduler_steps=15000))
        if cli.mode=='check':
            subprocess.run([sys.executable,str(HERE/'test_view.py')],cwd=HERE,check=True)
            subprocess.run([sys.executable,str(HERE/'test_ncut.py')],cwd=HERE,check=True)
            model=build_model(model_args(cfg,job/'results',False),'cpu');del model
        else:
            if not torch.cuda.is_available() or cfg['cuda']>=torch.cuda.device_count():raise RuntimeError('Training/smoke/NCut scoring requires a GPU')
            device=torch.device(f"cuda:{cfg['cuda']}");torch.cuda.set_device(device);torch.set_num_threads(2)
            if cli.mode in ('train','train-only'):(root/'LATEST_TRAIN.txt').write_text(str(job)+'\n')
            step=5000 if cli.mode=='ncut' else train(cfg,job,device,cli.mode=='smoke')
            if cli.mode in ('train','train-only'):(root/'LATEST_A_RESULTS.txt').write_text(str(job/'results')+'\n')
            score=None
            if cli.mode in ('train','smoke','ncut'):score,_=score_ncut(cfg,refs,job,step,device,cli.mode=='smoke')
            report(job,cfg,step,cli.mode=='smoke',score)
        status(job,'completed',mode=cli.mode)
        print('完成。结果目录：',job,flush=True)
    except BaseException as error:
        logging.exception('实验中断或失败');status(job,'failed',mode=cli.mode,reason=str(error));raise


if __name__=='__main__':main()
