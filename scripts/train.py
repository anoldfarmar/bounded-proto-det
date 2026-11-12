"""Portable fresh-training wrapper; reuses the unchanged recorded A5000 implementation."""
import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
from common import ROOT,experiment as ex,config,verify_dataset,references,sha
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path',type=Path)
    parser.add_argument('--paths',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--train-only',action='store_true')
    args=parser.parse_args();cfg=config(args.data_path,args.output_root,args.paths)
    if not torch.cuda.is_available():raise RuntimeError('Training requires CUDA')
    verify_dataset(cfg);refs=references(cfg);ex.validate_insplad(cfg['data_path'],'fault5')
    mode='smoke' if args.smoke else 'train';stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    job=Path(cfg['results_root'])/f'{mode}_{stamp}_{os.getpid()}';job.mkdir(parents=True,exist_ok=False)
    ex.snapshot(job);ex.dump(job/'config.json',cfg);ex.dump(job/'input_view.json',ex.SPEC)
    ex.dump(job/'portable_wrapper.json',dict(train_script_sha256=sha(Path(__file__)),common_script_sha256=sha(Path(__file__).with_name('common.py')),python=sys.version,
        note='Path handling/relative dataset identity adapted; model, loss, optimizer, schedule, input and scoring code are the unchanged recorded implementation.'))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(job/'log.txt'),logging.StreamHandler()])
    device=torch.device(f"cuda:{cfg['cuda']}");torch.cuda.set_device(device);torch.set_num_threads(2)
    try:
        ex.status(job,'running',phase='training');step=ex.train(cfg,job,device,args.smoke);score=None
        if not args.train_only:score,_=ex.score_ncut(cfg,refs,job,step,device,args.smoke)
        ex.report(job,cfg,step,args.smoke,score);ex.status(job,'completed')
        if not args.smoke:(Path(cfg['results_root'])/'LATEST_TRAIN.txt').write_text(str(job)+'\n')
        print('结果目录：',job)
    except BaseException as error:ex.status(job,'failed',reason=str(error));raise


if __name__=='__main__':main()
