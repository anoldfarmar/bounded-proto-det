"""Portable CostFilter orchestration; delegates all math to recorded source."""
import argparse
import csv
from datetime import datetime
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import ROOT, CODE, config, verify_dataset, read_csv, sha
from paths import load_paths, checkpoint_for
import torch

HERE = ROOT / 'vendor/costfilter/experiments/insplad_fullview_dinomaly2_costfilter'
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('recorded_costfilter', HERE / 'run.py')
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

def check_dtd(paths):
    from PIL import Image
    files = sorted(Path(paths['dtd_images']).glob('*/*.jpg'))
    if len(files) != 5640 or len({p.parent.name for p in files}) != 47:
        raise ValueError('DTD requires 47 folders / 5640 images')
    for p in files:
        with Image.open(p) as image:
            image.verify()
    return files

def make_reference(job, data_root, baseline_run):
    """Adapt stored A-score paths only; do not synthesize missing baseline scores."""
    target = job / 'baseline_reference/results'
    target.mkdir(parents=True)
    source = Path(baseline_run) / 'results/image_scores_005000.csv' if baseline_run else ROOT / 'results/recorded/a5000_scores.csv'
    rows = read_csv(source)
    for row in rows:
        relative = row.get('relative_path')
        if not relative:
            relative = row['category'] + '/' + row['path'].split('/' + row['category'] + '/', 1)[1]
        row['path'] = str(Path(data_root) / relative)
    ex.write_csv(target / 'image_scores_005000.csv', rows)
    ex.dump(target.parent / 'reference_origin.json', dict(source=str(source), source_sha256=sha(source), note='Path bridge only; not a training-run claim.'))
    return target.parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'smoke', 'train', 'evaluate'])
    parser.add_argument('--paths', type=Path)
    parser.add_argument('--baseline-run', type=Path, help='Use a new formal A5000 run instead of recorded weights/scores')
    parser.add_argument('--run', type=Path, help='Evaluate a completed portable CostFilter run')
    parser.add_argument('--resume', type=Path, help='Resume an interrupted portable CostFilter run')
    parser.add_argument('--smoke', action='store_true', help='Evaluate only 10 fixed test images; not formal performance')
    args = parser.parse_args()
    if args.resume and args.mode != 'train':
        parser.error('--resume is only for train')
    if args.run and args.mode != 'evaluate':
        parser.error('--run is only for evaluate')
    if args.smoke and args.mode != 'evaluate':
        parser.error('--smoke is only for evaluate; use mode smoke for a two-update test')
    paths = load_paths(args.paths)
    base_cfg = config(paths_file=args.paths)
    dataset = verify_dataset(base_cfg)
    existing = args.resume or args.run
    if existing:
        job = existing.resolve()
        cfg = json.loads((job / 'config.json').read_text())
        if json.loads((job / 'identity.json').read_text())['smoke']:
            raise ValueError('Smoke checkpoints cannot be formal results')
        if cfg['weights'] != paths['encoder_weights'] or cfg['data_path'] != paths['data_root']:
            raise ValueError('Resume/evaluate a portable run with its original paths configuration')
        if sha(cfg['baseline_checkpoint']) != cfg['baseline_checkpoint_sha256']:
            raise ValueError('Run baseline checkpoint changed')
        saved = json.loads((job / 'portable_source_sha256.json').read_text())
        if any(sha(ROOT / name) != value for name, value in saved.items()):
            raise ValueError('Portable runtime changed after this run')
        environment = sys.version + '\n' + subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        if environment != (job / 'environment.txt').read_text():
            raise ValueError('Resume/evaluate environment differs from original run')
        if args.resume:
            check_dtd(paths)
    else:
        checkpoint = checkpoint_for(paths, args.baseline_run)
        cfg = json.loads((ROOT / 'configs/costfilter.json').read_text())
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
        job = Path(paths['output_root']) / 'costfilter' / f'{args.mode}_{stamp}_{os.getpid()}'
        job.mkdir(parents=True, exist_ok=False)
        cfg.update(results_root=str(job.parent), dinomaly2_repo=str(CODE), baseline_checkpoint=str(checkpoint), baseline_checkpoint_sha256=sha(checkpoint), weights=paths['encoder_weights'], data_path=paths['data_root'], dtd_images=paths['dtd_images'])
        cfg['baseline_run'] = str(make_reference(job, cfg['data_path'], args.baseline_run))
        if args.mode != 'evaluate':
            dtd = check_dtd(paths)
            ex.write_csv(job / 'dtd_manifest.csv', [dict(relative_path=p.relative_to(paths['dtd_images']).as_posix(), size_bytes=p.stat().st_size) for p in dtd])
        ex.dump(job / 'config.json', cfg)
        ex.dump(job / 'identity.json', dict(smoke=args.mode == 'smoke', purpose='portable-reproduction', original_baseline=args.baseline_run is None))
        source_paths = [p for folder in ['scripts', 'vendor'] for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py', '.sh', '.json'}]
        ex.dump(job / 'portable_source_sha256.json', {p.relative_to(ROOT).as_posix():sha(p) for p in source_paths})
        (job / 'environment.txt').write_text(sys.version + '\n' + subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    ex.dump(job / 'portable_data_check.json', dataset)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler(job / 'portable_log.txt'), logging.StreamHandler()])
    logging.info('Portable CostFilter output: %s', job)
    torch.set_num_threads(2)
    ex.seed_all(cfg['seed'])
    try:
        if args.mode == 'check':
            ex.preview_synthetic(cfg, job)
            ex.status(job, 'complete', training_started=False)
            print('CPU checks passed; no training:', job)
            return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for smoke/train/evaluate')
        device = torch.device(f'cuda:{cfg["cuda"]}')
        torch.cuda.set_device(device)
        if args.mode in ['train', 'smoke']:
            ex.preview_synthetic(cfg, job)
            filter_checkpoint = ex.train(cfg, job, device, smoke=args.mode == 'smoke', resume=bool(args.resume))
        elif args.run:
            filter_checkpoint = job / 'checkpoints/final.pt'
        else:
            filter_checkpoint = Path(paths['costfilter_checkpoint'])
            if sha(filter_checkpoint) != 'b7912bb140756bb3012dfb8af301898af246c94835be9ec8626bcd076025fab4':
                raise ValueError('Recorded CostFilter checkpoint mismatch; use --run for a new trained filter')
        # A 10-image check of the original formal model remains a smoke result.
        # The original evaluate() protects against treating smoke weights as formal;
        # TestImages alone is restricted here, without changing checkpoint identity.
        if args.smoke:
            original_dataset = ex.TestImages
            ex.TestImages = lambda root, smoke=False: original_dataset(root, smoke=True)
        result = ex.evaluate(cfg, job, filter_checkpoint, device, smoke=args.mode == 'smoke')
        if args.smoke:
            result['smoke'] = True
            result['limitations'].append('10-image execution check only; not formal performance.')
            ex.dump(job / 'summary.json', result)
            (job / 'REPORT.md').write_text('# Smoke execution check only\n\n10 fixed test images using formal pretrained checkpoints. Metrics are not benchmark results.\nSee summary.json for the smoke flag and image count.\n')
        if args.mode == 'train':
            (job.parent / 'LATEST_TRAIN.txt').write_text(str(job) + '\n')
        ex.status(job, 'complete', mode=args.mode, smoke=args.smoke or args.mode == 'smoke')
        print('Completed:', job)
    except BaseException as error:
        ex.status(job, 'failed', reason=str(error))
        raise

if __name__ == '__main__':
    main()
