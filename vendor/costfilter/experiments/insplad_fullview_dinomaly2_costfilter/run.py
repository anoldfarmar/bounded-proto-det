"""Train only CostFilter on frozen full-view Dinomaly2 A5000, five classes."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from insplad_data import CATEGORIES, COUNTS, VIEW, SyntheticNormals, TestImages, evaluation_loader, seed_worker
from model_adapter import baseline_modules, build_filter, cost_inputs, digest, load_baseline, losses, loss_terms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCORES = ['A5000_full', 'costfilter_only', 'costfilter_fused_0p05']


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai'))


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def status(job, state, **extra):
    dump(job/'run_status.json', dict(status=state, updated_at=now().isoformat(), **extra))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate(cfg, job):
    if cfg['categories'] != CATEGORIES or cfg['image_size'] != 392 or cfg['perlin_size'] != 448:
        raise ValueError('Require the five fixed categories, complete 392 input, and 448 Perlin canvas')
    if cfg['filter_coefficient'] != .05 or cfg['score_grid'] != 256 or cfg['score_ratio'] != .01:
        raise ValueError('固定评分协议发生变化，应创建新实验')
    if cfg['base_channels'] != 48 or cfg['cost_channels'] != 768 or cfg['cost_grid'] != 64:
        raise ValueError('Require the provided 48-base/768-channel/64-grid CostFilter architecture')
    if cfg['mixed_precision']:
        raise ValueError('This first controlled experiment uses float32')
    for key in ['baseline_checkpoint', 'weights']:
        actual = digest(cfg[key])
        if actual != cfg[key+'_sha256']:
            raise ValueError('文件校验失败：' + key)
    # Check the same shared sources that produced A5000; do not modify them.
    expected = json.loads((Path(cfg['baseline_run'])/'source_sha256.json').read_text())
    shared = ['dinomaly_2D.py', 'dataset.py', 'utils.py', 'models/vit_encoder.py',
              'models/uad.py', 'models/vision_transformer.py',
              'dinov2/models/vision_transformer.py', 'dinov2/layers/attention.py',
              'dinov2/layers/block.py', 'insplad_adapter.py']
    for name in shared:
        if digest(Path(cfg['dinomaly2_repo'])/name) != expected[name]:
            raise ValueError('A5000 shared source changed: ' + name)
    _, _, adapter = baseline_modules(cfg)
    counts = adapter.validate_insplad(cfg['data_path'], subset='fault5')
    adapter.write_data_manifest(cfg['data_path'], CATEGORIES, job/'dataset_manifest.csv')
    if digest(job/'dataset_manifest.csv') != cfg['dataset_manifest_sha256']:
        raise ValueError('InsPLAD 划分/路径/文件大小与已有实验不一致')
    dtd = sorted(Path(cfg['dtd_images']).glob('*/*.jpg'))
    if len(dtd) != 5640 or len({p.parent.name for p in dtd}) != 47:
        raise ValueError(f'DTD 应有47类5640图，当前找到{len(dtd)}张：{cfg["dtd_images"]}')
    # Header verification catches incomplete extraction without decoding all pixels.
    for path in dtd:
        with Image.open(path) as image:
            image.verify()
    dtd_rows = [dict(path=str(p), size_bytes=p.stat().st_size) for p in dtd]
    write_csv(job/'dtd_manifest.csv', dtd_rows)
    report = dict(counts=counts, train_normal=9006, test_normal=2254,
                  test_anomalous=402, dtd_count=len(dtd), input_view=VIEW,
                  baseline_checkpoint_sha256=cfg['baseline_checkpoint_sha256'],
                  dtd_manifest_sha256=digest(job/'dtd_manifest.csv'),
                  manifest_note='InsPLAD and DTD manifests audit paths and sizes; not per-image content hashes.')
    dump(job/'data_check.json', report)
    target = job/'source'
    source_hashes = {}
    selected = list(HERE.glob('*.py')) + list(HERE.glob('*.json')) + list(HERE.glob('*.sh')) + list(HERE.glob('*.md'))
    selected += [ROOT/'Costfilter_Dinomaly'/n for n in
                 ['unet3d_att_dino_channel_test_48_min.py', 'loss.py', 'perlin_noise.py',
                  'dataset.py', 'costfilter_dinomaly_mvtec_uni_my_train_test.py', 'utils.py']]
    selected += list((ROOT/'scripts').glob('*'))
    for path in selected:
        if path.is_file():
            dest = target/'costfilter'/path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            source_hashes[str(path)] = digest(path)
    # Snapshot the complete small Python source of the frozen backbone implementation.
    baseline_repo = Path(cfg['dinomaly2_repo'])
    baseline_paths = [baseline_repo/n for n in shared]
    for folder in ['models', 'dinov1', 'dinov2', 'beit', 'optimizers']:
        baseline_paths.extend((baseline_repo/folder).rglob('*.py'))
    for path in set(baseline_paths):
        dest = target/'dinomaly2'/path.relative_to(baseline_repo)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        source_hashes[str(path)] = digest(path)
    dump(job/'source_sha256.json', source_hashes)
    environment = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (job/'environment.txt').write_text(sys.version + '\n' + environment)
    logging.info('数据验证通过：9006正常训练图；2656测试图；DTD5640图。')


def preview_synthetic(cfg, job):
    data = SyntheticNormals(cfg)
    output = job/'synthetic_preview'
    output.mkdir(exist_ok=True)
    rows = []
    for category in CATEGORIES:
        idx = next(i for i, r in enumerate(data.rows) if r['category'] == category)
        for source in ['dtd', 'self_shuffle']:
            sample = data.item(idx, force_source=source, force_anomaly=True)
            assert sample['image'].shape == (3, 392, 392)
            assert sample['mask'].shape == (1, 392, 392)
            assert set(sample['mask'].unique().tolist()).issubset({0., 1.})
            if not sample['mask'].any():
                raise RuntimeError('诊断合成掩码为空')
            mask = (sample['mask'][0].numpy()*255).astype(np.uint8)
            color_mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
            overlay = (.65*sample['synthetic'] + .35*color_mask).astype(np.uint8)
            strip = np.concatenate([sample['original'], sample['synthetic'], overlay], axis=1)
            name = f'{category}_{source}.jpg'
            Image.fromarray(strip).save(output/name, quality=90)
            rows.append(dict(category=category, source=source, source_path=data.rows[idx]['path'],
                             mask_fraction=float(sample['mask'].mean()), preview=name))
    dump(output/'manifest.json', rows)
    import html
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>训练合成检查</title>'
        '<h1>左：完整正常图；中：人工异常；右：红色合成区域</h1>'
        '<p>仅正常训练样本。两种素材路径分别展示，不是测试集结果。</p>' +
        ''.join(f'<h3>{html.escape(r["category"])} / {r["source"]}</h3>'
                f'<img style="max-width:100%" src="{r["preview"]}">' for r in rows))


def make_loader(cfg, epoch, next_batch):
    data = SyntheticNormals(cfg, epoch=epoch)
    generator = torch.Generator().manual_seed(cfg['seed'] + epoch)
    order = torch.randperm(len(data), generator=generator).tolist()
    order = order[next_batch*cfg['batch_size']:]
    worker_generator = torch.Generator().manual_seed(cfg['seed'] + epoch + 100000)
    return DataLoader(Subset(data, order), batch_size=cfg['batch_size'], shuffle=False,
                      num_workers=cfg['workers'], pin_memory=True, drop_last=False,
                      worker_init_fn=seed_worker, generator=worker_generator)


def save_state(path, model, cfg, step, **extra):
    temp = path.with_suffix('.tmp')
    torch.save(dict(model_state_dict=model.state_dict(), config=cfg, step=step,
                    baseline_checkpoint_sha256=cfg['baseline_checkpoint_sha256'], **extra), temp)
    temp.replace(path)


def train(cfg, job, device, smoke=False, resume=False):
    baseline = load_baseline(cfg, device)
    # Independent seed: the random new filter never changes the frozen A5000.
    seed_all(cfg['seed'])
    model = build_filter(cfg).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=tuple(cfg['betas']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=5)
    ssim = losses.SSIM().to(device)
    step, epoch, next_batch, epoch_sum, epoch_count = 0, 0, 0, 0., 0
    steps = 2 if smoke else cfg['training_steps']
    checkpoints = job/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    if resume:
        saved = torch.load(checkpoints/'last.pt', map_location=device, weights_only=True)
        if saved['config'] != cfg or saved['smoke'] != smoke:
            raise ValueError('恢复训练配置或 smoke 标识不一致')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        step, epoch, next_batch = saved['step'], saved['epoch'], saved['next_batch']
        epoch_sum, epoch_count = saved['epoch_sum'], saved['epoch_count']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        if step >= steps:
            logging.info('已完成%d步，跳过训练，直接评估最终检查点。', step)
            return checkpoints/'final.pt'
    initial_encoder = next(baseline.encoder.parameters()).detach().clone() if smoke else None
    first_weight = next(model.parameters()).detach().clone() if smoke else None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    run_elapsed_before = json.loads((job/'timing.json').read_text())['training_seconds'] if resume and (job/'timing.json').exists() else 0.
    train_batches_per_epoch = math.ceil(9006/cfg['batch_size'])
    logging.info('冻结A5000；仅训练CostFilter；步数%d；batch%d；lr%.6g；类别5；float32。',
                 steps, cfg['batch_size'], cfg['lr'])
    while step < steps:
        for batch in make_loader(cfg, epoch, next_batch):
            image, mask = batch['image'].to(device), batch['mask'].to(device)
            class_ids = batch['class_id'].to(device)
            with torch.no_grad():
                en, de = baseline(image)
                volume, features, guidance, _ = cost_inputs(en, de, cfg)
            optimizer.zero_grad(set_to_none=True)
            logits, cls_logits = model(volume, features, guidance)
            loss, terms = loss_terms(logits, cls_logits, mask, class_ids, ssim)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step+1}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            next_batch += 1
            epoch_sum += float(loss.detach())
            epoch_count += 1
            if smoke and step == 1:
                dump(job/'smoke_shapes.json', dict(input=list(image.shape), cost_volume=list(volume.shape),
                     guided_features=[list(f.shape) for f in features], guidance=list(guidance.shape),
                     mask=list(mask.shape), segmentation_logits=list(logits.shape), classification_logits=list(cls_logits.shape)))
            if next_batch == train_batches_per_epoch:
                scheduler.step(epoch_sum/epoch_count)
                epoch += 1
                next_batch, epoch_sum, epoch_count = 0, 0., 0
            if step == 1 or step % cfg['log_every'] == 0 or step == steps:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed = time.monotonic()-started
                peak = torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.
                eta = elapsed/max(1, step - (saved['step'] if resume else 0))*(steps-step)
                logging.info('step %d/%d loss %.6f grad_norm %.6f lr %.6g elapsed %.1fs ETA %.1fmin peak_VRAM %.2fGiB',
                    step, steps, float(loss.detach()), float(norm), optimizer.param_groups[0]['lr'], elapsed, eta/60, peak)
                status(job, 'running', phase='training', step=step, total_steps=steps,
                       elapsed_seconds=elapsed, eta_seconds=eta, peak_VRAM_GiB=peak)
            if step % cfg['checkpoint_every'] == 0 or step == steps:
                save_state(checkpoints/'last.pt', model, cfg, step, smoke=smoke,
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch,
                    next_batch=next_batch, epoch_sum=epoch_sum, epoch_count=epoch_count,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
                dump(job/'timing.json', dict(training_seconds=run_elapsed_before + time.monotonic()-started,
                                            step=step, epoch=epoch, next_batch=next_batch))
            if step >= steps:
                break
    save_state(checkpoints/'final.pt', model, cfg, step, smoke=smoke)
    if smoke:
        training_peak = torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.
        if torch.equal(first_weight, next(model.parameters()).detach()):
            raise RuntimeError('CostFilter weights were not updated')
        if not torch.equal(initial_encoder, next(baseline.encoder.parameters()).detach()):
            raise RuntimeError('Frozen encoder was changed')
        if any(p.grad is not None for p in baseline.parameters()):
            raise RuntimeError('Frozen A5000 accumulated gradients')
        reloaded = torch.load(checkpoints/'final.pt', map_location=device, weights_only=True)
        other = build_filter(cfg).to(device)
        other.load_state_dict(reloaded['model_state_dict'], strict=True)
        other.eval()
        model.eval()
        with torch.no_grad():
            expected = model(volume, features, guidance)[0]
            actual = other(volume, features, guidance)[0]
        error = float((actual-expected).abs().max())
        if error != 0:
            raise RuntimeError('Filter checkpoint reload changed outputs')
        # Formal evaluation has batch32 and a batch1 tail (vari-grip).
        # Check both with real grouped-feature/volume shapes, no extra updates.
        batch_checks = {}
        with torch.no_grad():
            for size in [1, cfg['eval_batch_size']]:
                v = volume[:1].repeat(size, 1, 1, 1, 1)
                f = [x[:1].repeat(size, 1, 1) for x in features]
                g = guidance[:1].repeat(size, 1, 1, 1)
                output, cls_output = model(v, f, g)
                if output.shape != (size, 2, 64, 64) or cls_output.shape != (size, 5):
                    raise RuntimeError('Evaluation batch size changed output dimensions')
                if not torch.isfinite(output).all() or not torch.isfinite(cls_output).all():
                    raise RuntimeError('Nonfinite evaluation batch outputs')
                batch_checks[str(size)] = dict(segmentation=list(output.shape), classification=list(cls_output.shape))
                del v, f, g, output, cls_output
        dump(job/'smoke_training_check.json', dict(passed=True, frozen_backbone=True,
             filter_weights_updated=True, checkpoint_reload_max_error=error,
             batch_size=cfg['batch_size'], steps=step,
             training_peak_VRAM_GiB=training_peak, evaluation_batch_checks=batch_checks,
             peak_VRAM_GiB=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.,
             elapsed_seconds=time.monotonic()-started))
    return checkpoints/'final.pt'


def metric(labels, values):
    labels, values = np.asarray(labels), np.asarray(values)
    if not np.isfinite(values).all() or set(labels) != {0, 1}:
        raise ValueError('Invalid metric inputs')
    precision, recall, _ = precision_recall_curve(labels, values)
    f1 = 2*precision*recall/np.maximum(precision+recall, 1e-12)
    fpr, tpr, _ = roc_curve(labels, values)
    return {'I-AUROC': float(roc_auc_score(labels, values)),
            'I-AP': float(average_precision_score(labels, values)),
            'I-F1': float(f1.max()), 'FPR@95TPR': float(fpr[np.where(tpr >= .95)[0][0]])}


def visualize(output, row, amap, filtered, fused):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    with Image.open(row['path']) as image:
        rgb = image.convert('RGB').resize((392, 392), Image.Resampling.BILINEAR)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].set_title('Full input: 392 x 392')
    shared_max = max(float(amap.max()), float(fused.max()), 1e-8)
    for ax, name, value, vmax in zip(axes[1:], SCORES, [amap, filtered, fused], [shared_max, 1., shared_max]):
        ax.imshow(rgb)
        ax.imshow(value, alpha=.55, cmap='inferno', vmin=0, vmax=vmax, extent=(0, 392, 392, 0))
        ax.set_title(f'{name}\nscore={row[name]:.5f}')
    for ax in axes:
        ax.axis('off')
    fig.suptitle(f'{row["category"]} | label={row["label"]} | {row["defect_type"]}')
    fig.tight_layout()
    filename = f'{row["category"]}_{row["label"]}.jpg'
    fig.savefig(output/filename, dpi=110)
    plt.close(fig)
    return filename


def evaluate(cfg, job, checkpoint, device, smoke=False):
    status(job, 'running', phase='final_evaluation')
    baseline = load_baseline(cfg, device)
    model = build_filter(cfg).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    path_keys = {'results_root', 'dinomaly2_repo', 'baseline_run', 'baseline_checkpoint', 'weights', 'data_path', 'dtd_images'}
    saved_protocol = {k:v for k,v in state['config'].items() if k not in path_keys}
    current_protocol = {k:v for k,v in cfg.items() if k not in path_keys}
    if saved_protocol != current_protocol or state['smoke'] != smoke:
        raise ValueError('Checkpoint numeric protocol/smoke mismatch')
    model.load_state_dict(state['model_state_dict'], strict=True)
    _, utils, _ = baseline_modules(cfg)
    kernel = utils.get_gaussian_kernel(cfg['gaussian_kernel'], cfg['gaussian_sigma']).to(device)
    data = TestImages(cfg['data_path'], smoke=smoke)
    loader = evaluation_loader(data, cfg)
    references = {r['path']: float(r['score']) for r in
                  read_csv(Path(cfg['baseline_run'])/'results/image_scores_005000.csv')}
    rows, pictures, seen = [], [], set()
    started = time.monotonic()
    with torch.no_grad():
        for image, indices in loader:
            image = image.to(device)
            en, de = baseline(image)
            volume, features, guidance, base = cost_inputs(en, de, cfg)
            logits, _ = model(volume, features, guidance)
            filtered = F.interpolate(logits.softmax(1)[:, 1:2], size=392,
                                     mode='bilinear', align_corners=True)
            fused = (1-cfg['filter_coefficient'])*base + cfg['filter_coefficient']*filtered
            processed = [kernel(F.interpolate(a, size=256, mode='bilinear', align_corners=False))
                         for a in [base, filtered, fused]]
            k = int(256*256*cfg['score_ratio'])
            scores = [a.flatten(1).sort(descending=True).values[:, :k].mean(1).cpu().tolist()
                      for a in processed]
            for i, index in enumerate(indices.tolist()):
                row = dict(data.rows[index])
                row.update({name: scores[j][i] for j, name in enumerate(SCORES)})
                row['A_reference_score'] = references[row['path']]
                row['A_reference_abs_error'] = abs(row['A5000_full']-row['A_reference_score'])
                rows.append(row)
                key = row['category'], row['label']
                if key not in seen:
                    filename = visualize(job/'visualizations', row,
                        *[x[i, 0].cpu().numpy() for x in [base, filtered, fused]])
                    pictures.append(filename)
                    seen.add(key)
            if len(rows) % (cfg['eval_batch_size']*20) == 0:
                logging.info('评估 %d/%d 张', len(rows), len(data))
    write_csv(job/'image_scores.csv', rows)
    max_error = max(r['A_reference_abs_error'] for r in rows)
    # Different NumPy/timm/PyTorch kernels or inference batch sizes may change
    # last bits, but a significant discrepancy invalidates a matched comparison.
    if max_error > 1e-4:
        raise RuntimeError(f'A5000 scores differ from original results: max_abs_error={max_error}')
    variants = {}
    flat = []
    for score in SCORES:
        per_class = {}
        for category in CATEGORIES:
            subset = [r for r in rows if r['category'] == category]
            per_class[category] = metric([r['label'] for r in subset], [r[score] for r in subset])
            flat.append(dict(method=score, category=category, **per_class[category]))
        mean = {key: float(np.mean([r[key] for r in per_class.values()]))
                for key in per_class[CATEGORIES[0]]}
        variants[score] = dict(mean=mean, per_class=per_class)
        flat.append(dict(method=score, category='macro_mean', **mean))
        logging.info('%s macro I-AUROC %.6f', score, mean['I-AUROC'])
    historical = dict(A5000_full=.8532808054567191,
                      NCut_region=.8828498620320158, NCut_full_fusion=.8855578404547944)
    report = dict(experiment_id=cfg['experiment_id'], smoke=smoke, filter_steps=state['step'],
        test_image_count=len(rows), categories=CATEGORIES, variants=variants,
        historical_same_checkpoint=historical, input_view=VIEW,
        baseline_checkpoint_sha256=cfg['baseline_checkpoint_sha256'],
        filter_checkpoint_sha256=digest(checkpoint), filter_coefficient=cfg['filter_coefficient'],
        max_A_reference_score_error=max_error, evaluation_seconds=time.monotonic()-started,
        pixel_metrics=None, pixel_metrics_reason='No real pixel-level ground truth is loaded for this InsPLAD image-level experiment.',
        limitations=['Dinomaly2+CostFilter adaptation, not unmodified upstream CostFilter+Dinomaly reproduction.',
                     'Additional DTD texture data and synthetic pixel supervision.',
                     '5000 filter steps versus upstream 17 epochs; full-view and matched 256/top1% postprocessing.',
                     'Guidance uses the direct 28-to64 path in both train/test, fixing an upstream train/test discrepancy.',
                     'Same exploratory test split as prior NCut experiments; no held-out benchmark claim.'])
    dump(job/'summary.json', report)
    write_csv(job/'per_category_metrics.csv', flat)
    text = ['# Dinomaly2 A5000 + CostFilter', '', f'Smoke: {smoke}; filter steps: {state["step"]}; test images: {len(rows)}.', '',
            '| Category | A5000 | CostFilter only | Fixed 0.05 fusion |', '|---|---:|---:|---:|']
    for category in CATEGORIES+['macro_mean']:
        values = [variants[s]['mean' if category == 'macro_mean' else 'per_class'] for s in SCORES]
        numbers = [v['I-AUROC'] if category == 'macro_mean' else v[category]['I-AUROC'] for v in values]
        text.append('| '+category+' | '+' | '.join(f'{100*v:.2f}%' for v in numbers)+' |')
    text += ['', 'Historical full-view NCut region: 88.28%; NCut/full fusion: 88.56%.',
             'Smoke metrics use only 10 fixed test images and are not valid benchmark results.' if smoke else
             'Final evaluation only; test AUROC is not used to select steps, checkpoints, fusion ratio, or learning rate.',
             '', 'See config.json, summary.json and source snapshots for all adaptations.']
    (job/'REPORT.md').write_text('\n'.join(text)+'\n')
    (job/'visualizations/index.html').write_text('<!doctype html><meta charset="utf-8"><title>A5000 + CostFilter</title>'
        '<h1>固定样本：A5000 / CostFilter / 融合</h1>'
        '<p>每类首张正常和首张异常。A与融合热图共用色标；过滤概率色标0–1。非人工前景真值。</p>'
        + ''.join(f'<img style="max-width:100%" src="{p}"><hr>' for p in pictures))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'smoke', 'train', 'evaluate'])
    parser.add_argument('--run', type=Path, help='Existing run for final evaluation')
    parser.add_argument('--resume', type=Path, help='Resume an interrupted training run from last.pt')
    parser.add_argument('--batch-size', type=int, help='Override only for a new run; saved explicitly')
    args = parser.parse_args()
    cfg = json.loads((HERE/'config.json').read_text())
    if args.resume and args.mode != 'train':
        parser.error('--resume is only valid for train')
    if args.run and args.mode != 'evaluate':
        parser.error('--run is only valid for evaluate')
    existing = args.resume or args.run
    if args.mode == 'evaluate' and existing is None:
        pointer = Path(cfg['results_root'])/'latest_train.txt'
        if not pointer.exists():
            parser.error('No training run found; provide --run')
        existing = Path(pointer.read_text().strip())
    if existing:
        job = existing.resolve()
        cfg = json.loads((job/'config.json').read_text())
        meta = json.loads((job/'identity.json').read_text())
        if meta['smoke']:
            parser.error('Smoke runs cannot be used as formal training/evaluation runs')
        if args.batch_size is not None:
            parser.error('Do not change batch size while resuming/evaluating')
        saved_sources = json.loads((job/'source_sha256.json').read_text())
        if any(digest(p) != h for p, h in saved_sources.items()):
            raise ValueError('Source changed since this run; do not resume/evaluate with different code')
        current_environment = sys.version + '\n' + subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'], text=True)
        if (job/'environment.txt').read_text() != current_environment:
            raise ValueError('Environment changed since this run; keep the same environment when resuming/evaluating')
    else:
        if args.batch_size is not None:
            if args.batch_size < 1:
                parser.error('Batch size must be positive')
            cfg['batch_size'] = args.batch_size
        branch = 'runs' if args.mode == 'train' else args.mode
        job = Path(cfg['results_root'])/branch/f'{args.mode}_seed{cfg["seed"]}_{now():%Y%m%d_%H%M%S}_{os.getpid()}'
        job.mkdir(parents=True, exist_ok=False)
        dump(job/'config.json', cfg)
        dump(job/'identity.json', dict(smoke=args.mode == 'smoke', created_at=now().isoformat()))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(job/'log.txt'), logging.StreamHandler()])
    logging.Formatter.converter = lambda *unused: now().timetuple()
    logging.info('输出目录：%s', job)
    torch.set_num_threads(2)
    seed_all(cfg['seed'])
    try:
        status(job, 'running', phase='validation')
        # Recheck all data/model identities on resume as well as on a new run.
        validate(cfg, job)
        if not existing:
            preview_synthetic(cfg, job)
        if args.mode == 'check':
            status(job, 'complete', phase='data_and_source_checks', training_started=False)
            print('检查完成；未启动模型训练：', job)
            return
        if not torch.cuda.is_available():
            raise RuntimeError('GPU unavailable. check works without a GPU; smoke/train/evaluate require CUDA.')
        device = torch.device(f'cuda:{cfg["cuda"]}')
        torch.cuda.set_device(device)
        if args.mode == 'train':
            (Path(cfg['results_root'])/'latest_train.txt').write_text(str(job)+'\n')
        if args.mode in ['train', 'smoke']:
            checkpoint = train(cfg, job, device, smoke=args.mode == 'smoke', resume=bool(args.resume))
        else:
            checkpoint = job/'checkpoints/final.pt'
        report = evaluate(cfg, job, checkpoint, device, smoke=args.mode == 'smoke')
        status(job, 'complete', phase='finished', filter_steps=report['filter_steps'],
               smoke=report['smoke'], summary=str(job/'summary.json'))
        print('完成：', job/'summary.json')
    except BaseException as exc:
        status(job, 'failed', exception=type(exc).__name__, message=str(exc))
        logging.exception('任务未完成')
        raise


if __name__ == '__main__':
    main()
