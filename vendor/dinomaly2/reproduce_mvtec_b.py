"""Dinomaly2-B / MVTec AD conventional MUAD, arXiv 2510.17611v3.

Reuses the released model, loss, optimizer and evaluator. No network downloads.
"""
import argparse
import csv
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision.datasets import ImageFolder

from dataset import MVTecDataset, get_data_transforms
from dinomaly_2D import build_model, setup_seed
from optimizers import StableAdamW
from utils import WarmupCosineScheduler, evaluation_batch, global_cosine_hm_percent

CATEGORIES = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable',
              'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush',
              'transistor', 'zipper']
METRICS = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']
PAPER = 'https://arxiv.org/html/2510.17611v3'
TARGET = dict(zip(METRICS, [99.8, 99.9, 99.3, 98.4, 69.3, 68.9, 94.8]))


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def validate_data(root):
    """Check the official split, mask pairing and PNG integrity before training."""
    root = Path(root)
    errors, report, totals = [], {}, np.zeros(3, dtype=int)
    for category in CATEGORIES:
        folder = root / category
        train = sorted((folder / 'train' / 'good').glob('*.png'))
        good = sorted((folder / 'test' / 'good').glob('*.png'))
        bad = sorted(p for p in (folder / 'test').glob('*/*.png') if p.parent.name != 'good')
        masks = sorted((folder / 'ground_truth').glob('*/*.png'))
        counts = [len(train), len(good), len(bad)]
        report[category] = dict(zip(['train_good', 'test_good', 'test_bad'], counts))
        totals += counts
        if not all(counts):
            errors.append(f'{category}: 缺少 train/good 或 test/good、test/缺陷类型 中的 PNG 图像')
        if (folder / 'train').is_dir():
            if {p.name for p in (folder / 'train').iterdir() if p.is_dir()} != {'good'}:
                errors.append(f'{category}: train 下应当只有 good 目录')
        expected = {folder / 'ground_truth' / p.parent.name / (p.stem + '_mask.png') for p in bad}
        if expected != set(masks):
            errors.append(f'{category}: 异常图与 *_mask.png 未一一匹配')
        for path in train + good + bad + masks:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as e:
                errors.append(f'损坏的图像 {path}: {e}')
        for image_path in bad:
            mask_path = folder / 'ground_truth' / image_path.parent.name / (image_path.stem + '_mask.png')
            if mask_path.exists():
                with Image.open(image_path) as image, Image.open(mask_path) as mask:
                    if image.size != mask.size:
                        errors.append(f'图像与掩码尺寸不同: {image_path}')
                    values = np.asarray(mask)
                    if not np.isin(values, [0, 255]).all() or not values.any():
                        errors.append(f'异常掩码应是非空的0/255二值图: {mask_path}')
    if totals.tolist() != [3629, 467, 1258]:
        errors.append(f'总数不符合原版 MVTec AD：{totals.tolist()}，应为 [3629, 467, 1258]')
    if errors:
        raise ValueError(f'数据尚未就绪：{root}\n' + '\n'.join(errors[:25]) +
                         '\n请解压到该目录，使 bottle、carpet 等15类目录直接位于其下。')
    return report


def save_checkpoint(path, model, optimizer, scheduler, step, args):
    # Frozen encoder remains in the separately hashed local pretrained file.
    state = dict(bottleneck=model.bottleneck.state_dict(), decoder=model.decoder.state_dict(),
                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 step=step, config=vars(args), weights_sha256=args.weights_sha256)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def save_eval_weights(output, model, step, args):
    folder = output / 'checkpoints'
    folder.mkdir(exist_ok=True)
    path = folder / f'step_{step:06d}.pt'
    temporary = path.with_suffix('.tmp')
    torch.save(dict(bottleneck=model.bottleneck.state_dict(), decoder=model.decoder.state_dict(),
                    step=step, config=vars(args), weights_sha256=args.weights_sha256), temporary)
    temporary.replace(path)


def evaluate(model, datasets, args, output, step, device):
    if args.dataset == 'insplad':
        from insplad_adapter import evaluate_insplad
        return evaluate_insplad(model, datasets, args, output, step, device)
    rows = []
    for category, dataset in datasets:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
        values = evaluation_batch(model, loader, device, max_ratio=0.01, resize_mask=256)
        row = dict(category=category, **{k: float(v) for k, v in zip(METRICS, values)})
        if not all(np.isfinite(row[k]) for k in METRICS):
            raise RuntimeError(f'评估指标存在 NaN/Inf: {category}')
        rows.append(row)
        logging.info('%s %s', category, json.dumps(row))
    mean = {k: float(np.mean([row[k] for row in rows])) for k in METRICS}
    rows.append(dict(category='Mean', **mean))
    with (output / f'metrics_{step:06d}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['category'] + METRICS)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(step=step, units='fraction_0_to_1', mean=mean, paper=PAPER,
                   target_percent=TARGET,
                   delta_percentage_points={k: 100 * mean[k] - TARGET[k] for k in METRICS})
    (output / f'summary_{step:06d}.json').write_text(json.dumps(summary, indent=2))
    logging.info('Mean at step %d: %s', step, json.dumps(mean))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'smoke', 'train', 'eval'])
    parser.add_argument('--dataset', choices=['mvtec', 'insplad'], default='mvtec')
    parser.add_argument('--insplad_subset', choices=['all', 'fault5'], default='all')
    parser.add_argument('--experiment_id')
    parser.add_argument('--experiment_config')
    parser.add_argument('--save_eval_weights', action='store_true')
    parser.add_argument('--data_path')
    parser.add_argument('--weights', default='/root/autodl-tmp/model/dinov2_vitb14_reg4/dinov2_vitb14_reg4_pretrain.pth')
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', help='eval 模式使用的 last.pt')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--total_iters', type=int, default=40000)
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--dropout', type=float, choices=[0.2, 0.4], default=0.4)
    parser.add_argument('--nb_lr', type=float, default=2e-3,
                        help='论文统一 LR=2e-3；预览代码首层稳定性调整为2e-4，可显式指定')
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    if args.data_path is None:
        args.data_path = ('/root/autodl-tmp/data/insplad/unsupervised_anomaly_detection'
                          if args.dataset == 'insplad' else
                          '/root/autodl-tmp/data/MVtec-AD/mvtec_anomaly_detection')
    # Fixed method configuration; snapshot into every run and checkpoint.
    args.backbone = 'dinov2reg_vit_base_14'
    args.image_size, args.crop_size = 448, 392
    args.la, args.lc, args.cr = 1, 2, 1
    args.ll_ratio, args.ll_factor, args.lr_decay_ratio = 0.9, 0.1, 0.1
    if min(args.batch_size, args.total_iters, args.eval_every) < 1 or args.workers < 0:
        parser.error('batch_size / total_iters / eval_every 必须为正，workers不能为负')
    if not Path(args.weights).is_file():
        parser.error(f'权重不存在：{args.weights}')
    setup_seed(args.seed)
    args.weights_sha256 = digest(args.weights)
    # Strict loading: any missing/unexpected tensor aborts here, without downloads.
    model = build_model(args, 'cpu')
    model.encoder.requires_grad_(False)
    print('本地 ViT-B/14 reg4 权重严格加载成功，SHA256:', args.weights_sha256, flush=True)
    report = None
    categories = CATEGORIES
    if args.dataset == 'insplad':
        from insplad_adapter import validate_insplad, ImageAnomalyDataset
        report = validate_insplad(args.data_path, args.insplad_subset)
        categories = list(report)
        args.categories = categories
        print('InsPLAD 数据完整性检查通过：', json.dumps(report, ensure_ascii=False), flush=True)
    elif args.mode != 'smoke':
        report = validate_data(args.data_path)
        print('15类数据完整性检查通过：3629训练正常 / 467测试正常 / 1258测试异常', flush=True)
    if args.mode == 'check':
        if args.experiment_id:
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=False)
            if args.dataset == 'insplad':
                from insplad_adapter import write_data_manifest
                write_data_manifest(args.data_path, categories, output / 'dataset_manifest.csv')
                args.dataset_manifest_sha256 = digest(output / 'dataset_manifest.csv')
            (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
            (output / 'dataset_counts.json').write_text(json.dumps(report, indent=2))
        return
    if not torch.cuda.is_available() or args.cuda >= torch.cuda.device_count():
        parser.error('训练/试跑/评估需要可用 GPU；无卡模式可执行 check')
    device = torch.device(f'cuda:{args.cuda}')
    torch.cuda.set_device(device)
    model = model.to(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(output / 'log.txt'), logging.StreamHandler()])
    if args.dataset == 'insplad' and args.insplad_subset == 'fault5':
        from insplad_adapter import write_data_manifest
        write_data_manifest(args.data_path, categories, output / 'dataset_manifest.csv')
        args.dataset_manifest_sha256 = digest(output / 'dataset_manifest.csv')
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    (output / 'dataset_counts.json').write_text(json.dumps(report, indent=2))
    (output / 'environment.txt').write_text(
        f'Python {platform.python_version()}\nGPU {torch.cuda.get_device_name(device)}\n' +
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    source = output / 'source'
    source.mkdir()
    project = Path(__file__).resolve().parent
    files = ['reproduce_mvtec_b.py', 'run_mvtec_b.sh', 'dinomaly_2D.py', 'dataset.py',
             'utils.py', 'models/vit_encoder.py', 'models/uad.py',
             'models/vision_transformer.py', 'optimizers/StableAdamW.py',
             'dinov2/models/vision_transformer.py', 'dinov2/layers/attention.py',
             'dinov2/layers/block.py']
    if args.dataset == 'insplad':
        files += ['insplad_adapter.py', 'run_insplad_b.sh']
    if args.experiment_config:
        config_path = Path(args.experiment_config).resolve()
        files += [str(p.relative_to(project)) for p in sorted(config_path.parent.iterdir())
                  if p.is_file() and p.suffix in {'.json', '.py', '.sh', '.md'}]
    for file in files:
        target = source / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / file, target)
    (output / 'source_sha256.json').write_text(json.dumps({f: digest(project / f) for f in files}, indent=2))
    logging.info('论文 %s；配置 %s', PAPER, vars(args))
    transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)
    if args.dataset == 'insplad':
        test_sets = [(c, ImageAnomalyDataset(Path(args.data_path) / c, transform,
                                            smoke=args.mode == 'smoke')) for c in categories]
    else:
        test_sets = [(c, MVTecDataset(str(Path(args.data_path) / c), transform, gt_transform, 'test'))
                     for c in categories] if args.mode != 'smoke' else []
    if args.mode == 'eval':
        if not args.checkpoint:
            parser.error('eval 模式需要 --checkpoint /path/to/last.pt')
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if state['weights_sha256'] != args.weights_sha256:
            raise ValueError('编码器权重哈希与检查点不同')
        if state['config'].get('dataset', 'mvtec') != args.dataset:
            raise ValueError('评估数据集与检查点的训练数据集不同')
        if args.dataset == 'insplad' and state['config'].get('insplad_subset', 'all') != args.insplad_subset:
            raise ValueError('检查点的训练类别范围不同；五类实验不能加载旧17类模型作为基线')
        for key in ['backbone', 'image_size', 'crop_size', 'lc', 'la', 'cr']:
            if state['config'][key] != getattr(args, key):
                raise ValueError(f'评估配置与检查点不同: {key}')
        model.bottleneck.load_state_dict(state['bottleneck'], strict=True)
        model.decoder.load_state_dict(state['decoder'], strict=True)
        evaluate(model, test_sets, args, output, state['step'], device)
        return
    if args.mode == 'smoke' and args.dataset == 'mvtec':
        # Synthetic input is only an execution/memory check, never a paper result.
        dataset = TensorDataset(torch.randn(args.batch_size, 3, args.crop_size, args.crop_size),
                                torch.zeros(args.batch_size, dtype=torch.long))
        iterations = 2
    else:
        if args.dataset == 'insplad':
            dataset = ConcatDataset([ImageAnomalyDataset(Path(args.data_path) / c, transform, phase='train')
                                     for c in categories])
        else:
            dataset = ConcatDataset([ImageFolder(str(Path(args.data_path) / c / 'train'), transform)
                                     for c in categories])
        iterations = 2 if args.mode == 'smoke' else args.total_iters
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)
    if args.dataset == 'insplad':
        expected_train = sum(v['train_good'] for v in report.values())
        if len(dataset) != expected_train:
            raise ValueError(f'实际训练集数量{len(dataset)}与已验证清单{expected_train}不一致')
        logging.info('训练类别: %s; 正常训练图: %d; 测试类别: %s',
                     categories, len(dataset), [c for c, _ in test_sets])
    if len(loader) == 0:
        raise ValueError('训练数据不足一个 batch')
    optimizer = StableAdamW([
        {'params': model.bottleneck[0].parameters(), 'lr': args.nb_lr},
        {'params': model.bottleneck[1].parameters()}, {'params': model.decoder.parameters()}],
        lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-10)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=100,
                                     total_epochs=args.total_iters, final_ratio=args.lr_decay_ratio)
    trainable = list(model.bottleneck.parameters()) + list(model.decoder.parameters())
    step, started = 0, time.monotonic()
    model.train()
    while step < iterations:
        for image, _ in loader:
            before = model.bottleneck[0][0].weight.detach().clone() if args.mode == 'smoke' else None
            en, de = model(image.to(device))
            loss = global_cosine_hm_percent(en, de, p=min(args.ll_ratio * step / 1000, args.ll_ratio),
                                            factor=args.ll_factor)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Loss NaN/Inf at step {step}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 0.1, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if args.mode == 'smoke' and step == 2:
                if torch.equal(before, model.bottleneck[0][0].weight):
                    raise RuntimeError('第二步优化未更新瓶颈权重')
            if step == 1 or step % 100 == 0 or step == iterations:
                logging.info('step %d/%d loss %.6f grad_norm %.6f lr %s elapsed %.1fs peak_VRAM %.2fGiB',
                             step, iterations, loss.item(), norm.item(), scheduler.get_last_lr(),
                             time.monotonic() - started, torch.cuda.max_memory_allocated(device) / 2**30)
            if step % args.eval_every == 0 or step == iterations:
                # Save before evaluation so evaluation failure cannot lose training.
                save_checkpoint(output / 'last.pt', model, optimizer, scheduler, step, args)
                if args.save_eval_weights:
                    save_eval_weights(output, model, step, args)
                if args.mode != 'smoke' or args.dataset == 'insplad':
                    evaluate(model, test_sets, args, output, step, device)
                model.train()
            if step >= iterations:
                break
    if args.mode == 'smoke':
        state = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        model.bottleneck.load_state_dict(state['bottleneck'], strict=True)
        model.decoder.load_state_dict(state['decoder'], strict=True)
        if args.dataset == 'insplad':
            logging.info('INSPLAD SMOKE PASSED：真实训练图两步更新，%d类测试子集评分和指标输出、检查点回读；不是完整实验结果', len(categories))
            return
        # Exercise the unchanged seven-metric evaluator on artificial masks.
        synthetic = []
        for i in range(4):
            mask = torch.zeros(1, args.crop_size, args.crop_size)
            if i % 2:
                mask[:, 100:160, 100:160] = 1
            synthetic.append((torch.randn(3, args.crop_size, args.crop_size), mask, i % 2, f'synthetic_{i}'))
        values = evaluation_batch(model, DataLoader(synthetic, batch_size=4), device,
                                  max_ratio=0.01, resize_mask=256)
        if len(values) != 7 or not np.isfinite(values).all():
            raise RuntimeError('合成数据上的七指标评估失败')
        logging.info('SMOKE PASSED：完整B模型、392输入、batch=%d、两步反向传播/参数更新/检查点回读/七指标评估；无真实指标', args.batch_size)


if __name__ == '__main__':
    try:
        main()
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(2)
