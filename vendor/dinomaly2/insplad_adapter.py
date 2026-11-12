"""Image-level InsPLAD adapter; preserves the released Dinomaly scoring rule."""
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from utils import cal_anomaly_maps, get_gaussian_kernel, f1_score_max

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
METRICS = ['I-AUROC', 'I-AP', 'I-F1']
FAULT5_COUNTS = {
    'glass-insulator': (2298, 581, 90),
    'lightning-rod-suspension': (462, 117, 50),
    'polymer-insulator-upper-shackle': (935, 235, 102),
    'vari-grip': (477, 114, 111),
    'yoke-suspension': (4834, 1207, 49),
}


def images(folder):
    return sorted(p for p in Path(folder).glob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)


def validate_insplad(root, subset='all'):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'InsPLAD 路径不存在: {root}')
    report = {}
    errors = []
    if subset not in {'all', 'fault5'}:
        raise ValueError(f'未知InsPLAD子集: {subset}')
    folders = ([root / c for c in FAULT5_COUNTS] if subset == 'fault5' else
               sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')))
    for folder in folders:
        train = images(folder / 'train' / 'good')
        good = images(folder / 'test' / 'good')
        bad = sorted(p for defect in (folder / 'test').glob('*')
                     if defect.is_dir() and defect.name != 'good' for p in images(defect))
        if not train or not good:
            errors.append(f'{folder.name}: 正常训练或测试图缺失')
        train_folders = {p.name for p in (folder / 'train').glob('*') if p.is_dir()}
        if train_folders != {'good'}:
            errors.append(f'{folder.name}: 训练集目录必须仅包含good')
        for path in train + good + bad:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as e:
                errors.append(f'损坏图像 {path}: {e}')
        report[folder.name] = dict(train_good=len(train), test_good=len(good), test_bad=len(bad),
                                   category_auroc_defined=bool(good and bad))
        if subset == 'fault5' and (len(train), len(good), len(bad)) != FAULT5_COUNTS[folder.name]:
            errors.append(f'{folder.name}: 样本数量不符合冻结五类配置')
    expected_classes = 5 if subset == 'fault5' else 17
    if len(report) != expected_classes:
        errors.append(f'预期{expected_classes}个类别，实际{len(report)}个')
    totals = [sum(row[k] for row in report.values()) for k in ['train_good', 'test_good', 'test_bad']]
    expected_totals = [9006, 2254, 402] if subset == 'fault5' else [21139, 5686, 795]
    if totals != expected_totals:
        errors.append(f'本地InsPLAD划分发生变化：{totals}，预期{expected_totals}（不含.ipynb_checkpoints副本）')
    if errors:
        raise ValueError('\n'.join(errors[:25]))
    return report


def write_data_manifest(root, categories, output):
    """Audit selected paths/labels/sizes; this is not a source-frame leakage audit."""
    rows = []
    for category in categories:
        folder = Path(root) / category
        for split in ['train', 'test']:
            for state in sorted(p for p in (folder / split).iterdir() if p.is_dir()):
                if split == 'train' and state.name != 'good':
                    raise ValueError('训练集包含非正常目录')
                for path in images(state):
                    rows.append(dict(category=category, split=split, defect_type=state.name,
                                     label=int(state.name != 'good'), path=str(path.resolve()),
                                     size_bytes=path.stat().st_size))
    write_csv(output, rows, ['category', 'split', 'defect_type', 'label', 'path', 'size_bytes'])


class ImageAnomalyDataset(Dataset):
    def __init__(self, root, transform, smoke=False, phase='test'):
        self.transform = transform
        self.phase = phase
        if phase == 'train':
            self.samples = [(p, 0) for p in images(Path(root) / 'train' / 'good')]
            return
        good = [(p, 0) for p in images(Path(root) / 'test' / 'good')]
        bad = [(p, 1) for d in sorted((Path(root) / 'test').iterdir())
               if d.is_dir() and d.name != 'good' for p in images(d)]
        self.samples = good[:1] + bad[:1] if smoke else good + bad

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert('RGB'))
        if self.phase == 'train':
            return tensor, label
        return tensor, label, str(path)


def image_scores(model, loader, device):
    # Identical to utils.evaluation_batch_noseg with resize_mask=256 and max_ratio=.01.
    kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    result = []
    model.eval()
    with torch.no_grad():
        for images_, labels, paths in loader:
            images_ = images_.to(device)
            en, de = model(images_)
            amap, _ = cal_anomaly_maps(en, de, images_.shape[-1])
            amap = F.interpolate(amap, size=256, mode='bilinear', align_corners=False)
            amap = kernel(amap).flatten(1)
            score = torch.sort(amap, dim=1, descending=True)[0][:, :int(amap.shape[1] * .01)].mean(1)
            if not torch.isfinite(score).all():
                raise RuntimeError('InsPLAD分数出现NaN/Inf')
            result += [dict(path=p, label=int(y), score=float(s))
                       for p, y, s in zip(paths, labels.tolist(), score.cpu().tolist())]
    return result


def metric_values(records):
    labels = np.array([r['label'] for r in records])
    scores = np.array([r['score'] for r in records])
    if len(np.unique(labels)) != 2:
        return {k: None for k in METRICS}
    return dict(zip(METRICS, [float(roc_auc_score(labels, scores)),
                             float(average_precision_score(labels, scores)),
                             float(f1_score_max(labels, scores))]))


def write_csv(path, rows, keys):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_insplad(model, datasets, args, output, step, device):
    rows, all_scores = [], []
    for category, dataset in datasets:
        records = image_scores(model, DataLoader(dataset, batch_size=args.batch_size,
                                                 num_workers=args.workers, shuffle=False), device)
        for record in records:
            record['category'] = category
        all_scores += records
        values = metric_values(records)
        row = dict(category=category, test_good=sum(r['label'] == 0 for r in records),
                   test_bad=sum(r['label'] == 1 for r in records), **values)
        rows.append(row)
        logging.info('%s %s', category, json.dumps(row))
    valid = [row for row in rows if row['I-AUROC'] is not None]
    if not valid:
        raise ValueError('没有可计算类别AUROC的类别')
    macro = {k: float(np.mean([row[k] for row in valid])) for k in METRICS}
    mixed = metric_values(all_scores)
    class_count = len(datasets)
    summary = dict(dataset='InsPLAD', subset=getattr(args, 'insplad_subset', 'all'),
                   experiment_id=getattr(args, 'experiment_id', None),
                   step=step, smoke=args.mode == 'smoke',
                   units='fraction_0_to_1', mean=macro, macro_category_count=len(valid),
                   excluded_from_macro=[row['category'] for row in rows if row['I-AUROC'] is None],
                   mixed_all_categories=mixed, test_image_count=len(all_scores),
                   pixel_metrics=None,
                   categories=[c for c, _ in datasets],
                   protocol=f'train {class_count} normal categories jointly; macro over {len(valid)} categories with both labels; mixed pools all {class_count}')
    write_csv(output / f'metrics_{step:06d}.csv', rows,
              ['category', 'test_good', 'test_bad'] + METRICS)
    write_csv(output / f'image_scores_{step:06d}.csv', all_scores,
              ['category', 'path', 'label', 'score'])
    (output / f'summary_{step:06d}.json').write_text(json.dumps(summary, indent=2))
    logging.info('InsPLAD macro(%d categories) step %d %s; mixed(%d categories) %s',
                 len(valid), step, json.dumps(macro), class_count, json.dumps(mixed))
    return summary
