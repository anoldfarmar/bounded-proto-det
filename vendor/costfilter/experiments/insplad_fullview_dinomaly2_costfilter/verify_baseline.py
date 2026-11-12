"""Read-only full-test A5000 check under the cloned CostFilter environment."""
import json
import logging
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from insplad_data import CATEGORIES, TestImages, evaluation_loader
from model_adapter import baseline_modules, load_baseline
from run import dump, metric, now, read_csv, seed_all


def main():
    here = Path(__file__).resolve().parent
    cfg = json.loads((here/'config.json').read_text())
    torch.set_num_threads(2)
    seed_all(cfg['seed'])
    device = torch.device('cuda:0')
    model = load_baseline(cfg, device)
    _, utils, _ = baseline_modules(cfg)
    kernel = utils.get_gaussian_kernel(5, 4).to(device)
    original = {r['path']: float(r['score']) for r in
                read_csv(Path(cfg['baseline_run'])/'results/image_scores_005000.csv')}
    data = TestImages(cfg['data_path'])
    rows, errors = [], []
    started = time.monotonic()
    with torch.no_grad():
        for image, indices in evaluation_loader(data, cfg):
            image = image.to(device)
            en, de = model(image)
            amap, _ = utils.cal_anomaly_maps(en, de, 392)
            amap = kernel(F.interpolate(amap, size=256, mode='bilinear', align_corners=False))
            values = amap.flatten(1).sort(descending=True).values[:, :655].mean(1).cpu().tolist()
            for index, value in zip(indices.tolist(), values):
                row = dict(data.rows[index], score=value)
                rows.append(row)
                errors.append(abs(value-original[row['path']]))
    per_class = {c: metric([r['label'] for r in rows if r['category'] == c],
                           [r['score'] for r in rows if r['category'] == c]) for c in CATEGORIES}
    mean = sum(v['I-AUROC'] for v in per_class.values())/5
    passed = max(errors) <= 1e-4 and abs(mean-.8532808054567191) < 1e-5
    report = dict(passed=passed, test_images=len(rows), max_score_abs_error=max(errors),
                  macro_I_AUROC=mean, per_class=per_class, elapsed_seconds=time.monotonic()-started,
                  baseline_checkpoint=cfg['baseline_checkpoint'], training_steps_run=0,
                  note='Read-only original A5000 check. No CostFilter training or evaluation.')
    output = Path(cfg['results_root'])/'validation'
    output.mkdir(parents=True, exist_ok=True)
    path = output/f'baseline_check_{now():%Y%m%d_%H%M%S}.json'
    dump(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Report:', path)
    if not passed:
        raise RuntimeError('Original A5000 results did not match')


if __name__ == '__main__':
    main()
