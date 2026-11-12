"""Recompute all three main results with Python standard library, no GPU/data."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from paths import ROOT, sha, verify_sources

def rows(name):
    with (ROOT / 'results/recorded' / name).open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

def auc(labels, scores):
    pairs = sorted(zip(scores, labels))
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        raise ValueError('AUROC needs both labels')
    rank_sum = 0.
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += (i + 1 + j) / 2 * sum(y for _, y in pairs[i:j])
        i = j
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)

def fpr95(labels, scores):
    pairs = sorted(zip(scores, labels), reverse=True)
    positive = sum(labels)
    negative = len(labels) - positive
    tp = fp = i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        tp += sum(y for _, y in pairs[i:j])
        fp += sum(1 - y for _, y in pairs[i:j])
        if tp / positive >= .95:
            return fp / negative
        i = j
    raise ValueError('ROC point not found')

def compute():
    a, n, c = rows('a5000_scores.csv'), rows('ncut_scores.csv'), rows('costfilter_scores.csv')
    keyed = [{r['relative_path']:r for r in values} for values in [a, n, c]]
    assert len(a) == len(n) == len(c) == 2656
    assert all(len(k) == 2656 for k in keyed)
    assert keyed[0].keys() == keyed[1].keys() == keyed[2].keys()
    for key in keyed[0]:
        ar, nr, cr = [k[key] for k in keyed]
        assert ar['category'] == nr['category'] == cr['category']
        assert ar['label'] == nr['label'] == cr['label']
        assert abs(float(ar['score']) - float(nr['full_top1'])) < 1e-6
        assert abs(float(ar['score']) - float(cr['A5000_full'])) < 1e-6
        assert abs(float(nr['ncut_seed_mix_0p5']) - .5 * (float(nr['full_top1']) + float(nr['ncut_seed_fixed_k']))) < 1e-12
    config = json.loads((ROOT / 'configs/a5000.json').read_text())
    methods = [('A5000', a, 'score'), ('NCut_full_fusion', n, 'ncut_seed_mix_0p5'), ('CostFilter_full_fusion', c, 'costfilter_fused_0p05')]
    result = dict(categories=config['categories'], test_images=2656, test_normal=2254, test_anomalous=402, averaging='unweighted mean of five per-category metrics', methods={})
    for name, values, field in methods:
        per_class = {}
        for category in config['categories']:
            subset = [r for r in values if r['category'] == category]
            labels = [int(r['label']) for r in subset]
            scores = [float(r[field]) for r in subset]
            per_class[category] = {'I-AUROC':auc(labels, scores), 'FPR@95TPR':fpr95(labels, scores)}
        result['methods'][name] = dict(score_column=field, per_class=per_class, mean={k:mean(m[k] for m in per_class.values()) for k in ['I-AUROC', 'FPR@95TPR']})
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Rebuild small comparison reports from recorded CSVs')
    parser.add_argument('--files', action='store_true', help='Check delivered files against SHA256SUMS')
    args = parser.parse_args()
    identities = verify_sources()
    result = compute()
    if args.write:
        (ROOT / 'results/summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        with (ROOT / 'results/comparison.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['category', 'A5000', 'NCut_full_fusion', 'CostFilter_full_fusion'])
            for category in result['categories'] + ['macro_mean']:
                writer.writerow([category] + [m['mean']['I-AUROC'] if category == 'macro_mean' else m['per_class'][category]['I-AUROC'] for m in result['methods'].values()])
    else:
        expected = json.loads((ROOT / 'results/summary.json').read_text())
        if result != expected:
            raise ValueError('Recorded scores no longer reproduce the saved three-method table')
    if args.files:
        for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
            value, filename = line.split('  ', 1)
            if sha(ROOT / filename) != value:
                raise ValueError('Delivered file checksum mismatch: ' + filename)
    print(json.dumps(dict(passed=True, source_identity=identities, means={k:v['mean'] for k,v in result['methods'].items()}, test_images=2656, no_gpu_or_external_assets_needed=True), indent=2))

if __name__ == '__main__':
    main()
