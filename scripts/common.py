"""Portable integration around unchanged A5000 train and NCut code."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
from paths import ROOT, CODE, sha, load_paths, verify_sources

EXPERIMENT = CODE / 'experiments/insplad_fullview'
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location('recorded_fullview_experiment', EXPERIMENT / 'run.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def config(data_path=None, output_root=None, paths_file=None):
    external = load_paths(paths_file)
    cfg = json.loads((ROOT / 'configs/a5000.json').read_text())
    cfg['weights'] = external['encoder_weights']
    cfg['data_path'] = str(Path(data_path).resolve()) if data_path else external['data_root']
    cfg['results_root'] = str(Path(output_root).resolve()) if output_root else str(Path(external['output_root']) / 'a5000')
    if cfg['categories'] != list(experiment.FAULT5_COUNTS):
        raise ValueError('Require exactly the five recorded categories')
    if sha(cfg['weights']) != cfg['weights_sha256']:
        raise ValueError('Pretrained DINOv2 checksum mismatch')
    verify_sources()
    return cfg

def verify_dataset(cfg):
    """Compare split, labels, relative paths and bytes independent of mount path."""
    base = Path(cfg['data_path'])
    expected = read_csv(ROOT / 'configs/data_manifest_relative.csv')
    actual = []
    for category in cfg['categories']:
        for split in ('train', 'test'):
            folder = base / category / split
            if not folder.is_dir():
                raise ValueError('Missing dataset folder: ' + str(folder))
            for state in sorted(p for p in folder.iterdir() if p.is_dir()):
                if split == 'train' and state.name != 'good':
                    raise ValueError('Only good images are allowed for training')
                for p in sorted(state.iterdir()):
                    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}:
                        actual.append(dict(relative_path=p.relative_to(base).as_posix(), category=category, split=split, defect_type=state.name, label=str(int(state.name != 'good')), size_bytes=str(p.stat().st_size)))
    fields = ('relative_path', 'category', 'split', 'defect_type', 'label', 'size_bytes')
    normalize = lambda rows: sorted(tuple(r[k] for k in fields) for r in rows)
    if normalize(actual) != normalize(expected):
        raise ValueError('Dataset paths, sizes, split or labels differ from recorded data')
    return dict(train_images=9006, test_normal=2254, test_anomalous=402, test_images=2656, identity='relative path, label and size; reference images also content-hashed')

def references(cfg):
    refs = read_csv(ROOT / 'configs/normal_references.csv')
    for row in refs:
        row['path'] = str(Path(cfg['data_path']) / row['relative_path'])
        if row['split'] != 'train' or row['label'] != '0' or sha(row['path']) != row['image_sha256']:
            raise ValueError('Normal calibration reference differs: ' + row['relative_path'])
    return refs
