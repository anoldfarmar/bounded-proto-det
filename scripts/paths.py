"""External assets and portable recorded-source identities (standard library)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'vendor/dinomaly2'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def load_paths(filename=None):
    path = Path(filename) if filename else ROOT / 'configs/paths.local.json'
    if not path.exists() and filename is None:
        path = ROOT / 'configs/paths.example.json'
    data = json.loads(path.read_text())
    required = ['data_root', 'encoder_weights', 'a5000_checkpoint', 'costfilter_checkpoint', 'dtd_images', 'output_root']
    if set(data) != set(required):
        raise ValueError('Paths config must contain exactly: ' + ', '.join(required))
    return {k: str((ROOT / Path(v).expanduser()).resolve()) for k, v in data.items()}

def verify_sources():
    expected = json.loads((ROOT / 'provenance/a5000/source_sha256.json').read_text())
    for name, value in expected.items():
        if sha(CODE / name) != value:
            raise ValueError('Recorded A5000 source differs: ' + name)
    recorded = json.loads((ROOT / 'provenance/source_identity.json').read_text())
    for name, value in recorded['unchanged_costfilter_runtime'].items():
        if sha(ROOT / name) != value:
            raise ValueError('Recorded CostFilter method source differs: ' + name)
    return {'a5000_recorded_files': len(expected), 'costfilter_recorded_files': len(recorded['unchanged_costfilter_runtime'])}

def checkpoint_for(paths, baseline_run=None):
    if baseline_run:
        run = Path(baseline_run).resolve()
        complete = json.loads((run / 'training_complete.json').read_text())
        if complete['smoke'] or complete['step'] != 5000:
            raise ValueError('Require completed formal A5000 run, not smoke')
        return run / 'results/checkpoints/step_005000.pt'
    path = Path(paths['a5000_checkpoint'])
    if sha(path) != '03cc73ef4cf9f9c019b9740ed06ac2cb578624886e62506e9a5c76820aae2941':
        raise ValueError('Recorded A5000 checkpoint mismatch; freshly trained weights require --baseline-run')
    return path
