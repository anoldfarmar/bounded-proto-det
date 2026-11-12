"""Print the last formally completed run; supports shell command substitution."""
import argparse
from pathlib import Path
from paths import load_paths

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('method', choices=['a5000', 'costfilter'])
parser.add_argument('--paths')
args = parser.parse_args()
pointer = Path(load_paths(args.paths)['output_root']) / args.method / 'LATEST_TRAIN.txt'
if not pointer.is_file():
    parser.error('No completed formal training run: ' + str(pointer))
print(pointer.read_text().strip())
