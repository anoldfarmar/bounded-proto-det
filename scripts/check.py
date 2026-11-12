"""CPU-only verification of assets, relative data split and normal references."""
import argparse
import json
from common import config, verify_dataset, references, ROOT, sha
from paths import load_paths, checkpoint_for

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths')
    args = parser.parse_args()
    cfg = config(paths_file=args.paths)
    paths = load_paths(args.paths)
    checkpoint = checkpoint_for(paths)
    refs = references(cfg)
    print(json.dumps(dict(dataset=verify_dataset(cfg), normal_references=len(refs), encoder_sha256=sha(cfg['weights']), a5000_checkpoint_sha256=sha(checkpoint), training_started=False), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
