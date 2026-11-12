"""Portable packaging contracts, recorded scores and independent metric examples."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from paths import load_paths, verify_sources
from verify_results import auc, compute

class PortableContract(unittest.TestCase):
    def test_tied_scores_and_reversed_order(self):
        self.assertEqual(auc([0, 1], [.2, .2]), .5)
        self.assertEqual(auc([0, 1], [.2, .3]), 1.)
        self.assertEqual(auc([0, 1], [.3, .2]), 0.)

    def test_all_three_metrics_match_original_reports(self):
        result = compute()
        cases = [('A5000', 'a5000_summary.json'), ('NCut_full_fusion', 'ncut_summary.json'), ('CostFilter_full_fusion', 'costfilter_summary.json')]
        expected_means = [.8532808054567191, .8855578404547944, .8639312584635712]
        for (name, _), expected in zip(cases, expected_means):
            self.assertAlmostEqual(result['methods'][name]['mean']['I-AUROC'], expected, places=14)

    def test_external_paths_with_spaces_are_data(self):
        value = json.loads((ROOT / 'configs/paths.example.json').read_text())
        value['data_root'] = '/tmp/图片 with spaces/insplad'
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'paths.json'
            file.write_text(json.dumps(value))
            self.assertEqual(load_paths(file)['data_root'], value['data_root'])

    def test_recorded_method_sources_unchanged(self):
        report = verify_sources()
        self.assertEqual(report['a5000_recorded_files'], 24)
        self.assertEqual(report['costfilter_recorded_files'], 5)

    def test_no_weights_or_dataset_images_in_source_tree(self):
        for p in ROOT.rglob('*'):
            if p.is_file() and '.git' not in p.parts:
                self.assertNotIn(p.suffix.lower(), {'.pt', '.pth', '.ckpt', '.safetensors', '.jpg', '.jpeg', '.png'}, str(p))

if __name__ == '__main__':
    unittest.main(verbosity=2)
