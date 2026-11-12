"""Checks for data leakage boundaries, geometry, source math, and stable losses."""
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from insplad_data import CATEGORIES, COUNTS, SyntheticNormals, TestImages, evaluation_loader, transform
from model_adapter import cost_inputs, losses, loss_terms

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE/'config.json').read_text())


def upstream_cost(en, de):
    """Literal shape operations from upstream, retained as an independent oracle."""
    batch = en[0].shape[0]
    volumes = []
    for rec_feat, org_feat in zip(en, de):
        pi = org_feat.reshape(batch, -1, 784).permute(0, 2, 1)
        pr = rec_feat.reshape(batch, -1, 784).permute(0, 2, 1)
        pi = pi/(pi.norm(dim=-1, keepdim=True)+1e-8)
        pr = pr/(pr.norm(dim=-1, keepdim=True)+1e-8)
        cost = 1-torch.bmm(pi, pr.permute(0, 2, 1))
        _, indices = torch.topk(cost, 768, dim=-1, largest=False, sorted=False)
        selected = torch.gather(cost, -1, torch.sort(indices, dim=-1)[0])
        volumes.append(selected.view(batch, 28, 28, 768).unsqueeze(1))
    result = torch.stack(volumes, dim=0).squeeze(2)
    result = torch.nn.UpsamplingBilinear2d(scale_factor=64/28)(
        result.permute(0, 1, 4, 2, 3).reshape(-1, 768, 28, 28))
    return result.view(2, batch, 768, 64, 64).permute(1, 0, 2, 3, 4).permute(0, 2, 1, 3, 4)


class Protocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.data = SyntheticNormals(CFG)

    def test_normal_only_training_scope(self):
        self.assertEqual(len(self.data), 9006)
        self.assertEqual(len(self.data.dtd), 5640)
        for category, counts in zip(CATEGORIES, COUNTS):
            rows = [r for r in self.data.rows if r['category'] == category]
            self.assertEqual(len(rows), counts[0])
            self.assertTrue(all(r['label'] == 0 and '/train/good/' in r['path'] for r in rows))
        self.assertEqual(len(TestImages(CFG['data_path'])), 2656)

    def test_full_view_matches_published_A_transform(self):
        from torchvision import transforms
        old = transforms.Compose([transforms.Resize((392, 392)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        array = np.zeros((233, 517, 3), np.uint8)
        array[:, :30, 0], array[:, -30:, 1] = 255, 255
        a = transform()(Image.fromarray(array))
        torch.testing.assert_close(a, old(Image.fromarray(array)), rtol=0, atol=0)
        self.assertGreater(float(a[0, :, 0].mean()), 1.)
        self.assertGreater(float(a[1, :, -1].mean()), 1.)

    def test_evaluation_batching_matches_original_A(self):
        data = TestImages(CFG['data_path'])
        batches = evaluation_loader(data, CFG).batch_sampler
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(flattened, list(range(2656)))
        self.assertEqual(CFG['eval_batch_size'], 32)
        for batch in batches:
            self.assertEqual(len({data.rows[i]['category'] for i in batch}), 1)
            self.assertLessEqual(len(batch), 32)
        for category, counts in zip(CATEGORIES, COUNTS):
            labels = [r['label'] for r in data.rows if r['category'] == category]
            self.assertEqual(labels, [0]*counts[1]+[1]*counts[2])

    def test_synthesis_both_sources_and_normal_identity(self):
        for source in ['dtd', 'self_shuffle']:
            item = self.data.item(0, force_source=source, force_anomaly=True)
            self.assertTrue(item['mask'].any())
            mask = item['mask'][0].numpy().astype(bool)
            self.assertTrue(np.array_equal(item['synthetic'][~mask], item['original'][~mask]))
        item = self.data.item(0, force_source='dtd', force_anomaly=False)
        self.assertFalse(item['mask'].any())
        self.assertTrue(np.array_equal(item['synthetic'], item['original']))

    def test_augmenters_uint8_compatibility(self):
        sample = np.full((392, 392, 3), 128, dtype=np.uint8)
        for augmenter in self.data.augmenters:
            result = augmenter(image=sample)
            self.assertEqual(result.shape, sample.shape)
            self.assertEqual(result.dtype, np.uint8)

    def test_worker_and_resume_sample_determinism(self):
        ids = [0, 2300, 3000, 4000]
        single = list(DataLoader(Subset(self.data, ids), batch_size=2, num_workers=0))
        multi = list(DataLoader(Subset(self.data, ids), batch_size=2, num_workers=2))
        for a, b in zip(single, multi):
            torch.testing.assert_close(a['image'], b['image'], rtol=0, atol=0)
            torch.testing.assert_close(a['mask'], b['mask'], rtol=0, atol=0)
        resumed = next(iter(DataLoader(Subset(self.data, ids[2:]), batch_size=2, num_workers=0)))
        torch.testing.assert_close(single[1]['image'], resumed['image'], rtol=0, atol=0)

    def test_cost_volume_matches_upstream_including_batch_one(self):
        for batch in [1, 2]:
            torch.manual_seed(3)
            en = [torch.randn(batch, 768, 28, 28) for _ in range(2)]
            de = [torch.randn_like(e) for e in en]
            actual, features, guidance, baseline = cost_inputs(en, de, CFG)
            expected = upstream_cost(en, de)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(tuple(actual.shape), (batch, 768, 2, 64, 64))
            self.assertEqual(tuple(features[0].shape), (batch, 784, 768))
            self.assertEqual(tuple(guidance.shape), (batch, 2, 64, 64))
            self.assertEqual(tuple(baseline.shape), (batch, 1, 392, 392))

    def test_loss_finite_at_saturated_probabilities(self):
        for foreground in [0, 1]:
            logits = torch.full((1, 2, 64, 64), -30.)
            logits[:, foreground] = 30.
            logits.requires_grad_()
            classes = torch.zeros((1, 5), requires_grad=True)
            mask = torch.full((1, 1, 392, 392), float(foreground))
            loss, _ = loss_terms(logits, classes, mask, torch.tensor([0]), losses.SSIM())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertTrue(torch.isfinite(classes.grad).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
