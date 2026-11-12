"""Frozen Dinomaly2 + the supplied CostFilter network; no downloads."""
from argparse import Namespace
import hashlib
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from insplad_data import CATEGORIES, VIEW, source_module

network = source_module('costfilter_network', 'unet3d_att_dino_channel_test_48_min.py')
losses = source_module('costfilter_losses', 'loss.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def baseline_modules(cfg):
    repo = str(Path(cfg['dinomaly2_repo']).resolve())
    # CostFilter's bundled models/dataset/utils have the same generic names;
    # import Dinomaly2 only on this path and load CostFilter files by filename.
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import dinomaly_2D
    import utils
    import insplad_adapter
    assert Path(dinomaly_2D.__file__).parent == Path(repo), '错误的 Dinomaly2 模块路径'
    assert Path(utils.__file__).parent == Path(repo), '错误的 utils 模块路径'
    return dinomaly_2D, utils, insplad_adapter


def load_baseline(cfg, device):
    baseline, _, _ = baseline_modules(cfg)
    state = torch.load(cfg['baseline_checkpoint'], map_location='cpu', weights_only=True)
    args = state['config']
    if state['step'] != 5000 or args['categories'] != CATEGORIES or args['input_view'] != VIEW:
        raise ValueError('必须使用五类、完整视野的 A5000 检查点')
    if state['weights_sha256'] != cfg['weights_sha256']:
        raise ValueError('编码器权重不匹配')
    args = dict(args, weights=cfg['weights'])
    baseline.setup_seed(cfg['seed'])
    model = baseline.build_model(Namespace(**args), 'cpu')
    model.bottleneck.load_state_dict(state['bottleneck'], strict=True)
    model.decoder.load_state_dict(state['decoder'], strict=True)
    return model.requires_grad_(False).to(device).eval()


def build_filter(cfg):
    model = network.DiscriminativeSubNetwork_3d_att_dino_channel(
        in_channels=cfg['cost_channels'], out_channels=2,
        base_channels=cfg['base_channels'], cls_classes=len(CATEGORIES))
    # Same Conv/BatchNorm initialization as the supplied training entry point.
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv3d)):
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            torch.nn.init.constant_(layer.weight, 1)
            torch.nn.init.constant_(layer.bias, 0)
    return model


@torch.no_grad()
def cost_inputs(en, de, cfg):
    """Preserve upstream query=decoder, reference=encoder, top768 index order."""
    if len(en) != 2 or len(de) != 2:
        raise ValueError('CostFilter requires exactly two feature groups')
    batch = en[0].shape[0]
    volumes = []
    for target, reconstruction in zip(en, de):
        if target.shape != (batch, 768, 28, 28) or reconstruction.shape != target.shape:
            raise ValueError('Expected (B,768,28,28) grouped features')
        query = reconstruction.flatten(2).transpose(1, 2)
        reference = target.flatten(2).transpose(1, 2)
        query = query / (query.norm(dim=-1, keepdim=True) + 1e-8)
        reference = reference / (reference.norm(dim=-1, keepdim=True) + 1e-8)
        cost = 1 - torch.bmm(query, reference.transpose(1, 2))
        indices = torch.topk(cost, cfg['cost_channels'], dim=-1, largest=False, sorted=False).indices
        selected = cost.gather(-1, indices.sort(dim=-1).values)
        plane = selected.view(batch, 28, 28, cfg['cost_channels']).permute(0, 3, 1, 2)
        volumes.append(F.interpolate(plane, size=64, mode='bilinear', align_corners=True))
    volume = torch.stack(volumes, dim=2)
    baseline = torch.stack([F.interpolate(1-F.cosine_similarity(e, d).unsqueeze(1),
                                        size=392, mode='bilinear', align_corners=True)
                            for e, d in zip(en, de)]).mean(0)
    # The source training entry uses the 28-grid diagonal map -> 64 guidance.
    # The source evaluator instead uses 392 -> 64. Match train and test here
    # using the direct 28 -> 64 path and disclose this consistency correction.
    diagonal28 = torch.stack([1-F.cosine_similarity(e, d).unsqueeze(1)
                              for e, d in zip(en, de)]).mean(0)
    guidance = F.interpolate(diagonal28, size=64, mode='bilinear', align_corners=True).repeat(1, 2, 1, 1)
    features = [e.flatten(2).transpose(1, 2) for e in en]
    return volume, features, guidance, baseline


def loss_terms(logits, cls_logits, mask, class_ids, ssim):
    probability = F.interpolate(logits.softmax(dim=1), size=392, mode='bilinear', align_corners=True)
    anomaly = probability[:, 1:2]
    # Upstream dynamic gamma uses raw class logits, not softmax confidence.
    gamma = (3-cls_logits.max(dim=1).values).clamp(min=1.5)
    gamma = torch.where(cls_logits.argmax(dim=1) == class_ids, gamma,
                        torch.full_like(gamma, 3.5))
    focal = losses.FocalLoss_gamma()
    # Preserve upstream FocalLoss while guarding fractional powers of 1-pt
    # when softmax saturates. The tiny clamp keeps the focal pt below one.
    probability = probability.clamp(min=1e-5, max=1-2e-5)
    fl = torch.stack([focal(probability[i:i+1], mask[i:i+1], gamma[i])
                      for i in range(len(mask))]).mean()
    values = dict(focal=fl, ssim=ssim(anomaly, mask), mse=F.mse_loss(anomaly, mask),
                  iou=losses.SoftIoULoss(anomaly, mask),
                  classification=F.cross_entropy(cls_logits, class_ids))
    total = values['focal'] + .1*values['ssim'] + values['mse'] + \
        .1*values['iou'] + .5*values['classification']
    return total, values
