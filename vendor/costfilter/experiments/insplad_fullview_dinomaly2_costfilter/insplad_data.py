"""Five-class full-view data and normal-only synthetic anomaly supervision."""
import importlib.util
from pathlib import Path

import cv2
import imgaug
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CATEGORIES = ['glass-insulator', 'lightning-rod-suspension',
              'polymer-insulator-upper-shackle', 'vari-grip', 'yoke-suspension']
COUNTS = [(2298, 581, 90), (462, 117, 50), (935, 235, 102),
          (477, 114, 111), (4834, 1207, 49)]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
MEAN, STD = [.485, .456, .406], [.229, .224, .225]
VIEW = dict(mode='whole_file_resize_square', resize=[392, 392], crop=None,
            padding=None, retained_field_of_view_fraction=1.,
            aspect_ratio_preserved=False, tensor_before_normalize=True,
            mean=MEAN, std=STD)


def source_module(name, filename):
    path = Path(__file__).resolve().parents[2] / 'Costfilter_Dinomaly' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


perlin = source_module('costfilter_perlin', 'perlin_noise.py')


def files(folder):
    return sorted(p for p in Path(folder).glob('*')
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)


def samples(root, split):
    result = []
    for class_id, category in enumerate(CATEGORIES):
        states = [Path(root) / category / 'train' / 'good'] if split == 'train' else \
            sorted((p for p in (Path(root) / category / 'test').iterdir()
                    if p.is_dir() and not p.name.startswith('.')),
                   key=lambda p: (p.name != 'good', p.name))
        for state in states:
            for path in files(state):
                result.append(dict(path=str(path), category=category,
                                   class_id=class_id, defect_type=state.name,
                                   label=int(state.name != 'good')))
    return result


def transform():
    # Identical order/interpolation to the existing full-view A5000 input.
    return transforms.Compose([transforms.Resize((392, 392)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    imgaug.seed(seed)


def evaluation_loader(data, cfg):
    # Match original A5000's within-category batches, good images first.
    # Batching all categories together changes CUDA GEMM shapes/roundoff.
    batches = []
    for category in CATEGORIES:
        indices = [i for i, row in enumerate(data.rows) if row['category'] == category]
        size = cfg['eval_batch_size']
        batches.extend(indices[i:i+size] for i in range(0, len(indices), size))
    return DataLoader(data, batch_sampler=batches, num_workers=cfg['workers'], pin_memory=True)


class TestImages(Dataset):
    def __init__(self, root, smoke=False):
        self.rows = samples(root, 'test')
        if smoke:
            self.rows = [next(r for r in self.rows
                             if r['category'] == c and r['label'] == label)
                         for c in CATEGORIES for label in [0, 1]]
        self.transform = transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, index


class SyntheticNormals(Dataset):
    """Upstream texture/self-shuffle + Perlin, applied to complete stored images.

    Perlin is generated at upstream size 448 (divisible by all sampled periods)
    then the binary mask is resized to 392. The image is never cropped.
    A per-(seed,epoch,index) RNG makes worker count and resume independent.
    """
    def __init__(self, cfg, epoch=0):
        self.cfg, self.epoch = cfg, epoch
        self.rows = samples(cfg['data_path'], 'train')
        self.dtd = sorted(Path(cfg['dtd_images']).glob('*/*.jpg'))
        if not self.dtd:
            raise ValueError('DTD 图片为空：' + cfg['dtd_images'])
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        self.augmenters = [iaa.GammaContrast((.5, 2.), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(.5, threshold=(32, 128)), iaa.Posterize(),
                           iaa.Invert(), iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(), iaa.Affine(rotate=(-45, 45))]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.rows)

    def synthesize(self, image, force_source=None, force_anomaly=None):
        selected = np.random.choice(len(self.augmenters), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[i] for i in selected])
        use_dtd = np.random.uniform() < .5 if force_source is None else force_source == 'dtd'
        if use_dtd:
            path = self.dtd[np.random.choice(len(self.dtd))]
            texture = cv2.imread(str(path))
            if texture is None:
                raise ValueError('无法读取 DTD 图片：' + str(path))
            source = cv2.resize(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB), (392, 392))
            # PIL-like imgaug operations (e.g. Autocontrast) reject float32.
            # Keep augmentation uint8, then convert for alpha blending.
            source = aug(image=source).astype(np.float32)
        else:
            # Match the upstream einops layout, including its inner-axis order.
            from einops import rearrange
            source = aug(image=image)
            patches = rearrange(source, '(h gh) (w gw) c -> (h w) gw gh c', gw=49, gh=49)
            order = np.arange(len(patches))
            np.random.shuffle(order)
            source = rearrange(patches[order], '(h w) gw gh c -> (h gh) (w gw) c', h=8, w=8)
            source = source.astype(np.float32)
        scale_x, scale_y = 2**np.random.randint(0, 6), 2**np.random.randint(0, 6)
        noise = perlin.rand_perlin_2d_np((448, 448), (scale_x, scale_y))
        mask = (self.rot(image=noise) > .3).astype(np.uint8)
        mask = cv2.resize(mask, (392, 392), interpolation=cv2.INTER_NEAREST)
        # Upstream unknown categories use mode=3: no assumed foreground gate.
        beta = np.random.uniform(0, .8)
        blend = image * (1-mask[..., None]) + \
            ((1-beta)*source + beta*image) * mask[..., None]
        insert = np.random.uniform() > .5 if force_anomaly is None else force_anomaly
        if not insert:
            mask = np.zeros((392, 392), np.uint8)
            blend = image
        return np.clip(blend, 0, 255).astype(np.uint8), mask, 'dtd' if use_dtd else 'self_shuffle'

    def item(self, index, force_source=None, force_anomaly=None):
        row = self.rows[index]
        seed = int(np.random.SeedSequence([self.cfg['seed'], self.epoch, index]).generate_state(1)[0])
        previous = np.random.get_state()
        try:
            np.random.seed(seed)
            imgaug.seed(seed)
            with Image.open(row['path']) as image:
                original = np.asarray(image.convert('RGB').resize((392, 392), Image.Resampling.BILINEAR))
            synthetic, mask, source = self.synthesize(original, force_source, force_anomaly)
        finally:
            np.random.set_state(previous)
        return dict(image=self.transform(synthetic.copy()),
                    mask=torch.from_numpy(mask.copy()).unsqueeze(0).float(),
                    class_id=row['class_id'], index=index, source=source,
                    synthetic_anomaly=int(mask.any()), original=original,
                    synthetic=synthetic)

    def __getitem__(self, index):
        result = self.item(index)
        del result['original'], result['synthetic']
        return result
