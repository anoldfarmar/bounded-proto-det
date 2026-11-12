"""Explicit complete-file input view, separate from the original cropped transform."""
from torchvision import transforms

MEAN=[.485,.456,.406]
STD=[.229,.224,.225]
SPEC=dict(mode='whole_file_resize_square',resize=[392,392],crop=None,padding=None,
          retained_field_of_view_fraction=1.,aspect_ratio_preserved=False,
          tensor_before_normalize=True,mean=MEAN,std=STD)


def transform():
    return transforms.Compose([transforms.Resize((392,392)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
