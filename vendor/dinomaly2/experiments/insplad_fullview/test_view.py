"""Check field-of-view retention, spatial alignment and explicit no-crop transform."""
import unittest
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from view import transform,MEAN,STD


class ViewTests(unittest.TestCase):
    def test_non_square_boundaries_remain_visible(self):
        image=np.zeros((180,360,3),np.uint8)
        image[:20]=[255,0,0];image[-20:]=[0,255,0];image[30:150,:30]=[0,0,255];image[30:150,-30:]=[255,255,0]
        output=transform()(Image.fromarray(image))*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None]
        for y,x,color in [(3,196,[1,0,0]),(388,196,[0,1,0]),(196,3,[0,0,1]),(196,388,[1,1,0])]:
            torch.testing.assert_close(output[:,y,x],torch.tensor(color,dtype=torch.float32),atol=1e-6,rtol=0)
        self.assertEqual(tuple(output.shape),(3,392,392))

    def test_no_crop_or_padding_operation(self):
        self.assertEqual([type(x).__name__ for x in transform().transforms],['Resize','ToTensor','Normalize'])

    def test_identical_to_resize392_crop392_without_pixel_removal(self):
        image=Image.fromarray(np.random.RandomState(5).randint(0,256,(131,287,3),dtype=np.uint8))
        expected=transforms.Compose([transforms.Resize((392,392)),transforms.ToTensor(),transforms.CenterCrop(392),transforms.Normalize(MEAN,STD)])(image)
        torch.testing.assert_close(transform()(image),expected,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
