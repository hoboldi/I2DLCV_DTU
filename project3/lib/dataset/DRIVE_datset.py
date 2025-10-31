import torch
import os
import glob
import numpy as np
from PIL import Image

DATA_PATH = '/dtu/datasets1/02516/'

class CustomSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):
        """
        train: True for training set, False for test set
        transform: transformations to apply to both images and masks
        """
        self.transform = transform

        if train:
            data_path = os.path.join(DATA_PATH, 'training')
            self.image_paths = sorted(glob.glob(os.path.join(data_path, 'images', '*.tif')))
            # You can choose either 'mask' or '1st_manual' for masks
            self.mask_paths = sorted(glob.glob(os.path.join(data_path, 'mask', '*.gif')))
        else:
            data_path = os.path.join(DATA_PATH, 'test')
            self.image_paths = sorted(glob.glob(os.path.join(data_path, 'images', '*.tif')))
            self.mask_paths = sorted(glob.glob(os.path.join(data_path, 'mask', '*.gif')))

        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks should match!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        # Apply transform if provided
        if self.transform:
            X = self.transform(image)
            Y = self.transform(mask)
        else:
            X = torch.tensor(np.array(image), dtype=torch.float32)
            Y = torch.tensor(np.array(mask), dtype=torch.float32)
        return X, Y
