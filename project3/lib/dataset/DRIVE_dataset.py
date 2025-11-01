import torch
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import torchvision.transforms.functional as TF

class DRIVE_dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        """
        root_dir: path to the DRIVE dataset folder
        train: True for training, False for test
        transform: torchvision transforms to apply to both image and mask
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        if train:
            self.image_dir = os.path.join(root_dir, 'training/images')
            self.mask_dir = os.path.join(root_dir, 'training/1st_manual')
        else:
            self.image_dir = os.path.join(root_dir, 'test/images')
            self.mask_dir = os.path.join(root_dir, 'test/mask')  # test masks

        # Get sorted list of file paths
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.tif')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, '*.gif')))

        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks do not match"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # grayscale mask

        # Optional transform
        if self.transform:
            X = self.transform(image)
            Y = self.transform(mask)
        else:
            X = TF.to_tensor(image)
            Y = TF.to_tensor(mask)

        return X, Y
