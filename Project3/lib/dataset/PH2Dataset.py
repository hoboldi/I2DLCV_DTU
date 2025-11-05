import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

DATA_PATH = '/dtu/datasets1/02516/PH2_Dataset_images'

class PH2Dataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def get_PH2_datasets(transform=None, test_size=0.2, val_size=0.1, seed=42):

    image_paths = sorted(glob(os.path.join(DATA_PATH, 'IMD*/IMD*_Dermoscopic_Image/*.bmp')))
    mask_paths  = sorted(glob(os.path.join(DATA_PATH, 'IMD*/IMD*_lesion/*.bmp')))

    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
    print("Example image path:", image_paths[0])
    print("Example mask path:", mask_paths[0])


    assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"

    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=seed
    )

    val_ratio = val_size / (1 - test_size)
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        train_imgs, train_masks, test_size=val_ratio, random_state=seed
    )

    train_dataset = PH2Dataset(train_imgs, train_masks, transform=transform)
    val_dataset = PH2Dataset(val_imgs, val_masks, transform=transform)
    test_dataset = PH2Dataset(test_imgs, test_masks, transform=transform)

    return train_dataset, val_dataset, test_dataset
