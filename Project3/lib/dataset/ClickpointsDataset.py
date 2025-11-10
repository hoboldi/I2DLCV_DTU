import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

DATA_PATH = '/dtu/datasets1/02516/PH2_Dataset_images'


def sample_points(mask_bool, total_points, random_state=None, min_pos=1, min_neg=1):
    """Uniformly sample positive/negative points based on lesion fraction."""
    if isinstance(random_state, (int, type(None))):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state

    H, W = mask_bool.shape
    total_pixels = H * W
    lesion_pixels = mask_bool.sum()
    lesion_fraction = float(lesion_pixels) / float(total_pixels) if total_pixels > 0 else 0.0

    pos_n = int(round(total_points * lesion_fraction))
    neg_n = total_points - pos_n
    pos_n = max(pos_n, min_pos)
    neg_n = max(neg_n, min_neg)

    lesion_indices = np.flatnonzero(mask_bool.ravel())
    background_indices = np.flatnonzero((~mask_bool).ravel())

    pos_n = min(pos_n, lesion_indices.size)
    neg_n = min(neg_n, background_indices.size)

    if lesion_indices.size > 0:
        chosen_pos = rng.choice(lesion_indices, size=pos_n, replace=False)
        pos_rows, pos_cols = np.unravel_index(chosen_pos, (H, W))
        pos_pts = np.vstack([pos_cols, pos_rows]).T
    else:
        pos_pts = np.empty((0, 2), dtype=int)

    if background_indices.size > 0:
        chosen_neg = rng.choice(background_indices, size=neg_n, replace=False)
        neg_rows, neg_cols = np.unravel_index(chosen_neg, (H, W))
        neg_pts = np.vstack([neg_cols, neg_rows]).T
    else:
        neg_pts = np.empty((0, 2), dtype=int)

    meta = {
        "total_pixels": int(total_pixels),
        "lesion_pixels": int(lesion_pixels),
        "lesion_fraction": lesion_fraction,
        "positive_points": int(pos_pts.shape[0]),
        "negative_points": int(neg_pts.shape[0])
    }
    return pos_pts, neg_pts, meta


def points_to_mask_with_ignore(pos_pts, neg_pts, shape, ignore_index=-1):
    """
    Create a single-channel mask where:
        - positive clicks = 1
        - negative clicks = 0
        - everything else = ignore_index (e.g., -1)
    """
    H, W = shape
    label = torch.full((H, W), ignore_index, dtype=torch.float32)
    if pos_pts.shape[0] > 0:
        label[pos_pts[:, 1], pos_pts[:, 0]] = 1.0
    if neg_pts.shape[0] > 0:
        label[neg_pts[:, 1], neg_pts[:, 0]] = 0.0
    return label


# ---------- Dataset ----------
class PH2Dataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None,
                 total_points=5000, random_seed=42, ignore_index=-1):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.total_points = total_points
        self.rng = np.random.RandomState(random_seed)
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert mask to boolean
        mask_bool = mask.squeeze().numpy() > 0.5

        # Sample positive and negative points
        pos_pts, neg_pts, meta = sample_points(mask_bool, self.total_points, random_state=self.rng)

        # Convert sampled points into a mask with ignore index
        weak_mask = points_to_mask_with_ignore(pos_pts, neg_pts, mask_bool.shape, self.ignore_index)

        return image, weak_mask


# ---------- Split function ----------
def get_PH2_datasets(transform=None, test_size=0.2, val_size=0.1, seed=42):
    image_paths = sorted(glob(os.path.join(DATA_PATH, 'IMD*/IMD*_Dermoscopic_Image/*.bmp')))
    mask_paths = sorted(glob(os.path.join(DATA_PATH, 'IMD*/IMD*_lesion/*.bmp')))

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
