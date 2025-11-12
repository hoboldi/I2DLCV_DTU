import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF

class Clickpoints_dataset(Dataset):
    """
    PH2-based weakly supervised dataset using clickpoints.
    Provides the same interface as DRIVE_dataset with train/val/test split.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_transform=None,
        mask_transform=None,
        total_points: int = 5000,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        ignore_index: int = -1,
    ):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.root_dir = root_dir
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.total_points = total_points
        self.ignore_index = ignore_index

        # Gather all images/masks
        image_paths = sorted(glob.glob(os.path.join(root_dir, 'IMD*/IMD*_Dermoscopic_Image/*.bmp')))
        mask_paths  = sorted(glob.glob(os.path.join(root_dir, 'IMD*/IMD*_lesion/*.bmp')))
        assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"

        # Deterministic split
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(len(image_paths), generator=g).tolist()

        n_total = len(idx)
        n_test  = int(round(n_total * test_ratio))
        n_val   = int(round(n_total * val_ratio))

        # fallback if rounding exceeds total
        if n_test + n_val > n_total:
            overflow = n_test + n_val - n_total
            reduce_test = min(overflow, n_test)
            n_test -= reduce_test
            overflow -= reduce_test
            n_val -= overflow

        test_idx  = idx[:n_test]
        val_idx   = idx[n_test:n_test+n_val]
        train_idx = idx[n_test+n_val:]

        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        chosen_idx = split_map[split]

        self.image_paths = [image_paths[i] for i in chosen_idx]
        self.mask_paths  = [mask_paths[i]  for i in chosen_idx]

        # RNG for sampling
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask (PIL)
        img_pil = Image.open(self.image_paths[idx]).convert("RGB")
        mask_pil = Image.open(self.mask_paths[idx]).convert("L")  # full mask (PIL)
        # Image transform -> tensor CxHxW (resized)
        if self.image_transform:
            img = self.image_transform(img_pil)
        else:
            img = TF.to_tensor(img_pil)

        # Transform full mask to SAME target size as image (nearest to preserve labels)
        if self.mask_transform:
            # mask_transform returns tensor 1xHxW or HxW depending on your function
            full_mask_t = self.mask_transform(mask_pil)  # should be long tensor  HxW (since your mask_tfm squeezes)
            # ensure HxW (no channel dim)
            if full_mask_t.dim() == 3 and full_mask_t.size(0) == 1:
                full_mask_t = full_mask_t.squeeze(0)
        else:
            full_mask_t = TF.to_tensor(mask_pil).squeeze(0).to(torch.long)

        # Now full_mask_t is a tensor (H_target, W_target) with {0,1} values (long)
        # Use numpy boolean for sampling (convert to numpy on CPU)
        mask_bool = full_mask_t.cpu().numpy().astype(bool)

        # Sample positive/negative points in the *transformed* (resized) mask space
        pos_pts, neg_pts = self.sample_points(mask_bool)

        # Create weak mask in the same HxW as full_mask_t, with ignore index
        H, W = mask_bool.shape
        weak_mask = torch.full((H, W), self.ignore_index, dtype=torch.float32)
        if pos_pts.shape[0] > 0:
            weak_mask[pos_pts[:, 1], pos_pts[:, 0]] = 1.0
        if neg_pts.shape[0] > 0:
            weak_mask[neg_pts[:, 1], neg_pts[:, 0]] = 0.0

        # full_mask_t should be long; ensure dtype
        full_mask_tensor = full_mask_t.to(torch.long)

        return img, weak_mask, full_mask_tensor



    # --- Helpers ---
    def sample_points(self, mask_bool):
        H, W = mask_bool.shape
        total_pixels = H * W
        lesion_pixels = mask_bool.sum()
        lesion_fraction = float(lesion_pixels) / float(total_pixels) if total_pixels > 0 else 0.0

        pos_n = max(int(round(self.total_points * lesion_fraction)), 1)
        neg_n = max(self.total_points - pos_n, 1)

        lesion_indices = np.flatnonzero(mask_bool.ravel())
        background_indices = np.flatnonzero((~mask_bool).ravel())

        pos_n = min(pos_n, lesion_indices.size)
        neg_n = min(neg_n, background_indices.size)

        pos_pts = np.empty((0, 2), dtype=int)
        neg_pts = np.empty((0, 2), dtype=int)

        if lesion_indices.size > 0:
            chosen_pos = self.rng.choice(lesion_indices, size=pos_n, replace=False)
            rows, cols = np.unravel_index(chosen_pos, (H, W))
            pos_pts = np.vstack([cols, rows]).T

        if background_indices.size > 0:
            chosen_neg = self.rng.choice(background_indices, size=neg_n, replace=False)
            rows, cols = np.unravel_index(chosen_neg, (H, W))
            neg_pts = np.vstack([cols, rows]).T

        return pos_pts, neg_pts

    def points_to_mask_with_ignore(self, pos_pts, neg_pts, shape):
        H, W = shape
        mask = torch.full((H, W), self.ignore_index, dtype=torch.float32)
        if pos_pts.shape[0] > 0:
            mask[pos_pts[:, 1], pos_pts[:, 0]] = 1.0
        if neg_pts.shape[0] > 0:
            mask[neg_pts[:, 1], neg_pts[:, 0]] = 0.0
        return mask
