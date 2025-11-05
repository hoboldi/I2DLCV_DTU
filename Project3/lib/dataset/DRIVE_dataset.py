import torch
import torchvision.transforms
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import torchvision.transforms.functional as TF

class DRIVE_dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_transform=None,
        mask_transform=None,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 1337,
        masks_subdir: str = "training/1st_manual",
        images_subdir: str = "training/images",
    ):
        """
        A single-source dataset for DRIVE that derives train/val/test splits
        **only from the training set**. The official test/ images are ignored
        here because they do not have public ground-truth labels.

        Args:
            root_dir: Path to the DRIVE dataset folder.
            split: One of {"train","val","test"} for the internal split.
            transform: Optional callable applied to BOTH image and mask.
            val_ratio: Fraction of training set reserved for validation.
            test_ratio: Fraction of training set reserved for test (hold-out).
            seed: RNG seed for deterministic splitting.
            masks_subdir: Subdirectory (under root_dir) where masks are stored.
            images_subdir: Subdirectory (under root_dir) where images are stored.
            binarize_mask: If True, convert mask to {0,1} after ToTensor().
        """
        super().__init__()
        assert split in {"train", "val", "test"}, "split must be 'train', 'val', or 'test'"
        assert 0.0 <= val_ratio < 1.0, "val_ratio must be in [0,1)"
        assert 0.0 <= test_ratio < 1.0 and (val_ratio + test_ratio) < 1.0, "val_ratio + test_ratio must be < 1.0"

        self.root_dir = root_dir
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Always read from the TRAINING tree
        self.image_dir = os.path.join(root_dir, images_subdir)
        self.mask_dir  = os.path.join(root_dir, masks_subdir)

        # Gather paired paths
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.tif")))
        mask_paths  = sorted(glob.glob(os.path.join(self.mask_dir, "*.gif")))

        # Basic check
        assert len(image_paths) == len(mask_paths), (
            f"Mismatch between images ({len(image_paths)}) and masks ({len(mask_paths)}). "
            f"Check your folder structure:\n  images: {self.image_dir}\n  masks:  {self.mask_dir}"
        )

        # Deterministic split
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(len(image_paths), generator=g).tolist()

        n_total = len(idx)
        n_test = int(round(n_total * test_ratio))
        n_val  = int(round(n_total * val_ratio))
        # ensure we don't exceed n_total due to rounding
        if n_test + n_val > n_total:
            # fallback: reduce test first, then val
            overflow = n_test + n_val - n_total
            reduce_test = min(overflow, n_test)
            n_test -= reduce_test
            overflow -= reduce_test
            n_val -= overflow

        test_idx = idx[:n_test]
        val_idx  = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        split_to_indices = {
            "train": train_idx,
            "val":   val_idx,
            "test":  test_idx,
        }

        chosen = split_to_indices[split]
        self.image_paths = [image_paths[i] for i in chosen]
        self.mask_paths  = [mask_paths[i]  for i in chosen]

        # Helpful debug
        # print(f"[DRIVE_dataset] split={split} | N={len(self.image_paths)} | "
        #       f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks do not match after splitting."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        msk = Image.open(self.mask_paths[idx]).convert("L")

        X = self.image_transform(img)   # tensor CxHxW in [0,1]
        Y = self.mask_transform(msk)    # tensor 1xHxW in {0,1}
        return X, Y
