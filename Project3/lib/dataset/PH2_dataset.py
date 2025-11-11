import torch
from torch.utils.data import Dataset
import os, glob
from PIL import Image
import torchvision.transforms.functional as TF


class PH2_dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_transform=None,
        mask_transform=None,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 1337,
    ):
        """
        PH2 dataset loader with the same interface as DRIVE_dataset.

        Args:
            root_dir: path to PH2_Dataset_images (contains IMD### folders)
            split: 'train', 'val', or 'test'
            image_transform, mask_transform: optional torchvision transforms
            val_ratio: fraction of data for validation
            test_ratio: fraction of data for testing
            seed: random seed for deterministic splits
        """
        super().__init__()
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"

        self.root_dir = root_dir
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # find all IMDxxx case directories
        case_dirs = sorted(glob.glob(os.path.join(root_dir, "IMD*")))

        image_paths, mask_paths = [], []
        for case_dir in case_dirs:
            image_file = glob.glob(os.path.join(case_dir, "*_Dermoscopic_Image", "*.bmp"))
            mask_file = glob.glob(os.path.join(case_dir, "*_lesion", "*.bmp"))

            if len(image_file) == 1 and len(mask_file) == 1:
                image_paths.append(image_file[0])
                mask_paths.append(mask_file[0])
            else:
                print(f"[Warning] {case_dir} — found {len(image_file)} images, {len(mask_file)} masks")

        assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"

        # deterministic split
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(len(image_paths), generator=g).tolist()

        n_total = len(idx)
        n_test = int(round(n_total * test_ratio))
        n_val = int(round(n_total * val_ratio))
        if n_test + n_val > n_total:
            overflow = n_test + n_val - n_total
            n_test -= min(overflow, n_test)
            overflow -= min(overflow, n_test)
            n_val -= overflow

        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        chosen_idx = split_map[split]

        self.image_paths = [image_paths[i] for i in chosen_idx]
        self.mask_paths = [mask_paths[i] for i in chosen_idx]

        print(f"[PH2_dataset] {split}: {len(self.image_paths)} samples "
              f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.image_transform:
            X = self.image_transform(image)
        else:
            X = TF.to_tensor(image)

        if self.mask_transform:
            Y = self.mask_transform(mask)
        else:
            Y = TF.to_tensor(mask)

        return X, Y
