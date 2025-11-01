import torch
import os
import glob
import PIL.Image as Image
import torchvision.transforms.functional as TF

class PH2_dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to the folder containing all IMDxxx case folders
        transform: torchvision transforms to apply to both image and mask
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get a sorted list of all case folders, e.g. IMD001, IMD002, ..., IMD437
        self.case_dirs = sorted(glob.glob(os.path.join(root_dir, 'IMD*')))

        # Build lists of image and mask file paths
        self.image_paths = []
        self.mask_paths = []

        for case_dir in self.case_dirs:
            image_file = glob.glob(os.path.join(case_dir, '*_Dermoscopic_Image', '*.bmp'))
            mask_file = glob.glob(os.path.join(case_dir, '*_lesion', '*.bmp'))

            if len(image_file) == 1 and len(mask_file) == 1:
                self.image_paths.append(image_file[0])
                self.mask_paths.append(mask_file[0])
            else:
                # Warn if files are missing or multiple files exist
                print(f"Warning: case {case_dir} has {len(image_file)} image(s) and {len(mask_file)} mask(s)")

    def __len__(self):
        """Returns the total number of cases in the dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load one sample: image and mask, apply transforms, return tensors"""
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Open images
        image = Image.open(image_path).convert("RGB")  # ensure 3 channels
        mask = Image.open(mask_path).convert("L")      # grayscale (binary mask)

        # Apply transforms if provided
        if self.transform:
            X = self.transform(image)
            Y = self.transform(mask)
        else:
            # Convert to PyTorch tensors (default float, scale 0-1)
            X = TF.to_tensor(image)
            Y = TF.to_tensor(mask)

        return X, Y
