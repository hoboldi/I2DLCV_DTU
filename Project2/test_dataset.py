# Project2/test_dataset.py
import os
from torchvision import transforms as T

ROOT = '/dtu/datasets1/02516/ucf101_noleakage'  # change if needed
SPLIT = 'val'  # set to 'train' or 'val' depending on what you have

def main():
    import torch
    print("CUDA available:", torch.cuda.is_available())

    print("Checking dataset dir:", ROOT)
    if not os.path.exists(ROOT):
        print("ERROR: Root path does not exist:", ROOT)
        return

    # quick listing
    print("Top-level folders:", os.listdir(ROOT))

    # check metadata csv
    meta_dir = os.path.join(ROOT, 'metadata')
    csv_path = os.path.join(meta_dir, f'{SPLIT}.csv')
    if not os.path.exists(csv_path):
        print("ERROR: metadata CSV not found:", csv_path)
        print("Available metadata files:", os.listdir(meta_dir) if os.path.exists(meta_dir) else 'metadata folder missing')
        return
    else:
        print("Found metadata:", csv_path)
        with open(csv_path, 'r') as f:
            for i, line in enumerate(f):
                print("  meta line", i+1, ":", line.strip())
                if i >= 4:
                    break

    # import the dataset classes
    try:
        from datasets import FrameImageDataset, FrameVideoDataset
    except Exception as e:
        print("ERROR importing datasets.py:", e)
        return

    transform = T.Compose([T.Resize((64,64)), T.ToTensor()])

    # Try FrameImageDataset
    try:
        ds_img = FrameImageDataset(root_dir=ROOT, split=SPLIT, transform=transform)
        print("FrameImageDataset length:", len(ds_img))
        if len(ds_img) > 0:
            x, label = ds_img[0]
            print("FrameImage sample shape:", getattr(x, 'shape', 'unknown'), "label:", label)
    except Exception as e:
        print("FrameImageDataset error:", type(e).__name__, e)

    # Try FrameVideoDataset (stacked)
    try:
        ds_vid = FrameVideoDataset(root_dir=ROOT, split=SPLIT, transform=transform, stack_frames=True)
        print("FrameVideoDataset length:", len(ds_vid))
        if len(ds_vid) > 0:
            frames, label = ds_vid[0]
            print("FrameVideo sample type:", type(frames), "shape:", getattr(frames, 'shape', 'unknown'), "label:", label)
    except Exception as e:
        print("FrameVideoDataset error:", type(e).__name__, e)

if __name__ == '__main__':
    main()