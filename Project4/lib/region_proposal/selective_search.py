import cv2
import os
import numpy as np
from tqdm import tqdm

def extract_selective_search_proposals(image_path, max_proposals=2000):
    """
    Extract bounding boxes using OpenCV's Selective Search.
    Returns a list of [x1, y1, x2, y2] boxes.
    """
    img = cv2.imread(image_path)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()  # can switch to switchToSelectiveSearchQuality()
    rects = ss.process()
    rects = rects[:max_proposals]  # limit number of proposals
    # Convert to [x1, y1, x2, y2]
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    return boxes

def main(images_dir, save_dir, max_proposals=2000):
    os.makedirs(save_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))]
    for img_file in tqdm(image_files):
        img_path = os.path.join(images_dir, img_file)
        boxes = extract_selective_search_proposals(img_path, max_proposals)
        save_path = os.path.join(save_dir, os.path.splitext(img_file)[0] + '.npy')
        np.save(save_path, boxes)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default="/dtu/datasets1/02516/potholes/images")
    parser.add_argument('--save_dir', type=str, default="data/proposals/selective_search")
    parser.add_argument('--max_proposals', type=int, default=500)
    args = parser.parse_args()
    main(args.images_dir, args.save_dir, args.max_proposals)
