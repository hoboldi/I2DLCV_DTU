# edgebox_canny_parallel.py
import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# ------------------------------
# Worker initializer
# ------------------------------
def worker_init():
    """Initialize EdgeBoxes once per process."""
    global edge_boxes
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setAlpha(0.65)
    edge_boxes.setBeta(0.75)
    # maxBoxes will be set per image

# ------------------------------
# Helper functions
# ------------------------------
def compute_orientation_map(gray):
    """Compute edge orientation map using Sobel gradients."""
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.arctan2(gy, gx)

def process_image(args):
    """Process one image and save EdgeBoxes proposals."""
    img_path, save_dir, max_proposals = args

    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: could not read {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    orientation_map = compute_orientation_map(gray)

    edge_boxes.setMaxBoxes(max_proposals)
    boxes, scores = edge_boxes.getBoundingBoxes(edges, orientation_map)

    if len(boxes) == 0:
        proposals = np.zeros((0,5), dtype=np.float32)
    else:
        proposals = np.hstack([
            np.array(boxes, dtype=np.int32),
            np.array(scores, dtype=np.float32).reshape(-1,1)
        ])

    save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[0] + ".npy")
    np.save(save_path, proposals)

# ------------------------------
# Main function
# ------------------------------
def main(images_dir, save_dir, max_proposals=2000, num_workers=4):
    os.makedirs(save_dir, exist_ok=True)
    image_files = sorted([os.path.join(images_dir, f)
                          for f in os.listdir(images_dir)
                          if f.lower().endswith(('.jpg', '.png'))])

    args_list = [(img_path, save_dir, max_proposals) for img_path in image_files]

    # Multiprocessing pool with initializer
    with Pool(processes=num_workers, initializer=worker_init) as pool:
        list(tqdm(pool.imap_unordered(process_image, args_list), total=len(image_files)))

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default="/dtu/datasets1/02516/potholes/images")
    parser.add_argument('--save_dir', type=str, default="data/proposals/edgeboxes")
    parser.add_argument('--max_proposals', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    main(
        images_dir=args.images_dir,
        save_dir=args.save_dir,
        max_proposals=args.max_proposals,
        num_workers=args.num_workers
    )