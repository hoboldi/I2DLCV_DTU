# edgebox_canny_sobel.py
import os
import cv2
import numpy as np
from tqdm import tqdm

def compute_orientation_map(gray):
    """Compute edge orientation map using Sobel gradients."""
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    orientation = np.arctan2(gy, gx)  # radians
    return orientation

def extract_edgebox_proposals(img_path, max_proposals=2000):
    # Read image
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute edges using Canny
    edges = cv2.Canny(img_gray, 50, 150).astype(np.float32) / 255.0

    # Compute orientation map
    orientation_map = compute_orientation_map(img_gray)

    # Create EdgeBoxes object
    edge_boxes = cv2.ximgproc.createEdgeBoxes(maxBoxes=max_proposals)

    # Get proposals
    boxes, scores = edge_boxes.getBoundingBoxes(edges, orientation_map)

    # Convert to numpy array [x1, y1, x2, y2]
    proposals = []
    for box in boxes:
        x, y, w, h = box
        proposals.append([x, y, x + w, y + h])
    return np.array(proposals)

def main(images_dir, save_dir, max_proposals=2000):
    os.makedirs(save_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])

    for img_file in tqdm(image_files):
        img_path = os.path.join(images_dir, img_file)
        proposals = extract_edgebox_proposals(img_path, max_proposals=max_proposals)

        save_path = os.path.join(save_dir, os.path.splitext(img_file)[0] + ".npy")
        np.save(save_path, proposals)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default="/dtu/datasets1/02516/potholes/images")
    parser.add_argument('--save_dir', type=str, default="data/proposals/edgeboxes")
    parser.add_argument('--max_proposals', type=int, default=500)
    args = parser.parse_args()

    main(args.images_dir, args.save_dir, args.max_proposals)