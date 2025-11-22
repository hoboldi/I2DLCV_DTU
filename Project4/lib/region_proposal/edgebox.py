# edgebox_canny_sobel.py
import os
import cv2
import numpy as np
from tqdm import tqdm

#IMPORTANT: if you want to run this file you need to extract the opencv SED model and copy it into the data folder of region proposal
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.yml.gz")
sed = cv2.ximgproc.createStructuredEdgeDetection(MODEL_PATH)
edge_boxes = cv2.ximgproc.createEdgeBoxes()  # you can set maxBoxes here or in getBoundingBoxes

def extract_edgebox_proposals(img_path, max_proposals=2000, resize_to=None):
    # Read image
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]

    # Optional resize for speed
    if resize_to is not None:
        img = cv2.resize(img, resize_to)

    img_float = img.astype(np.float32) / 255.0

    # Structured Edge Detection
    edges = sed.detectEdges(img_float)
    orientation_map = sed.computeOrientation(edges)

    # Get EdgeBoxes proposals
    edge_boxes.setMaxBoxes(max_proposals)
    boxes, scores = edge_boxes.getBoundingBoxes(edges, orientation_map)

    # Convert to numpy array [x1, y1, x2, y2]
    proposals = []
    for (box, score) in zip(boxes, scores):
        x, y, w, h = box
        # Scale back if needed
        if resize_to is not None:
            scale_x = orig_w / resize_to[0]
            scale_y = orig_h / resize_to[1]
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
        proposals.append([x, y, x + w, y + h, score])

    proposals = np.array(proposals)
    return np.array(proposals)


def main(images_dir, save_dir, max_proposals=2000, resize_to=None):
    os.makedirs(save_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])

    for img_file in tqdm(image_files):
        img_path = os.path.join(images_dir, img_file)
        proposals = extract_edgebox_proposals(img_path, max_proposals=max_proposals, resize_to=resize_to)

        save_path = os.path.join(save_dir, os.path.splitext(img_file)[0] + ".npy")
        np.save(save_path, proposals)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default="/dtu/datasets1/02516/potholes/images")
    parser.add_argument('--save_dir', type=str, default="data/proposals/edgeboxes")
    parser.add_argument('--max_proposals', type=int, default=2000)
    args = parser.parse_args()

    main(args.images_dir, args.save_dir, args.max_proposals)