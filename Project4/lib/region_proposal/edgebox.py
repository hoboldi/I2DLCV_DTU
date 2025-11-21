import cv2
import os
import numpy as np
from tqdm import tqdm

def extract_edgebox_proposals(image_path, max_proposals=2000, alpha=0.65, beta=0.75):
    """
    Extract bounding boxes using EdgeBoxes (OpenCV).
    Returns a list of [x1, y1, x2, y2] boxes.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edge_detector = cv2.ximgproc.createStructuredEdgeDetection(
        cv2.data.haarcascades + "model.yml")  # You may need a model path for cluster
    edges = edge_detector.detectEdges(np.float32(img) / 255.0)
    orimap = edge_detector.computeOrientation(edges)
    edge_nms = edge_detector.edgesNms(edges, orimap)
    
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(max_proposals)
    edge_boxes.setAlpha(alpha)
    edge_boxes.setBeta(beta)
    
    boxes, _ = edge_boxes.getBoundingBoxes(edge_nms, orimap)
    # Convert to numpy array [x1, y1, x2, y2]
    boxes_np = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    return boxes_np

def main(images_dir, save_dir, max_proposals=2000):
    os.makedirs(save_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))]
    for img_file in tqdm(image_files):
        img_path = os.path.join(images_dir, img_file)
        boxes = extract_edgebox_proposals(img_path, max_proposals)
        save_path = os.path.join(save_dir, os.path.splitext(img_file)[0] + '.npy')
        np.save(save_path, boxes)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default="/dtu/datasets1/02516/potholes/images")
    parser.add_argument('--save_dir', type=str, default="data/proposals/edgeboxes")
    parser.add_argument('--max_proposals', type=int, default=2000)
    args = parser.parse_args()
    main(args.images_dir, args.save_dir, args.max_proposals)