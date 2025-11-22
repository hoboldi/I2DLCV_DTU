import cv2
import os
import numpy as np
from tqdm import tqdm

def extract_selective_search_proposals(image_path, max_proposals=2000, resize_to=None):
    """
    Extract bounding boxes using OpenCV's Selective Search.
    Optionally resizes the image for faster processing and rescales boxes back.
    Returns boxes in format [x1, y1, x2, y2].
    """
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]

    # Optional resize
    if resize_to is not None:
        img_resized = cv2.resize(img, resize_to)
        scale_x = orig_w / resize_to[0]
        scale_y = orig_h / resize_to[1]
    else:
        img_resized = img
        scale_x = 1.0
        scale_y = 1.0

    # Create Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_resized)
    ss.switchToSelectiveSearchFast()  # very important for speed

    # Extract raw proposals
    rects = ss.process()

    # Limit count
    rects = rects[:max_proposals]

    # Convert to [x1, y1, x2, y2] and rescale if needed
    boxes = []
    for (x, y, w, h) in rects:
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)
        boxes.append([x1, y1, x2, y2])

    return np.array(boxes)


def main(images_dir, save_dir, max_proposals=2000, resize_to=None):
    os.makedirs(save_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))])

    for img_file in tqdm(image_files):
        img_path = os.path.join(images_dir, img_file)

        boxes = extract_selective_search_proposals(
            img_path,
            max_proposals=max_proposals,
            resize_to=resize_to
        )

        save_path = os.path.join(save_dir, os.path.splitext(img_file)[0] + '.npy')
        np.save(save_path, boxes)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default="/dtu/datasets1/02516/potholes/images")
    parser.add_argument('--save_dir', type=str, default="data/proposals/selective_search")
    parser.add_argument('--max_proposals', type=int, default=2000)
    parser.add_argument('--resize_to', type=int, nargs=2, default=None,
                        help="Optional: width height. E.g. --resize_to 600 600")
    args = parser.parse_args()

    resize_tuple = tuple(args.resize_to) if args.resize_to is not None else None

    main(args.images_dir, args.save_dir, args.max_proposals, resize_to=resize_tuple)
