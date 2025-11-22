import os
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

def load_pascal_voc_boxes(xml_file):
    """Load ground-truth boxes from a Pascal VOC XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes)

def compute_iou_vectorized(proposals, gt_boxes):
    """
    proposals: (N,4) [x1,y1,x2,y2]
    gt_boxes: (M,4)
    returns: (N,M) IoU matrix
    """
    N = proposals.shape[0]
    M = gt_boxes.shape[0]

    # Expand dims to broadcast
    boxes1 = np.expand_dims(proposals, 1)  # (N,1,4)
    boxes2 = np.expand_dims(gt_boxes, 0)   # (1,M,4)

    x1 = np.maximum(boxes1[:,:,0], boxes2[:,:,0])
    y1 = np.maximum(boxes1[:,:,1], boxes2[:,:,1])
    x2 = np.minimum(boxes1[:,:,2], boxes2[:,:,2])
    y2 = np.minimum(boxes1[:,:,3], boxes2[:,:,3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:,:,2]-boxes1[:,:,0]) * (boxes1[:,:,3]-boxes1[:,:,1])
    area2 = (boxes2[:,:,2]-boxes2[:,:,0]) * (boxes2[:,:,3]-boxes2[:,:,1])

    union_area = area1 + area2 - inter_area
    iou = np.zeros_like(inter_area)
    mask = union_area > 0
    iou[mask] = inter_area[mask] / union_area[mask]
    return iou

def xywh_to_xyxy(proposals):
    """
    Convert [x,y,w,h] or [x,y,w,h,score] -> [x1,y1,x2,y2,(score)]
    """
    if proposals.shape[1] == 4:
        x, y, w, h = proposals.T
        return np.stack([x, y, x + w, y + h], axis=1)
    elif proposals.shape[1] == 5:
        x, y, w, h, score = proposals.T
        return np.stack([x, y, x + w, y + h, score], axis=1)
    else:
        raise ValueError("Proposals must have 4 or 5 columns")

def evaluate_topN_vectorized(proposals_xyxy, gt_boxes, topN_list, iou_thresholds=[0.3,0.5,0.7]):
    """
    Compute top-N recalls and MABO.
    proposals_xyxy: (N,4) or (N,5)
    gt_boxes: (M,4)
    topN_list: list of ints
    Returns dict: {N: {'recall_0.5':..., 'avg_iou':...}}
    """
    # Sort by score if present
    if proposals_xyxy.shape[1] == 5:
        proposals_xyxy = proposals_xyxy[proposals_xyxy[:,4].argsort()[::-1]]

    results = {}
    for N in topN_list:
        topN = proposals_xyxy[:N,:4]  # ignore score for IoU
        if len(gt_boxes) == 0 or len(topN) == 0:
            best_ious = np.array([])
        else:
            ious = compute_iou_vectorized(topN, gt_boxes)  # (N,M)
            best_ious = ious.max(axis=0)  # max IoU per GT box

        recall_dict = {f"recall_{t:.2f}": float(np.mean(best_ious >= t) if len(best_ious) > 0 else 0.0)
                       for t in iou_thresholds}
        avg_iou = float(best_ious.mean() if len(best_ious) > 0 else 0.0)
        recall_dict['avg_iou'] = avg_iou
        results[N] = recall_dict
    return results

# -------------------
# Main evaluation
# -------------------
def main(proposals_dir, annotations_dir, save_dir,
         max_proposals=2000, step=10, iou_thresholds=[0.3,0.5,0.7], method_name="selective_search"):

    os.makedirs(save_dir, exist_ok=True)
    output_csv_path = os.path.join(save_dir, f"{method_name}_evaluation.csv")

    proposal_files = sorted([f for f in os.listdir(proposals_dir) if f.endswith('.npy')])
    all_results = []

    for pf in tqdm(proposal_files, desc=f"Evaluating {method_name}"):
        image_id = os.path.splitext(pf)[0]
        proposals = np.load(os.path.join(proposals_dir, pf))

        # Convert to x1,y1,x2,y2 (preserves score if present)
        proposals_xyxy = xywh_to_xyxy(proposals)

        # Top-N list
        topN_list = list(range(step, min(max_proposals,len(proposals_xyxy))+1, step))

        # Load GT boxes
        xml_file = os.path.join(annotations_dir, image_id + '.xml')
        if not os.path.exists(xml_file):
            print(f"Warning: XML not found for {image_id}, skipping.")
            continue
        gt_boxes = load_pascal_voc_boxes(xml_file)
        num_gt = len(gt_boxes)

        # Evaluate top-N
        metrics = evaluate_topN_vectorized(proposals_xyxy, gt_boxes, topN_list, iou_thresholds)
        for N in topN_list:
            result = {
                'image_id': image_id,
                'method': method_name,
                'num_proposals': N,
                'num_gt_boxes': num_gt,
            }
            result.update(metrics[N])
            all_results.append(result)

    # Save CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv_path, index=False)
    print(f"Evaluation complete. Results saved to {output_csv_path}")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--method_name', type=str, required=True,
                        choices=["selective_search", "edgeboxes"],
                        help="Region proposal method to evaluate")
    parser.add_argument('--max_proposals', type=int, default=2000)
    parser.add_argument('--step', type=int, default=50)
    args = parser.parse_args()

    # Modular paths based on method_name
    proposals_dir = os.path.join("data/proposals", args.method_name)
    annotations_dir = "/dtu/datasets1/02516/potholes/annotations"
    save_dir = os.path.join("data/metrics", args.method_name)

    os.makedirs(save_dir, exist_ok=True)

    main(
        proposals_dir=proposals_dir,
        annotations_dir=annotations_dir,
        save_dir=save_dir,
        max_proposals=args.max_proposals,
        step=args.step,
        method_name=args.method_name
    )
