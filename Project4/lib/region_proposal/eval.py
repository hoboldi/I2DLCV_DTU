# evaluate_proposals.py
import os
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd

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

def compute_iou(box, gt_box):
    x1 = max(box[0], gt_box[0])
    y1 = max(box[1], gt_box[1])
    x2 = min(box[2], gt_box[2])
    y2 = min(box[3], gt_box[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    box_area = (box[2]-box[0])*(box[3]-box[1])
    gt_area = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
    
    union_area = box_area + gt_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def evaluate_image(proposals, gt_boxes, iou_threshold=0.5):
    recalls = []
    ious = []
    for gt in gt_boxes:
        iou_max = max([compute_iou(gt, prop) for prop in proposals])
        ious.append(iou_max)
        recalls.append(int(iou_max >= iou_threshold))
    recall = sum(recalls) / len(gt_boxes) if len(gt_boxes) > 0 else 0
    avg_iou = sum(ious) / len(gt_boxes) if len(gt_boxes) > 0 else 0
    return recall, avg_iou

def main(proposals_dir, annotations_dir, output_csv, save_dir,
         max_proposals=2000, step=10, iou_thresholds=[0.3, 0.5, 0.7], method_name="selective_search"):

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Construct file paths for saving proposals and evaluation CSV
    proposals_npz_path = os.path.join(save_dir, f"{method_name}_proposals.npz")
    output_csv_path = os.path.join(save_dir, f"{method_name}_evaluation.csv")

    # The rest of your code can now use these paths
    proposal_files = sorted([f for f in os.listdir(proposals_dir) if f.endswith('.npy')])
    all_results = []
    all_proposals_dict = {}

    for pf in proposal_files:
        image_id = os.path.splitext(pf)[0]
        proposals = np.load(os.path.join(proposals_dir, pf))
        all_proposals_dict[image_id] = proposals  # store for npz

        xml_file = os.path.join(annotations_dir, image_id + '.xml')
        if not os.path.exists(xml_file):
            print(f"Warning: XML not found for {image_id}, skipping.")
            continue
        gt_boxes = load_pascal_voc_boxes(xml_file)
        num_gt = len(gt_boxes)

        for N in range(step, min(max_proposals, len(proposals)) + 1, step):
            topN = proposals[:N]
            recall_dict = {}
            for t in iou_thresholds:
                recall_t, _ = evaluate_image(topN, gt_boxes, iou_threshold=t)
                recall_dict[f"recall_{t:.2f}"] = recall_t

            # Average IoU (all best overlaps) is always computed without threshold
            _, avg_iou = evaluate_image(topN, gt_boxes, iou_threshold=0)  # threshold ignored

            result = {
                'image_id': image_id,
                'method': method_name,
                'num_proposals': N,
                'avg_iou': avg_iou,
                'num_gt_boxes': num_gt
            }
            result.update(recall_dict)
            all_results.append(result)

    # Save evaluation CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv_path, index=False)
    print(f"Evaluation complete. Results saved to {output_csv_path}")

    # Save all proposals
    np.savez_compressed(proposals_npz_path, **all_proposals_dict)
    print(f"All proposals saved to {proposals_npz_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposals_dir', type=str, required=True)
    parser.add_argument('--annotations_dir', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--proposals_npz', type=str, required=True,
                        help="Path to save all proposals in compressed npz")
    parser.add_argument('--max_proposals', type=int, default=2000)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--method_name', type=str, default="selective_search")
    args = parser.parse_args()

    main(
        proposals_dir=args.proposals_dir,
        annotations_dir=args.annotations_dir,
        output_csv=args.output_csv,
        proposals_npz_path=args.proposals_npz,
        max_proposals=args.max_proposals,
        step=args.step,
        method_name=args.method_name
    )
