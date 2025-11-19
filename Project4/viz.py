#!/usr/bin/env python3
import os
import glob
import random
import xml.etree.ElementTree as ET
import cv2

# ==========================
# User configuration
# ==========================
images_path = "/dtu/datasets1/02516/potholes/images"        # path to images
annotations_path = "/dtu/datasets1/02516/potholes/annotations"  # path to XML annotations
output_dir = "potholes"         # folder to save visualizations
num_samples = 10                                           # number of images to visualize

os.makedirs(output_dir, exist_ok=True)

# ==========================
# Functions
# ==========================
def parse_voc_xml(xml_file):
    """Parse Pascal VOC XML and return bounding boxes."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        boxes.append({'label': label, 'bbox': [xmin, ymin, xmax, ymax]})
    return boxes

def draw_boxes_on_image(img_path, boxes):
    """Draw bounding boxes on an image."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: could not read {img_path}")
        return None
    for box in boxes:
        xmin, ymin, xmax, ymax = box['bbox']
        label = box['label']
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0,0,255), thickness=2)
        cv2.putText(img, label, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return img

# ==========================
# Main script
# ==========================
# Get all XML files
annotation_files = glob.glob(os.path.join(annotations_path, "*.xml"))
print(f"Found {len(annotation_files)} annotation files.")

# Randomly sample files to visualize
sample_files = random.sample(annotation_files, min(num_samples, len(annotation_files)))

# Process and save
for i, xml_file in enumerate(sample_files):
    image_file = os.path.join(images_path, os.path.basename(xml_file).replace('.xml', '.png'))
    boxes = parse_voc_xml(xml_file)
    img_with_boxes = draw_boxes_on_image(image_file, boxes)
    if img_with_boxes is not None:
        save_path = os.path.join(output_dir, f"viz_{i}.jpg")
        cv2.imwrite(save_path, img_with_boxes)
        print(f"Saved annotated image: {save_path}")

print(f"Done! Annotated images saved to {output_dir}.")
