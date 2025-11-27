import os
import glob
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim

# ============================================================
# 1. CNN to classify object proposals
# ============================================================
class ProposalClassifier(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        backbone = resnet18(weights=weights)

        # Remove final FC layer
        self.feature = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, crops):
        x = self.feature(crops)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================================================

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []

    for obj in root.findall("object"):
        cls = obj.find("name").text
        bnd = obj.find("bndbox")
        xmin = int(bnd.find("xmin").text)
        ymin = int(bnd.find("ymin").text)
        xmax = int(bnd.find("xmax").text)
        ymax = int(bnd.find("ymax").text)
        objects.append({"class": cls, "bbox": [xmin, ymin, xmax, ymax]})

    return objects


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0

def proposal_collate(batch):
    crops = torch.cat([item[0] for item in batch], dim=0)
    labels = torch.cat([item[1] for item in batch], dim=0)
    img_ids = [item[2] for item in batch]
    boxes = [item[3] for item in batch]
    return crops, labels, img_ids, boxes

# ============================================================
# 2. Dataset
# ============================================================
class PotholeDataset(Dataset):
    def __init__(self, root, proposal_dir, img_ids,
                 proposals_per_image=64,
                 pos_ratio=0.25,
                 pos_thresh=0.7,
                 neg_thresh=0.5):

        self.root = root
        self.images = os.path.join(root, "images")
        self.annotations = os.path.join(root, "annotations")
        self.proposals_dir = proposal_dir
        self.ids = img_ids

        self.proposals_per_image = proposals_per_image
        self.num_pos = int(proposals_per_image * pos_ratio)
        self.num_neg = proposals_per_image - self.num_pos
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        classes = set()
        for img_id in img_ids:
            xml = os.path.join(self.annotations, img_id + ".xml")
            for obj in parse_voc_xml(xml):
                classes.add(obj["class"])
        self.classes = sorted(list(classes))
        self.class_to_idx = {c: i+1 for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.ids)

    def to_xyxy(self, p):
        if len(p) == 4:
            x, y, w, h = p
        elif len(p) == 5:  
            x, y, w, h = p[:4]
        else:
            raise ValueError("Unknown proposal format:", p)
        return [int(x), int(y), int(x+w), int(y+h)]

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.images, img_id + ".png")
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Ground truth objects
        objects = parse_voc_xml(os.path.join(self.annotations, img_id + ".xml"))
        for obj in objects:
            obj["label_idx"] = self.class_to_idx[obj["class"]]

        # load proposals
        raw_props = np.load(os.path.join(self.proposals_dir, img_id + ".npy"), allow_pickle=True)
        proposals = [self.to_xyxy(p) for p in raw_props]

        # Label proposals
        positives, negatives = [], []

        for p in proposals:
            best_iou = 0
            assigned_label = 0
            for gt in objects:
                i = iou(p, gt["bbox"])
                if i > best_iou:
                    best_iou = i
                    assigned_label = gt["label_idx"]

            if best_iou >= self.pos_thresh:
                positives.append((p, assigned_label))
            elif 0.1 < best_iou < self.neg_thresh:
                negatives.append((p, 0))


        # sample positives (up to num_pos)
        if len(positives) > self.num_pos:
            positives = random.sample(positives, self.num_pos)

        # sample negatives (up to num_neg)
        if len(negatives) > self.num_neg:
            negatives = random.sample(negatives, self.num_neg)

        samples = positives + negatives
        random.shuffle(samples)

        crops, labels, boxes = [], [], []

        for box, label in samples:
            x1, y1, x2, y2 = box
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = self.base_transform(crop)
            crops.append(crop)
            labels.append(label)
            boxes.append(box)

        if len(crops) == 0:
            return (torch.zeros(1, 3, 224, 224),
                    torch.tensor([0]),
                    img_id,
                    [[0,0,0,0]])

        return torch.stack(crops), torch.tensor(labels), img_id, boxes


# ============================================================
root = "/dtu/datasets1/02516/potholes"
proposal_dir = "/zhome/7b/3/168395/02516/Project4_ObjectDetection/proposals/selective_search"

image_paths = sorted(glob.glob(os.path.join(root, "images", "*.png")))
all_ids = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

random.shuffle(all_ids)

n = len(all_ids)
train_ids = all_ids[:int(0.7*n)]
val_ids   = all_ids[int(0.7*n):int(0.8*n)]
test_ids  = all_ids[int(0.8*n):]

train_dataset = PotholeDataset(root, proposal_dir, train_ids)
val_dataset = PotholeDataset(root, proposal_dir, val_ids)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                          collate_fn=proposal_collate, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                        collate_fn=proposal_collate, num_workers=4, pin_memory=True)

# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProposalClassifier(num_classes=len(train_dataset.classes)+1).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)


scaler = torch.cuda.amp.GradScaler()

proposal_batch = 32
epochs = 15
best_acc = 0.0  

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

def accuracy_fn(logits, labels):
    preds = logits.argmax(1)
    return (preds == labels).sum().item(), labels.numel()

for epoch in range(epochs):

    # --------------------------
    # TRAIN
    # --------------------------
    model.train()
    total_train_loss = total_correct = total_samples = 0

    for crops, labels, img_ids, _ in train_loader:
        for i in range(0, len(crops), proposal_batch):
            batch_crops = crops[i:i+proposal_batch].to(device)
            batch_labels = labels[i:i+proposal_batch].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(batch_crops)
                loss = criterion(logits, batch_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            c, t = accuracy_fn(logits, batch_labels)
            total_correct += c
            total_samples += t

    train_acc = total_correct / total_samples

    # --------------------------
    # VALIDATION
    # --------------------------
    model.eval()
    total_val_loss = val_correct = val_samples = 0

    with torch.no_grad():
        for crops, labels, img_ids, _ in val_loader:
            for i in range(0, len(crops), proposal_batch):
                bc = crops[i:i+proposal_batch].to(device)
                bl = labels[i:i+proposal_batch].to(device)

                logits = model(bc)
                loss = criterion(logits, bl)
                total_val_loss += loss.item()

                c, t = accuracy_fn(logits, bl)
                val_correct += c
                val_samples += t

    val_acc = val_correct / val_samples

    # Logging
    train_loss_history.append(total_train_loss)
    val_loss_history.append(total_val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {total_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss:   {total_val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")


    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "proposal_classifier_best.pth")
        print("✓ Saved new best model (based on accuracy)")


# Save final model
torch.save(model.state_dict(), "proposal_classifier_final.pth")
print("Model saved: proposal_classifier_final.pth")


# ============================================================
# PLOT
# ============================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.legend(); plt.grid(); plt.title("Loss")

plt.subplot(1,2,2)
plt.plot([a*100 for a in train_acc_history], label="Train Acc")
plt.plot([a*100 for a in val_acc_history], label="Val Acc")
plt.legend(); plt.grid(); plt.title("Accuracy")

plt.savefig("training_curves.png")
print("Saved training_curves.png")
