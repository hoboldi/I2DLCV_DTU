#Helper functions
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
