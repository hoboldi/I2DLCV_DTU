class PotholeDataset(Dataset):
    def __init__(self, root, proposal_dir, img_ids,
                 proposals_per_image=64,
                 pos_thresh=0.7,
                 neg_thresh=0.5):

        self.root = root
        self.images = os.path.join(root, "images")
        self.annotations = os.path.join(root, "annotations")
        self.proposals_dir = proposal_dir
        self.ids = img_ids

        self.proposals_per_image = proposals_per_image
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Build class mapping (background = 0)
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
        elif len(p) == 6:
            x, y, w, h = p[:4]
        else:
            raise ValueError("Unknown proposal format:", p)

        return [int(x), int(y), int(x+w), int(y+h)]

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Load image
        img_path = os.path.join(self.images, img_id + ".png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load ground truth
        objects = parse_voc_xml(os.path.join(self.annotations, img_id + ".xml"))
        for obj in objects:
            obj["label_idx"] = self.class_to_idx[obj["class"]]

        # Load proposals
        raw_props = np.load(
            os.path.join(self.proposals_dir, img_id + ".npy"),
            allow_pickle=True
        )
        proposals = [self.to_xyxy(p) for p in raw_props]

        # Label proposals
        labeled = []
        for p in proposals:
            best_i, best_label = 0, 0
            for gt in objects:
                i = iou(p, gt["bbox"])
                if i > best_i:
                    best_i = i
                    best_label = gt["label_idx"]

            if best_i >= self.pos_thresh:
                labeled.append((p, best_label))
            elif best_i < self.neg_thresh:
                labeled.append((p, 0))

        #  Adressing the class imbalance issue (25% pos, 75% neg)
        total = self.proposals_per_image
        num_pos = int(0.25 * total)
        num_neg = total - num_pos

        positives = [(p, l) for (p, l) in labeled if l != 0]
        negatives = [(p, 0) for (p, l) in labeled if l == 0]

        random.shuffle(positives)
        random.shuffle(negatives)

        samples = positives[:num_pos] + negatives[:num_neg]

        crops, labels = [], []

        for box, label in samples:
            x1, y1, x2, y2 = box
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = self.transform(crop)
            crops.append(crop)
            labels.append(label)

        if len(crops) == 0:
            crops = [torch.zeros(3, 224, 224)]
            labels = [0]

        return torch.stack(crops), torch.tensor(labels), img_id, box
