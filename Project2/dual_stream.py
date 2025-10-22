import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class No_Leakage(torch.utils.data.Dataset):
    def __init__(self, split, transform, data_path='/dtu/datasets1/02516/ucf101_noleakage'):
        'Initialization'
        self.transform = transform
        os.path.join(data_path, split)

        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path + '/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]

        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y


def preprocess(size=128,batch_size=64):
    train_transform = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor()])

    trainset = No_Leakage(split='train', transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)

    valset = No_Leakage(split='val', transform=val_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=3)

    testset = No_Leakage(split='test', transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)