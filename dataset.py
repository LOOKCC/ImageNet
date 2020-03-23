import os
import torch
from torch.utils import data
from PIL import Image
import cv2
import json
import numpy as np

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_path, transform, train=True):
        'Initialization'
        fin = open(list_path, 'r')
        self.meta = fin.readlines()
        fin.close()
        print('load meta: ', list_path)
        self.transform = transform
        self.train = train

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.meta)

    def __getitem__(self, index):
        'Generates one sample of data'
        image_path, label = self.meta[index].split()
        label = int(label)
        image = Image.open(image_path) 
        if self.transform is not None:
            sample = self.transform(image)
        if self.train:
            return sample, label
        else:
            return image_path, sample, label