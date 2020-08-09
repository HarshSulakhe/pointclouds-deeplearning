import torch
import torchvision
import numpy as np
import os
from utils import *

class ModelNet10(torch.utils.data.Dataset):

    def __init__(self,root_dir,folder = 'train',transforms = None):
        self.root_dir = root_dir
        self.classes = {}
        for i,class_name in enumerate(os.listdir(root_dir)):
            self.classes[class_name] = i
        self.files = []
        self.transforms = transforms
        # self.classes = [i for i,class_name in enumerate(os.listdir(root_dir))]
        for category in os.listdir(root_dir):
            path = root_dir+'/'+category+'/'+folder+'/'
            for file in os.listdir(path):
                self.files.append((path+file,self.classes[category]))

    def __len__(self):
        return len(self.files)

    def preprocess(self,filepath):
        verts,faces = read_off(filepath)
        if self.transforms is not None:
            pointcloud = self.transforms((verts,faces))
            return pointcloud
        return verts

    def __getitem__(self,index):
        filepath = self.files[index][0]
        category = self.files[index][1]
        pointcloud = self.preprocess(filepath)
        return pointcloud.T,category


class TrainTransforms(object):
    def __call__(self,pointcloud):
        return torch.from_numpy(noise_pointcloud(rotate_pointcloud(normalize_pointcloud(pointcloud))))
