import torch
import torchvision
import numpy as np
import os,sys
sys.path.append('../')
from utils import *

class CreateModelNet():

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
                if file.endswith('.off'):
                    self.files.append((path+file,self.classes[category]))

        self.write('/home/harsh/M40')

    def preprocess(self,filepath):
        verts,faces = read_off(filepath)
        if self.transforms is not None:
            pointcloud = self.transforms((verts,faces))
            return pointcloud
        return verts

    def write(self,new_root):
        for filepath in self.files:
            temp = filepath[0].split('/')[-3:]
            temp = '/'.join(temp)
            new_path = new_root+'/'+ temp
            print(filepath[0])
            new = self.preprocess(filepath[0])
            with open(new_path,'w') as f:
                for i in range(len(new)):
                    for k in range(len(new[i])):
                        f.write(str(new[i][k].item()) + ' ')
                    f.write('\n')

class ModelNet10(torch.utils.data.Dataset):

    def __init__(self,root_dir,folder = 'train'):
        self.root_dir = root_dir
        self.classes = {}
        for i,class_name in enumerate(os.listdir(root_dir)):
            self.classes[class_name] = i
        self.files = []
        # self.classes = [i for i,class_name in enumerate(os.listdir(root_dir))]
        for category in os.listdir(root_dir):
            path = root_dir+'/'+category+'/'+folder+'/'
            for file in os.listdir(path):
                if file.endswith('.off'):
                    self.files.append((path+file,self.classes[category]))

    def __len__(self):
        return len(self.files)

    def preprocess(self,filepath):
        pointcloud = read_new(filepath)
        return torch.from_numpy(pointcloud)

    def __getitem__(self,index):
        filepath = self.files[index][0]
        category = self.files[index][1]
        # print(filepath)
        pointcloud = self.preprocess(filepath)
        return pointcloud.T,category


class TrainTransforms(object):
    def __call__(self,pointcloud):
        return torch.from_numpy(noise_pointcloud(rotate_pointcloud(normalize_pointcloud(pointcloud))))

class TestTransforms(object):
    def __call__(self,pointcloud):
        return torch.from_numpy(normalize_pointcloud(pointcloud))
