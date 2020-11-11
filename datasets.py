from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from skimage import io
from torch.utils.data import Dataset
import  matplotlib.pyplot as plt
import numpy as np

class Images_Dataset(Dataset):
    """Class for getting data as a Dict
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        sample : Dict of images and labels"""

    def __init__(self, images_dir, labels_dir, transformI = None, transformM = None):

        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.transformI = transformI
        self.transformM = transformM


    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):

        for i in range(len(self.images_dir)):
            image = Image.open(self.images_dir[i]).convert('RGB')
            # #改为单通道图片进行训练
            # r,g,b = image.split()
            # image = r
            label = Image.open(self.labels_dir[i])
            imgs = self.transforms(image)
            label = self.transforms(label)
            sample = {'images': imgs, 'labels': label}

        return sample


class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir,transformI = None,transformM = None):
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transformI = transformI
        self.transformM = transformM
        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
               torchvision.transforms.Resize((240,240)),
                torchvision.transforms.ToTensor(),
            ])

        if self.transformM:
            self.lx = self.transformM
        else:
            pass


    def __len__(self):

        return len(self.images)

    def __getitem__(self, i):
        i1 = Image.open(self.images_dir + self.images[i]).convert('RGB')
        r, g, b = i1.split()
        # i1 = r
        l1 = Image.open(self.labels_dir + self.labels[i])
        l1 = l1.resize((240, 240))
        img1 = np.asarray(l1)
        new_label = np.zeros(img1.shape,dtype=np.int64)
        new_label[img1 > 128 ] = 0
        new_label[img1 == 128 ] = 1
        new_label[img1 < 128 ] = 2
        l1 = new_label

        self.lx = torch.from_numpy(np.array(l1)).long().unsqueeze(0)
        return self.tx(i1), self.lx

##read dataset for pre processing
class Images_Dataset_folder_pre(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir,transformI = None,transformM = None):
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transformI = transformI
        self.transformM = transformM

        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
               torchvision.transforms.Resize((400,400)),
                torchvision.transforms.ToTensor(),
            ])

        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
               torchvision.transforms.Resize((400,400)),

                torchvision.transforms.ToTensor(),
            ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, i):
        i1 = Image.open(self.images_dir + self.images[i]).convert('RGB')
        l1 = Image.open(self.labels_dir + self.labels[i])
        return self.tx(i1), self.lx(l1)