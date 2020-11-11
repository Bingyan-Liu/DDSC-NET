from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from skimage import io
from torch.utils.data import Dataset
import cv2
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

    def __init__(self, images_dir, transformI = None):

        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.transformI = transformI
        # self.transformM = transformM


    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):

        for i in range(len(self.images_dir)):
            image = Image.open(self.images_dir[i]).convert('RGB')
            image_name = self.images_dir[i]
            # image = cv2.imread(self.images_dir[i])
            # image = np.transpose(image, (2, 0, 1))

            # r,g,b = image.split()
            # image = r
            # label = Image.open(self.labels_dir[i])
            # label = cv2.imread(self.labels_dir[i])
            # label = torch.from_numpy(label)

            imgs = self.transforms(image)
            label = []
            sample = {'images': imgs, 'labels': label,'name':image_name}

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

    def __init__(self, images_dir,transformI = None):
        self.images = sorted(os.listdir(images_dir))
        self.images_dir = images_dir
        self.transformI = transformI

        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
               torchvision.transforms.Resize((256,256)),
              #   torchvision.transforms.CenterCrop(960),
              #   torchvision.transforms.RandomRotation((-10,10)),
               # torchvision.transforms.RandomHorizontalFlip(),
               #  torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])


    def __len__(self):

        return len(self.images)

    def __getitem__(self, i):
        i1 = Image.open(self.images_dir + self.images[i]).convert('RGB')
        i2 = self.images[i]
        # i1 = cv2.imread(self.images_dir + self.images[i])
        # i1 = np.transpose(i1, (2, 0, 1))
        # r,g,b = i1.split()
        # i1 = r
        # l1 = Image.open(self.labels_dir + self.labels[i])
        # l1 = cv2.imread(self.labels_dir + self.labels[i])
        # l1 = torch.from_numpy(l1)


        return self.tx(i1),i2


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

    def __init__(self, images_dir,transformI = None):
        self.images = sorted(os.listdir(images_dir))
        self.images_dir = images_dir
        self.transformI = transformI

        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
               torchvision.transforms.Resize((256,256)),
              #   torchvision.transforms.CenterCrop(960),
              #   torchvision.transforms.RandomRotation((-10,10)),
               # torchvision.transforms.RandomHorizontalFlip(),
               #  torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])


    def __len__(self):

        return len(self.images)

    def __getitem__(self, i):
        i1 = Image.open(self.images_dir + self.images[i]).convert('RGB')
        img_size = i1.size
        i2 = self.images[i]
        # i1 = cv2.imread(self.images_dir + self.images[i])
        # i1 = np.transpose(i1, (2, 0, 1))
        # r,g,b = i1.split()
        # i1 = r
        # l1 = Image.open(self.labels_dir + self.labels[i])
        # l1 = cv2.imread(self.labels_dir + self.labels[i])
        # l1 = torch.from_numpy(l1)


        return self.tx(i1),i2,img_size