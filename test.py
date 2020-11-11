import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
import PIL.Image as Image
import torch.nn.functional as F
import datetime
from torch.utils.data.sampler import SubsetRandomSampler
from read_testdata import Images_Dataset, Images_Dataset_folder
import cv2
from model.DDSC import DDSC_Net

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gamma=2,
alpha=0.25
reduction='elementwise_mean'
set_seed() #设置随机种子
dir_checkpoint = './checkpoints/'
#参数设置
MAX_EPOCH = 1000
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
shuffle = True
num_workers = 2
random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))
#可视化数量
vis_num = 10
#------step 1/5 数据-------------
valid_size = 0.
data_path = './'
test_dir = './test/crop/img/'
save_path_img = './test/test1/'

#构建MyDataset实例
Testing_Data = Images_Dataset_folder(test_dir)
num_train = len(Testing_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

test_idx = indices[split:]
test_sampler = SubsetRandomSampler(test_idx)
test_loader = DataLoader(Testing_Data, batch_size=1, sampler=test_sampler,
                                           num_workers=0 )
print('Test data number are %d'%len(test_loader))

print('DataLoader Done')
#--------------step 2/5 模型---------------
# net = U_Net
net = DDSC_Net(3,3)
net.to(device)
path_checkpoint = './checkpoints/...'
checkpoint = torch.load(path_checkpoint,map_location='cpu')
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs,image_name = data
        inputs = inputs.to(device)
        image_name = image_name[0]
        output = net(inputs)
        output = output.to('cpu')
        output3 = np.squeeze(output)
        out3 = np.argmax(output3, 0)
        out5 = F.softmax(output3, dim=0)
        n25 = out5.numpy()
        out6 = torch.max(out5, 0)
        index6 = out6.indices
        n26 = index6.numpy()
        target = n26
        target = target.astype(np.uint8)
        target = cv2.resize(target,(480,480),interpolation=cv2.INTER_CUBIC)
        new_label = np.zeros(target.shape, dtype=np.int64)
        new_label[target == 0] = 255
        new_label[target == 1] = 128
        new_label[target == 2] = 0
        target = new_label
        cv2.imwrite(save_path_img+image_name[:-4]+'.bmp',target)
        print(i)








