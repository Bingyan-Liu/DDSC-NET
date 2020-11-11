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
from datasets import Images_Dataset, Images_Dataset_folder
import cv2
from model.DDSC import DDSC_Net
# from model.DDSC_simple import XNet

from torch.utils.tensorboard import SummaryWriter

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss=weight_decay*reg_loss
        return reg_loss

    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")



def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def dice_loss(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1.0 - (((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)))

gamma=2,
alpha=0.25
reduction='elementwise_mean'

set_seed() #设置随机种子
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_checkpoint = './checkpoints/'
#参数设置
MAX_EPOCH = 30
BATCH_SIZE = 8
LR = 0.00001
log_interval = 10
val_interval = 1
shuffle = True
num_workers = 0
random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))
#可视化数量
vis_num = 10
#---正则化参数--------
weight_decay = 0
#------step 1/5 数据-------------

# data_path = './'
train_dir = './data/crop/img/img800/img_30000/'
label_dir = './data/crop/img/img800/label_30000/'
# valid_dir = './val'
valid_size = 0.1


#构建MyDataset实例
Training_Data = Images_Dataset_folder(train_dir,label_dir)
num_train = len(Training_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
#构建Dataloader
# train_loader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
# valid_loader = DataLoader(dataset=valid_data,batch_size=BATCH_SIZE)

train_loader = DataLoader(Training_Data, batch_size=BATCH_SIZE, sampler=train_sampler,
                                           num_workers=num_workers, )
print('Train data number are %d'%len(train_loader))

valid_loader = DataLoader(Training_Data, batch_size=BATCH_SIZE, sampler=valid_sampler,
                                           num_workers=num_workers, )
print(('Valid data number are %d'%len(valid_loader)))
print('DataLoader Done')
#--------------step 2/5 模型---------------
# net = U_Net
net = DDSC_Net(3,3)

#--------------step 3/5 损失函数-----------

criterion = nn.CrossEntropyLoss()

criterion.to(device)
net.to(device)

#--------------step 4/5 优化器-------------
optimizer = optim.Adam(net.parameters(),lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.1)#设置学习率下降策略



#--------------正则化------------------------

if weight_decay > 0:
    reg_loss = Regularization(net, weight_decay, p=1).to(device)
else:
    print("no regularization")

#--------------step 5/5 训练---------------
train_curve = list()
valid_curve = list()
train_dice_curve = list()
valid_dice_curve = list()
writer = SummaryWriter(comment='train_comment',filename_suffix='_train_suffix')

# #---------------断点恢复-----------------
# path_checkpoint = './checkpoints/checkpoint_21_epoch.pkl'
# checkpoint = torch.load(path_checkpoint)
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch']
# scheduler.last_epoch = start_epoch
# for epoch in range(start_epoch+1,MAX_EPOCH):
# log_dir = './log/'

for epoch in range(MAX_EPOCH):
    print('epoch %d'%(epoch))
    train_loss_total = 0.
    train_dice_total = 0.
    valid_loss_mean = 0.
    loss_mean = 0.
    correct = 0.
    step = 0

    start = datetime.datetime.now()
    net.train()
    for i,data in enumerate(train_loader):
        print(i)
        step += 1
        inputs,labels = data

        labels = torch.squeeze(labels,1)
        n1 = labels.numpy()
        inputs = inputs.to(device)
        labels = labels.to(device)

        #forward
        outputs = net(inputs)

        ###-------计算ACC------
        pred = torch.max(outputs, 1)[1]
        train_acc = (pred == labels).sum().item() / np.array(pred.size())

        #backward
        optimizer.zero_grad()
        loss = criterion(outputs,labels)

        if weight_decay > 0:
            loss = loss + reg_loss(net)
        loss.backward()

        #update weights
        optimizer.step()

        #打印训练信息
        train_loss_total += loss.item()
        train_curve.append(loss.item())
        if i % 10 ==0:
            writer.add_scalar('Train', loss.item(), epoch)
        if (i+1) % log_interval == 0 :
            loss_mean = loss_mean / log_interval
        print('Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] loss: {:4f},mean_loss:{:.4f} ACC{}'.format(
                epoch,MAX_EPOCH,i+1,len(train_loader),loss.item(),train_loss_total/(i+1),train_acc
            ))

        end = datetime.datetime.now()
        print("Running Time：" + str((end - start).seconds) + "s")
    if (epoch+1) % 1 == 0:
        checkpoint = {'model_state_dict': net.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch

        }
        path_checkpoint = './checkpoints/checkpoint_{}_epoch.pkl'.format(epoch)
        torch.save(checkpoint,path_checkpoint)
        print('Checkpoint {} saved !'.format(epoch))
    scheduler.step() #更新学习率
    with SummaryWriter(comment='Net1')as w:
        w.add_graph(net, (inputs,))


    #validdate the model
    if (epoch+1)%val_interval == 0:

        net.eval()
        valid_loss_total = 0.
        valid_dice_total = 0.
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                # inputs = inputs
                labels = torch.squeeze(labels)
                if BATCH_SIZE == 1:
                    labels = torch.unsqueeze(labels, 0)

                labels = labels.long()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                outputs = net(inputs)
                loss = criterion(outputs,labels)

                pred = torch.max(outputs, 1)[1]
                valid_loss_total += loss.item()
                # print(j)


                # 打印训练信息
            valid_loss_mean += valid_loss_total/len(valid_loader)
            valid_curve.append(valid_loss_mean)
            print('Valid:Epoch[{:0>3}/{:0>3}]  mean_loss: {:4f} '.format(
                        epoch, MAX_EPOCH, valid_loss_mean
                    ))


#plot curve
train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)

valid_x = np.arange(1,len(valid_curve)+1)*train_iters*val_interval
valid_y = valid_curve

plt.plot(train_x,train_y,label = 'Train')
plt.plot(valid_x,valid_y,label = 'Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.title('Plot in {} epochs'.format(MAX_EPOCH))
plt.show()
torch.cuda.empty_cache()
writer.close()
writer.close()
