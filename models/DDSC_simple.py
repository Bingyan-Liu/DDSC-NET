import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,F_int,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi




class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sconv1 = SeparableConv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels + out_channels)
        self.sconv2 = SeparableConv2d(in_channels + out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels + out_channels*2)
        self.sconv3 = SeparableConv2d(in_channels + out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(in_channels + out_channels * 3)
        self.sconv4 = SeparableConv2d(in_channels + out_channels * 3, out_channels, kernel_size=3, stride=1, padding=1,
                                      bias=False)
        self.bn5 = nn.BatchNorm2d(in_channels + out_channels * 4)
        self.sconv5 = SeparableConv2d(in_channels + out_channels * 4, out_channels, kernel_size=3, stride=1, padding=1,
                                      bias=False)
        self.conv1_1 = nn.Conv2d(in_channels + out_channels*5,out_channels,1,1,bias=False)


    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu(x1)
        out1 = self.sconv1(x1)
        inp2 = torch.cat((x,out1),1)
        x2 = self.bn2(inp2)
        x2 = self.relu(x2)
        out2 = self.sconv2(x2)
        inp3 = torch.cat((inp2,out2),1)
        x3 = self.bn3(inp3)
        x3 = self.relu(x3)
        out3 = self.sconv3(x3)
        inp4 = torch.cat((inp3, out3), 1)
        x4 = self.bn4(inp4)
        x4 = self.relu(x4)
        out4 = self.sconv4(x4)
        inp5 = torch.cat((inp4, out4), 1)
        x5 = self.bn5(inp5)
        x5 = self.relu(x5)
        out5 = self.sconv5(x5)
        out = torch.cat((inp5,out5),1)
        out = self.conv1_1(out)

        return out

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=2, stride=2, padding=0, bias=True)

    def forward(self, x):
        out = self.convTrans(x)
        return out



class DDSC_Net(nn.Module):

    def __init__(self, in_channel=3,out_channel=3):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(DDSC_Net, self).__init__()
        self.subsample = nn.AvgPool2d(kernel_size=2,stride=2)
        self.subsample_conv1 = nn.Conv2d(3,64,3,1,padding=1,bias=False)
        self.subsample_conv2 = nn.Conv2d(3,128,3,1,padding=1,bias=False)
        self.subsample_conv3 = nn.Conv2d(3,256,3,1,padding=1,bias=False)


        self.conv1 = nn.Conv2d(in_channel, 32, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        # do relu here
        self.pool0 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.block1 = Block(32, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.block2 = Block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.block3 = Block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.block4 = Block(256, 512)
        self.upcon4 = TransitionUp(512,256)
        self.block5 = Block(512, 256)
        self.upcon5 = TransitionUp(256,128)
        self.block6 = Block(256, 128)
        self.upcon6 = TransitionUp(128,64)
        self.block7 = Block(128, 64)
        self.upcon7 = TransitionUp(64,32)
        self.bn8 = nn.BatchNorm2d(64)
        # do relu here
        self.conv8 = nn.Conv2d(64, 32, 3, 1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(32)
        # do relu here
        self.conv9 = nn.Conv2d(32, out_channel, 3, 1, padding=1, bias=False)

    def forward(self, inp):
        x = self.bn1(inp)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x0 = self.conv2(x)
        x = self.pool0(x0)
        x1_1 = self.block1(x)
        x = self.pool1(x1_1)
        x2_1 = self.block2(x)
        x= self.pool2(x2_1)
        x3_1 = self.block3(x)
        x = self.pool3(x3_1)
        x4_1 = self.block4(x)
        up4 = self.upcon4(x4_1)
        x = torch.cat((x3_1,up4),1)
        x5_1 = self.block5(x)
        up5 = self.upcon5(x5_1)
        x = torch.cat((x2_1,up5),1)
        x6_1 = self.block6(x)
        up6 = self.upcon6(x6_1)
        x = torch.cat((x1_1,up6),1)
        x7_1 = self.block7(x)
        up7 = self.upcon7(x7_1)
        x = torch.cat((x0,up7),1)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.conv9(x)




        return x
