# 这个是想要分开训练slab的分类和attention map的生成模型，然后再单独训练一个RNN来结合3D的信息，这是前一部分

import sys
import timm
import torch
import torch.nn.functional as F

from torch import nn, optim
from collections import OrderedDict

sys.path.append('../../')
from src.coat import CoaT,coat_lite_mini,coat_lite_small,coat_lite_medium
from src.layers import *

# Hyperparameters

number_of_segments = 3  # Number of segmentation classes
num_classes = 14        # Number of classification classes

class Model(nn.Module):
    def __init__(self, pre=None, num_classes=num_classes, ps=0,mask_head=True, **kwargs):
        super().__init__()

        self.enc = coat_lite_medium(return_interm_layers=True)
        nc = [128,256,320,512]

        feats = 512
        drop = 0.0
        self.mask_head = mask_head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        lstm_embed = feats * 1
        
        self.lstm2 = nn.GRU(lstm_embed, lstm_embed, num_layers=1, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Sequential(
            #nn.Linear(lstm_embed, lstm_embed//2),
            #nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(0.1),
            #nn.LeakyReLU(0.1),
            # nn.Linear(lstm_embed*2, num_classes),
            nn.Linear(nc[-1], num_classes),
        )
        
        if pre is not None:
            sd = torch.load(pre)['model']
            print(self.enc.load_state_dict(sd,strict=False))
        
        self.lstm = nn.ModuleList([LSTM_block(nc[-2]),LSTM_block(nc[-1])])
        self.dec4 = UnetBlock(nc[-1], nc[-2], 384)
        self.dec3 = UnetBlock(384, nc[-3], 192)
        self.dec2 = UnetBlock(192, nc[-4], 96)
        self.fpn = FPN([nc[-1], 384, 192], [32]*3)

        # self.mask_head_3 = self.get_mask_head(nc[-2])
        # self.mask_head_4 = self.get_mask_head(nc[-1])

        self.drop = nn.Dropout2d(ps)
        self.final_conv = nn.Sequential(UpBlock(96+32*3, number_of_segments, blur=True))
        self.up_result=2

    @staticmethod
    def get_mask_head(nb_ft):
        """
        Returns a segmentation head.

        Args:
            nb_ft (int): Number of input features.

        Returns:
            nn.Sequential: Segmentation head.
        """
        # Input - 特征图shape: [batch_size, nb_ft, height, width]
        # Output - 分割mask: [batch_size, 4, height, width], 可以改要输出几个类别的mask

        return nn.Sequential(
            nn.Conv2d(nb_ft, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, number_of_segments, kernel_size=1, padding=0),
        )
    
    def forward(self, x):
        # x: [bs, n_slice_per_c, in_chans, image_size, image_size]

        x = torch.nan_to_num(x, 0, 0, 0) # Replace NaN/Inf values with 0
        
        bs, in_chans, image_size, _ = x.shape
        
        x = x.view(bs, in_chans, image_size, image_size)
        
        encs = self.enc(x)
        encs = [encs[k] for k in encs]

        #encs[0]: [bs, 128, H/4, W/4]    # 第1阶段特征，下采样4倍
        #encs[1]: [bs, 256, H/8, W/8]    # 第2阶段特征，下采样8倍
        #encs[2]: [bs, 320, H/16, W/16]  # 第3阶段特征，下采样16倍
        #encs[3]: [bs, 512, H/32, W/32]  # 第4阶段特征，下采样32倍

        dec4 = encs[-1] # [bs, 512, H/32, W/32]

        if self.mask_head:
            dec3 = self.dec4(dec4,encs[-2]) # [bs, 384, H/16, W/16]
            dec2 = self.dec3(dec3,encs[-3]) # [bs, 192, H/8, W/8]
            dec1 = self.dec2(dec2,encs[-4]) # [bs, 96, H/4, W/4]
            x = self.fpn([dec4, dec3, dec2], dec1) # [bs, 96+32*3, H/4, W/4]
            x = self.final_conv(self.drop(x))
            if self.up_result != 0: x = F.interpolate(x,scale_factor=self.up_result,mode='bilinear')

        feat = dec4 # [bs, 512, H/32, W/32]
        avg_feat = self.avgpool(feat) # [bs * n_slice_per_c, 512, 1, 1]
        avg_feat = avg_feat.view(bs, -1) # [bs, n_slice_per_c, 512]        
        feat = avg_feat

        # feat, _ = self.lstm2(feat)
        # feat = feat.contiguous().view(bs * n_slice_per_c, -1) # [bs * n_slice_per_c, 1024]

        # print("feat shape before lstm:", feat.shape)
        feat = self.head(feat) # [bs * n_slice_per_c, num_classes]
        feat = feat.view(bs, -1).contiguous()
        feat = torch.nan_to_num(feat, 0, 0, 0)
        
        if self.mask_head:
            return feat, x
        else:
            return feat, None
    
if __name__ == '__main__':
    model = Model(ps=0.1, mask_head=True)
    x = torch.randn(2, 3, 384, 384)
    feat, mask = model(x)
    print(feat.shape)
    if mask is not None:
        print(mask.shape)