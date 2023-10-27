from policy.models import ResidualBlock, conv3x3
from policy.network import FeatureTunk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
import utils.logger as logging
from utils.constants import *
from collections import OrderedDict

class ActionNet(nn.Module):
    def __init__(self, args, is_train=True):
        super(ActionNet, self).__init__()

        self.args = args
        self.is_train = is_train

        self.nr_rotations = 16
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.rb1 = self._make_layer(64, 128)
        self.rb2 = self._make_layer(128, 256)
        self.rb3 = self._make_layer(256, 512)
        self.rb4 = self._make_layer(512, 256)
        self.rb5 = self._make_layer(256, 128)
        self.rb6 = self._make_layer(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)


    def _make_layer(self, in_channels, out_channels, blocks=1, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    def _predict(self, depth):
        x = F.relu(self.conv1(depth))
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.rb1(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.rb6(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final_conv(x)
        return out
    

    def forward(self, depth_heightmap, target_mask=None, specific_rotation=-1, is_volatile=[]):
        # compute rotated feature maps            
        prob_depth = self._predict(depth_heightmap.float())
        # logging.info("prob_depth.shape:", prob_depth.shape)  

        prob_target = self._predict(target_mask.float())
        # logging.info("prob_target.shape:", prob_target.shape)  

        prob = torch.cat((prob_depth, prob_target), dim=1)
        # logging.info("prob.shape:", prob.shape)           #torch.Size([4, 2, 144, 144])

        out_prob = torch.mean(prob, dim=1, keepdim=True)
        # logging.info("mean prob.shape:", prob.shape)           #torch.Size([4, 1, 144, 144])

        if not is_volatile:
            # Image-wide softmax
            output_shape = out_prob.shape
            out_prob = out_prob.view(output_shape[0], -1)
            # logging.info("out_prob.shape:", out_prob.shape)

            out_prob = torch.softmax(out_prob, dim=1)
            out_prob = out_prob.view(output_shape).to(dtype=torch.float)
            # logging.info("out_prob.shape:", out_prob.shape)

        return out_prob