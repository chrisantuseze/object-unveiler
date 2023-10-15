from policy.models import ResidualBlock, conv3x3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
import utils.logger as logging
from utils.constants import *

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

        # Define training parameters
        input_size = IMAGE_SIZE * IMAGE_SIZE * 3 
        hidden_size = 64
        output_dim1 = IMAGE_SIZE * IMAGE_SIZE 

        # self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        self.encoder_fc = nn.Linear(input_size, output_dim1)

         # LSTM for sequence processing
        self.encoder = nn.LSTM(input_size=output_dim1, hidden_size=hidden_size) # out, hid, cell = output_dim1, _, _
        self.decoder = nn.LSTM(input_size=output_dim1, hidden_size=hidden_size) # decoder(in, hid, cell) = output_dim1, output_dim1, output_dim1)

        self.out = nn.Linear(hidden_size, output_dim1)

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
    
    def forward(self, sequence, actions=None, rot_ids=[], is_volatile=False):
        # Process the image sequence through the CNN and a fully connected layer
        image_features = []

        # logging.info("heightmap.shape:", sequence[0][0].shape)
        for i in range(len(sequence)):
            heightmap, target_mask, obstacle_mask = sequence[i]
            heightmap = heightmap.to(self.device)
            target_mask = target_mask.to(self.device)
            obstacle_mask = obstacle_mask.to(self.device)

            heightmap_features_t = self._predict(heightmap) #self.feature_extractor(heightmap) #self._predict(heightmap)
            target_features_t = self._predict(target_mask) #self.feature_extractor(target_mask) #self._predict(target_mask)
            obstacle_features_t = self._predict(obstacle_mask) #self.feature_extractor(obstacle_mask) #self._predict(obstacle_mask)

            heightmap_features_t = heightmap_features_t.view(self.args.batch_size, -1)
            # logging.info("view heightmap_features_t.shape:", heightmap_features_t.shape)

            target_features_t = target_features_t.view(self.args.batch_size, -1)
            obstacle_features_t = obstacle_features_t.view(self.args.batch_size, -1)

            concat_image_features = torch.cat((heightmap_features_t, target_features_t, obstacle_features_t), dim=1).to(self.device)
            # logging.info("concat_image_features.shape:", concat_image_features.shape)

            features = self.encoder_fc(concat_image_features)
            # logging.info("features.shape:", features.shape)

            image_features.append(features)
            
        input_seq = torch.stack(image_features, dim=0)
        # logging.info("input_seq.shape:", input_seq.shape)

        input_seq = input_seq.view(self.args.sequence_length, self.args.batch_size, -1)
        # logging.info("view input_seq.shape:", input_seq.shape)

        output, (hidden, cell) = self.encoder(input_seq)
        # logging.info("encoder output.shape:", output.shape, "hidden.shape:", hidden.shape)

        outputs = torch.zeros(self.args.sequence_length, self.args.batch_size, 1, IMAGE_SIZE, IMAGE_SIZE).to(self.device)
        decoder_input = torch.empty(1, self.args.batch_size, input_seq.shape[2], dtype=torch.float, device=self.args.device)

        for i in range(len(actions)):
            # print(actions[i].shape)
            decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
            # logging.info("decoder_output.shape:", decoder_output.shape)

            output = self.out(decoder_output)
            # logging.info("out output.shape:", output.shape)

            output = output.view(self.args.batch_size, 1, outputs.shape[3], outputs.shape[4])
            # logging.info("view output.shape:", output.shape)
            
            outputs[i] = output

            if actions is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = actions[i].view(1, self.args.batch_size, -1)
                # decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        logging.info("outputs.shape:", outputs.shape)

        return outputs