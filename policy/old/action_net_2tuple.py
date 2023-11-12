from policy.models import ResidualBlock, conv3x3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
import utilities.logger as logging
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
        input_size = IMAGE_SIZE * IMAGE_SIZE * 2
        hidden_size = IMAGE_SIZE  # LSTM hidden state size
        num_layers = 2  # Number of LSTM layers
        bidirectional = False  # Use bidirectional LSTM
        output_dim1 = IMAGE_SIZE * IMAGE_SIZE
        output_dim2 = IMAGE_SIZE * IMAGE_SIZE * self.nr_rotations

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc_train = nn.Linear(hidden_size, output_dim1)
        self.fc_eval = nn.Linear(hidden_size, output_dim2)

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
    
    def _volatile(self, depth_heightmap, object_mask):
        batch_rot_depth = torch.zeros((self.nr_rotations, 1, depth_heightmap.shape[3], depth_heightmap.shape[3])).to(self.device)
        batch_rot_object = torch.zeros((self.nr_rotations, 1, object_mask.shape[3], object_mask.shape[3])).to(self.device)
        
        for rot_id in range(self.nr_rotations):
            # Compute sample grid for rotation before neural network
            theta = np.radians(rot_id * (360 / self.nr_rotations))
            affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0], [-np.sin(theta), np.cos(theta), 0.0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()

            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).to(self.device), depth_heightmap.size(), align_corners=True)
            
            # Rotate images clockwise
            rotate_depth = F.grid_sample(Variable(depth_heightmap, requires_grad=False).to(self.device),
                flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
            
            rotate_object_mask = F.grid_sample(Variable(object_mask, requires_grad=False).to(self.device),
                flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
            
            batch_rot_depth[rot_id] = rotate_depth[0]
            batch_rot_object[rot_id] = rotate_object_mask[0]

        # compute rotated feature maps
        prob_depth = self._predict(batch_rot_depth)
        prob_object = self._predict(batch_rot_object)

        probs = torch.cat((prob_depth, prob_object), dim=1)

        return self._volatile_undo_rotate(probs)

        
    def _volatile_undo_rotate(self, prob):
        # undo rotation
        affine_after = torch.zeros((self.nr_rotations, 2, 3))
        for rot_id in range(self.nr_rotations):
            # compute sample grid for rotation before neural network
            theta = np.radians(rot_id * (360 / self.nr_rotations))
            affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0], [-np.sin(-theta), np.cos(-theta), 0.0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            affine_after[rot_id] = affine_mat_after

        flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device), prob.size(), align_corners=True)
        out_prob = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True)

        logging.info("out_prob.shape:", out_prob.shape)
        out_prob = torch.mean(out_prob, dim=0, keepdim=True)

        return out_prob
    

    def _non_volatile(self, depth_heightmap, object_mask, specific_rotation=-1):
        thetas = np.radians(specific_rotation * (360 / self.nr_rotations))
        affine_before = torch.zeros((depth_heightmap.shape[0], 2, 3))
        for i in range(len(thetas)):
            # Compute sample grid for rotation before neural network
            theta = thetas[i]
            affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                            [-np.sin(theta), np.cos(theta), 0.0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            affine_before[i] = affine_mat_before

        flow_grid_before = F.affine_grid(Variable(affine_before, requires_grad=False).to(self.device), depth_heightmap.size(), align_corners=True)

        # Rotate image clockwise_
        rotate_depth = F.grid_sample(Variable(depth_heightmap, requires_grad=False).to(self.device),
                                        flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
        
        rotate_object_mask = F.grid_sample(Variable(object_mask, requires_grad=False).to(self.device),
                                        flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
        
        # compute rotated feature maps
        prob_depth = self._predict(rotate_depth)
        prob_object = self._predict(rotate_object_mask)

        probs = torch.cat((prob_depth, prob_object), dim=1)

        return self._non_volatile_undo_rotate(depth_heightmap, probs, thetas)

        
    def _non_volatile_undo_rotate(self, depth_heightmap, prob, thetas):
        # Compute sample grid for rotation after branches
        affine_after = torch.zeros((depth_heightmap.shape[0], 2, 3))
        for i in range(len(thetas)):
            theta = thetas[i]
            affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0], [-np.sin(-theta), np.cos(-theta), 0.0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            affine_after[i] = affine_mat_after

        flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device), prob.size(), align_corners=True)

        # Forward pass through branches, undo rotation on output predictions, upsample results
        out_prob = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True)

        # Image-wide softmax
        output_shape = out_prob.shape
        out_prob = out_prob.view(output_shape[0], -1)
        out_prob = torch.softmax(out_prob, dim=1)
        out_prob = out_prob.view(output_shape).to(dtype=torch.float)

        return out_prob
    

    def forward(self, sequence, rot_ids=[], is_volatile=False):
        probs = []
        for i in range(len(sequence)):
            heightmap, object_mask = sequence[i]
            
            heightmap = heightmap.to(self.device)
            object_mask = object_mask.to(self.device)

            if is_volatile:
                prob = self._volatile(heightmap, object_mask)
                probs.append(prob)
            else:
                rot_id = rot_ids[i]
                prob = self._non_volatile(heightmap, object_mask, rot_id)
                probs.append(prob)

            # logging.info("prob.shape:", prob.shape)                     #torch.Size([1, 2, 224, 224])

        probs_stack = torch.stack(probs, dim=0)
        # logging.info("probs_stack.shape:", probs_stack.shape)           #torch.Size([1, 1, 2, 224, 224])

        sequence_length, batch_size, channels, height, width = probs_stack.shape

        probs_stack = probs_stack.view(-1, channels, height, width)
        # logging.info("view probs_stack.shape:", probs_stack.shape)      #torch.Size([1, 2, 224, 224])

        # Pad the tensor to achieve the desired shape
        if not self.is_train:
            sequence_length, channels, height, width = probs_stack.shape
            pad_sequence_length = max(0, self.args.sequence_length - sequence_length)
            pad = (0,0, 0,0, 0,0, 0,pad_sequence_length) # it starts from the back of the dimension i.e 224, 224, 3, 1
            probs_stack = torch.nn.functional.pad(probs_stack, pad, mode='constant', value=0)
            logging.info("padded probs_stack.shape:", probs_stack.shape)    #torch.Size([4, 2, 224, 224])


        embeddings = probs_stack.view(batch_size, self.args.sequence_length, -1)
        # logging.info("view embeddings.shape:", embeddings.shape)        #torch.Size([1, 4, 100352])

        outputs, (hidden, cell) = self.lstm(embeddings)
        # logging.info("lstm outputs.shape:", outputs.shape)              #torch.Size([1, 4, 224])

        if self.is_train:
            predictions = self.fc_train(outputs)
            predictions = predictions.view(self.args.sequence_length, batch_size * 1, 1, IMAGE_SIZE, IMAGE_SIZE) #torch.Size([4, 1, 1, 224, 224])
        else:
            predictions = self.fc_eval(outputs)
            predictions = predictions.view(self.args.sequence_length, batch_size * self.nr_rotations, 1, IMAGE_SIZE, IMAGE_SIZE)

        # logging.info("view predictions.shape:", predictions.shape)      
        
        return predictions
    