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
        hidden_size = IMAGE_SIZE
        output_dim1 = IMAGE_SIZE * IMAGE_SIZE 

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
    
    def _volatile(self, depth_heightmap, target_mask, obstacle_mask):
        batch_rot_depth = torch.zeros((self.nr_rotations, 1, depth_heightmap.shape[3], depth_heightmap.shape[3])).to(self.device)
        batch_rot_target = torch.zeros((self.nr_rotations, 1, target_mask.shape[3], target_mask.shape[3])).to(self.device)
        batch_rot_obstacle = torch.zeros((self.nr_rotations, 1, obstacle_mask.shape[3], obstacle_mask.shape[3])).to(self.device)
        
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
            
            rotate_target_mask = F.grid_sample(Variable(target_mask, requires_grad=False).to(self.device),
                flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
            
            rotate_obstacle_mask = F.grid_sample(Variable(obstacle_mask, requires_grad=False).to(self.device),
                flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

            batch_rot_depth[rot_id] = rotate_depth[0]
            batch_rot_target[rot_id] = rotate_target_mask[0]
            batch_rot_obstacle[rot_id] = rotate_obstacle_mask[0]

        # return batch_rot_depth, batch_rot_target, batch_rot_obstacle

        # compute rotated feature maps
        prob_depth = self._predict(batch_rot_depth)
        prob_target = self._predict(batch_rot_target)
        prob_obstacle = self._predict(batch_rot_obstacle)

        # logging.info("prob_depth.shape:", prob_depth.shape)
        # logging.info("prob_target.shape:", prob_target.shape)
        # logging.info("prob_obstacle.shape:", prob_obstacle.shape)

        probs = torch.cat((prob_depth, prob_target, prob_obstacle), dim=1)

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

        # batch_rot_target = torch.mean(batch_rot_target, dim=0, keepdim=True)

        # batch_rot_obstacle = torch.mean(batch_rot_obstacle, dim=0, keepdim=True)

        # logging.info("mean out_prob.shape:", out_prob.shape)
        # logging.info("batch_rot_target.shape:", batch_rot_target.shape)
        # logging.info("batch_rot_obstacle.shape:", batch_rot_obstacle.shape)

        return out_prob
    
    def _non_volatile(self, depth_heightmap, target_mask, obstacle_mask, specific_rotation=-1):
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
        
        rotate_target_mask = F.grid_sample(Variable(target_mask, requires_grad=False).to(self.device),
                                        flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")
        
        rotate_obstacle_mask = F.grid_sample(Variable(obstacle_mask, requires_grad=False).to(self.device),
                                        flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

        # Compute intermediate features
        # prob_depth = self.predict(rotate_depth)
        # prob_target = self.predict(rotate_target_mask)

        # prob = torch.cat((prob_depth, prob_target), dim=1)

        # return rotate_depth, rotate_target_mask, rotate_obstacle_mask
    
        # compute rotated feature maps
        prob_depth = self._predict(rotate_depth)
        prob_target = self._predict(rotate_target_mask)
        prob_obstacle = self._predict(rotate_obstacle_mask)

        probs = torch.cat((prob_depth, prob_target, prob_obstacle), dim=1)

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
    

    def forward(self, sequence, actions=None, rot_ids=[], is_volatile=False):
        # Process the image sequence through the CNN and a fully connected layer
        image_features = []

        # logging.info("heightmap.shape:", sequence[0][0].shape)
        for i in range(len(sequence)):
            heightmap, target_mask, obstacle_mask = sequence[i]
            heightmap = heightmap.to(self.device)
            target_mask = target_mask.to(self.device)
            obstacle_mask = obstacle_mask.to(self.device)

            heightmap_features_t = self._predict(heightmap)
            target_features_t = self._predict(target_mask)
            obstacle_features_t = self._predict(obstacle_mask)

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

            output = output.view(self.args.batch_size, 1, decoder_output.shape[2], decoder_output.shape[2])
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