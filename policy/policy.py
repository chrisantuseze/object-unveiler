import os
import pickle
from policy.models_lstm import Regressor, ResFCN
from policy.action_net_linear import ActionNet
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform, io
from PIL import Image

import pybullet as p
from trainer.memory import ReplayBuffer

import utils.utils as utils
import utils.orientation as ori
from utils.constants import *
import env.cameras as cameras
import policy.grasping as grasping
import utils.logger as logging

class Policy:
    def __init__(self, args, params) -> None:
        self.args = args
        self.params = params
        self.rng = np.random.RandomState()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.rotations = params['agent']['fcn']['rotations']
        self.aperture_limits = params['agent']['regressor']['aperture_limits']
        self.pxl_size = params['env']['pixel_size']
        self.bounds = np.array(params['env']['workspace']['bounds'])

        self.crop_size = 32
        self.push_distance = 0.10 #0.08 #0.02 #0.15 # distance of the floating hand from the object to be grasped
        self.z = 0.1 # distance of the floating hand from the table (vertical distance)

        # self.fcn = ResFCN().to(self.device)
        # self.fcn_optimizer = optim.Adam(self.fcn.parameters(), lr=params['agent']['fcn']['learning_rate'])
        # self.fcn_criterion = nn.BCELoss(reduction='None')

        self.fcn = ResFCN(args).to(self.device) #ActionNet(args, is_train=False).to(self.device)
        self.fcn_optimizer = optim.Adam(self.fcn.parameters(), lr=params['agent']['fcn']['learning_rate'])
        self.fcn_criterion = nn.BCELoss(reduction='None')

        self.reg = Regressor().to(self.device)
        self.reg_optimizer = optim.Adam(self.reg.parameters(), lr=params['agent']['regressor']['learning_rate'])
        self.reg_criterion = nn.L1Loss()

        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

        demo_save_dir = 'save/ppg-dataset'
        self.replay_buffer = ReplayBuffer(demo_save_dir)

    def seed(self, seed):
        self.rng.seed(seed)

    def is_state_init_valid(self, obs):
        """
        Checks if the state/scene initialization is a valid
        """
        flat_objs = 0
        for obj in obs['full_state']:
            obj_pos, obj_quat = p.getBasePositionAndOrientation(obj.body_id)
            rot_mat = ori.Quaternion(x=obj_quat[0], y=obj_quat[1],
                                     z=obj_quat[2], w=obj_quat[3]).rotation_matrix()
            
            angle_z = np.arccos(np.dot(np.array([0, 0, 1]), rot_mat[0:3, 2]))

            if np.abs(angle_z) > 0.1:
                flat_objs += 1

        return (flat_objs != len(obs['full_state']))

    def state_representation(self, obs):
        state = utils.get_fused_heightmap(obs, cameras.RealSense.CONFIG, self.bounds, self.pxl_size)
        return state
    
    def preprocess_old(self, state):
        """
        Pre-process heightmap (padding and normalization)
        """
        # Pad heightmap.
        diagonal_length = float(state.shape[0]) * np.sqrt(2)
        diagonal_length = np.ceil(diagonal_length/16) * 16
        self.padding_width = int((diagonal_length - state.shape[0])/2)
        padded_heightmap = np.pad(state, self.padding_width, 'constant', constant_values=-0.01)

        # Normalize heightmap
        image_mean = 0.01
        image_std = 0.03
        padded_heightmap = (padded_heightmap - image_mean)/image_std

        # Add extra channel
        padded_heightmap = np.expand_dims(padded_heightmap, axis=0)

        padded_heightmap = padded_heightmap.astype(np.float32)
        return padded_heightmap
    
    
    def postprocess_old(self, q_maps):
        """
        Remove extra padding
        """

        w = int(q_maps.shape[2] - 2 * self.padding_width)
        h = int(q_maps.shape[3] - 2 * self.padding_width)
        remove_pad = np.zeros((q_maps.shape[0], q_maps.shape[1], w, h))

        for i in range(q_maps.shape[0]):
            for j in range(q_maps.shape[1]):

                # remove extra padding
                q_map = q_maps[i, j, self.padding_width:int(q_maps.shape[2] - self.padding_width),
                               self.padding_width:int(q_maps.shape[3] - self.padding_width)]
                
                remove_pad[i][j] = q_map.detach().cpu().numpy()

        return remove_pad
    
    def postprocess(self, q_maps):
        """
        Remove extra padding
        """

        w = int(q_maps.shape[3] - 2 * self.padding_width)
        h = int(q_maps.shape[4] - 2 * self.padding_width)

        remove_pad = np.zeros((q_maps.shape[0], q_maps.shape[1], q_maps.shape[2], w, h))

        for i in range(q_maps.shape[0]):
            for j in range(q_maps.shape[1]):
                for k in range(q_maps.shape[2]):
                    # remove extra padding
                    q_map = q_maps[i, j, k, self.padding_width:int(q_maps.shape[3] - self.padding_width),
                                self.padding_width:int(q_maps.shape[4] - self.padding_width)]
                    
                    remove_pad[i][j][k] = q_map.detach().cpu().numpy()

        return remove_pad
    
    def preprocess_aperture_image(self, state, p1, theta, plot=False):
        """
        Add extra padding, rotate image so the push always points to the right, crop around
        the initial push (something like attention) and finally normalize the cropped image.
        """

        # add extra padding (to handle rotations inside network)
        diag_length = float(state.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/16) * 16
        padding_width = int((diag_length - state.shape[0])/2)
        depth_heightmap = np.pad(state, padding_width, 'constant')
        padded_shape = depth_heightmap.shape

        p1 += padding_width
        action_theta = -((theta + (2 * np.pi)) % (2 * np.pi))
        
        # rotate image (push always on the right)
        rot = cv2.getRotationMatrix2D((int(padded_shape[0] / 2), int(padded_shape[1] / 2)),
                                      action_theta * 180 / np.pi, 1.0)
        rotated_heightmap = cv2.warpAffine(depth_heightmap, rot, (padded_shape[0], padded_shape[1]))

        # compute the position of p1 on the rotated heightmap
        rotated_pt = np.dot(rot, (p1[0], p1[1], 1.0))
        rotated_pt = (int(rotated_pt[0]), int(rotated_pt[1]))

         # crop heightmap
        cropped_map = np.zeros((2 * self.crop_size, 2 * self.crop_size), dtype=np.float32)
        y_start = max(0, rotated_pt[1] - self.crop_size)
        y_end = min(padded_shape[0], rotated_pt[1] + self.crop_size)
        x_start = rotated_pt[0]
        x_end = min(padded_shape[0], rotated_pt[0] + 2 * self.crop_size)
        cropped_map[0:y_end - y_start, 0:x_end - x_start] = rotated_heightmap[y_start: y_end, x_start: x_end]

        # logging.info( action['opening']['min_width'])
        if plot:
            p2 = np.array([0, 0])
            p2[0] = p1[0] + 20 * np.cos(theta)
            p2[1] = p1[1] - 20 * np.sin(theta)

            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(depth_heightmap)
            ax[0].plot(p1[0], p1[1], 'o', 2)
            ax[0].plot(p2[0], p2[1], 'x', 2)
            ax[0].arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], width=1)

            rotated_p2 = np.array([0, 0])
            rotated_p2[0] = rotated_pt[0] + 20 * np.cos(0)
            rotated_p2[1] = rotated_pt[1] - 20 * np.sin(0)
            ax[1].imshow(rotated_heightmap)
            ax[1].plot(rotated_pt[0], rotated_pt[1], 'o', 2)
            ax[1].plot(rotated_p2[0], rotated_p2[1], 'x', 2)
            ax[1].arrow(rotated_pt[0], rotated_pt[1], rotated_p2[0] - rotated_pt[0], rotated_p2[1] - rotated_pt[1],
                        width=1)

            ax[2].imshow(cropped_map)
            plt.show()

        # normalize maps 
        image_mean = 0.01
        image_std = 0.03
        cropped_map = (cropped_map - image_mean)/image_std
        cropped_map = np.expand_dims(cropped_map, axis=0)

        three_channel_img = np.zeros((3, cropped_map.shape[1], cropped_map.shape[2]))
        three_channel_img[0], three_channel_img[1], three_channel_img[2] = cropped_map, cropped_map, cropped_map
        
        p1 -= padding_width
        
        return three_channel_img
    
    def random_sample(self, state):
        action = np.zeros((4,))
        action[0] = self.rng.randint(0, state.shape[0])
        action[1] = self.rng.randint(0, state.shape[1])
        action[2] = self.rng.randint(0, 16) * 2 * np.pi / self.rotations
        action[3] = self.rng.uniform(self.aperture_limits[0], self.aperture_limits[1])
        return action

    def guided_exploration(self, state, target, sample_limits=[0.1, 0.15]):
        obj_ids = np.argwhere(state > self.z)

        # Sample initial position.
        valid_pxl_map = np.zeros(state.shape)
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                dists = np.linalg.norm(np.array([y, x]) - obj_ids, axis=1)
                if sample_limits[0] / self.pxl_size < np.min(dists) < sample_limits[1] / self.pxl_size:
                    valid_pxl_map[y, x] = 255

        valid_pxl_map = utils.resize_mask(transform, target)

        valid_pxls = np.argwhere(valid_pxl_map == 1)
        valid_ids = np.arange(0, valid_pxls.shape[0])

        objects_mask = np.zeros(state.shape)
        objects_mask[state > self.z] = 255

        # print(target.shape, state.shape)
        # np.savetxt('target.txt', target)

        # fig, ax = plt.subplots(1, 4)
        # ax[0].imshow(state)
        # ax[1].imshow(valid_pxl_map)
        # ax[2].imshow(objects_mask)
        # ax[3].imshow(target)

        # plt.show()


        _, thresh = cv2.threshold(objects_mask.astype(np.uint8), 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        while True:
            pxl = valid_pxls[self.rng.choice(valid_ids, 1)[0]]
            p1 = np.array([pxl[1], pxl[0]])

            # Sample pushing direction. Push directions point always towards the objects.
            # Keep only contour points that are around the sample pixel position.
            pushing_area = self.push_distance / self.pxl_size
            points = []
            for cnt in contours:
                for pnt in cnt:
                    if (p1[0] - pushing_area < pnt[0, 0] < p1[0] + pushing_area) and \
                       (p1[1] - pushing_area < pnt[0, 1] < p1[1] + pushing_area):
                        points.append(pnt[0])
            if len(points) > 0:
                break

        ids = np.arange(len(points))
        random_id = self.rng.choice(ids, 1)[0]
        p2 = points[random_id]
        push_dir = p2 - p1
        theta = -np.arctan2(push_dir[1], push_dir[0])
        step_angle = 2 * np.pi / self.rotations
        discrete_theta = round(theta / step_angle) * step_angle

        # Sample aperture uniformly
        aperture = self.rng.uniform(self.aperture_limits[0], self.aperture_limits[1])

        action = np.zeros((4,))
        action[0] = p1[0]
        action[1] = p1[1]
        action[2] = discrete_theta
        action[3] = aperture

        # if not grasping.is_grasped_object(target, action):
        #     self.guided_exploration(state, target)

        return action

    def guided_exploration_old(self, state, target_mask, sample_limits=[0.1, 0.15]):
        obj_ids = np.argwhere(state > self.z)

        # sample initial position
        # valid_pxl_map = np.zeros(state.shape)
        # for x in range(state.shape[0]):
        #     for y in range(state.shape[1]):
        #         dists = np.linalg.norm(np.array([y, x]) - obj_ids, axis=1) # gets the distances of the pixels (objs) to the vertical pos of the hand

        #         if sample_limits[0]/self.pxl_size < np.min(dists) < sample_limits[1]/self.pxl_size: # pixel/obj with shortest distance gets picked
        #             valid_pxl_map[y, x] = 255


        valid_pxl_map = utils.resize_mask(transform, target_mask)

        valid_pxl_map = np.array(valid_pxl_map, dtype=np.uint8)

        # Finds the indices of yellow pixels (where the value is 1)
        valid_pxls = np.argwhere(valid_pxl_map == 1) # gets indices of the pixels with values equal to 255
        valid_ids = np.arange(0, valid_pxls.shape[0]) # creates an array containing values from 0 - valid_pxls.shape[0]

        objects_mask = np.zeros(state.shape)
        objects_mask[state > self.z] = 255
        _, thresh = cv2.threshold(objects_mask.astype(np.uint8), 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        while True:
            pxl = valid_pxls[self.rng.choice(valid_ids, 1)[0]]
            p1 = np.array([pxl[1], pxl[0]]) # p1 is the initial distance of the palm from the object

            # Sample pushing direction. Push directions point always towards the objects.
            # Keep only contour points that are around the sample pixel position.
            pushing_area = self.push_distance / self.pxl_size
            points = []
            for cnt in contours:
                for pnt in cnt:
                    if (p1[0] - pushing_area < pnt[0, 0] < p1[0] + pushing_area) and \
                       (p1[1] - pushing_area < pnt[0, 1] < p1[1] + pushing_area):
                        points.append(pnt[0])

            if len(points) > 0:
                break

        ids = np.arange(len(points))
        random_id = self.rng.choice(ids, 1)[0] # creates an array by randomizing items in the ids array and picking one item from it.
        p2 = points[random_id]
        push_dir = p2 - p1
        theta = -np.arctan2(push_dir[1], push_dir[0])
        step_angle = 2 * np.pi / self.rotations
        discrete_theta = round(theta / step_angle) * step_angle

        # sample aperture uniformly
        aperture = self.rng.uniform(self.aperture_limits[0], self.aperture_limits[1])

        action = np.zeros((4,))
        action[0] = p1[0] * 1.05
        action[1] = p1[1] * 1.05
        action[2] = discrete_theta
        action[3] = aperture

        # logging.info("action:", action)

        return action
    
    def exploit(self, state, target_mask):

        # data_transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize to the input size expected by ResNet (can be adjusted)
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     transforms.Normalize(mean=(0.449), std=(0.226))
        # ])
        
        # # find optimal position and orientation
        # heightmap, self.padding_width = utils.preprocess_data(state)
        # # x = torch.FloatTensor(heightmap).unsqueeze(0).to(self.device)

        # x = data_transform(heightmap).to(self.device)
        # x = x.view(1, 1, IMAGE_SIZE, IMAGE_SIZE)

        # # Resize the image using seam carving to match with the heightmap
        # resized_target = utils.resize_mask(transform, target_mask)
        # target, self.padding_width = utils.preprocess_data(resized_target) # this is might not be necessary

        # # target = torch.FloatTensor(target).unsqueeze(0).to(self.device)
        # target = data_transform(target).to(self.device)
        # target = target.view(1, 1, IMAGE_SIZE, IMAGE_SIZE)

        # # combine the two features into a list
        # sequence = [(x, target, target)]
        # # sequence = [(x, target)]
        
        # out_prob = self.fcn(sequence, is_volatile=True)
        # logging.info("out_prob.shape:", out_prob.shape)

        # find optimal position and orientation
        heightmap = self.preprocess_old(state)

        resized_target = utils.resize_mask(transform, target_mask)
        target = self.preprocess_old(resized_target)
        target = torch.FloatTensor(target).unsqueeze(0).to(self.device)

        x = torch.FloatTensor(heightmap).unsqueeze(0).to(self.device)

        out_prob = self.fcn(x, target, is_volatile=True)

        out_prob = self.postprocess(out_prob)

        best_actions = []
        actions = []
        for i in range(out_prob.shape[0]):
            prob = out_prob[i]
            best_action = np.unravel_index(np.argmax(prob), prob.shape)
            best_actions.append(best_action)


            p1 = np.array([best_actions[i][3], best_actions[i][2]])
            theta = best_actions[i][0] * 2 * np.pi/self.rotations

            p2 = np.array([0, 0])
            p2[0] = p1[0] + 20 * np.cos(theta)
            p2[1] = p1[1] - 20 * np.sin(theta)

            # find optimal aperture
            aperture_img = self.preprocess_aperture_image(state, p1, theta)
            x = torch.FloatTensor(aperture_img).unsqueeze(0).to(self.device)
            aperture = self.reg(x).detach().cpu().numpy()[0, 0]
        
            # undo normalization
            aperture = utils.min_max_scale(aperture, range=[0, 1], 
                                        target_range=[self.aperture_limits[0], 
                                                        self.aperture_limits[1]])

            action = np.zeros((4,))
            action[0] = p1[0]
            action[1] = p1[1]
            action[2] = theta
            action[3] = aperture

            print("action:", action)

            actions.append(action)

        return actions
    
    def exploit_old(self, state, target_mask):

        # find optimal position and orientation
        heightmap = self.preprocess_old(state)

        resized_target = utils.resize_mask(transform, target_mask)
        target = self.preprocess_old(resized_target)
        target = torch.FloatTensor(target).unsqueeze(0).to(self.device)

        x = torch.FloatTensor(heightmap).unsqueeze(0).to(self.device)

        out_prob = self.fcn(x, target, is_volatile=True)
        out_prob = self.postprocess_old(out_prob)

        best_action = np.unravel_index(np.argmax(out_prob), out_prob.shape)
        p1 = np.array([best_action[3], best_action[2]])
        theta = best_action[0] * 2 * np.pi/self.rotations

        p2 = np.array([0, 0])
        p2[0] = p1[0] + 20 * np.cos(theta)
        p2[1] = p1[1] - 20 * np.sin(theta)

        # find optimal aperture
        aperture_img = self.preprocess_aperture_image(state, p1, theta)
        x = torch.FloatTensor(aperture_img).unsqueeze(0).to(self.device)
        aperture = self.reg(x).detach().cpu().numpy()[0, 0]
       
        # undo normalization
        aperture = utils.min_max_scale(aperture, range=[0, 1], 
                                       target_range=[self.aperture_limits[0], 
                                                     self.aperture_limits[1]])

        action = np.zeros((4,))
        action[0] = p1[0]
        action[1] = p1[1]
        action[2] = theta
        action[3] = aperture

        return action
    
    def explore(self, state, target_mask):
        explore_prob = max(0.8 * np.power(0.9998, self.learn_step_counter), 0.1)

        if self.rng.rand() < explore_prob:
            action = self.guided_exploration(state, target_mask)
        else:
            action = self.exploit(state, target_mask)
        logging.info('explore action:', action)
        return action
    
    def get_fcn_labels(self, state, action):
        heightmap = self.preprocess(state)
        angle = (action[2] + (2 * np.pi)) % (2 * np.pi)
        rot_id = round(angle/(2 * np.pi/16))

        action_area = np.zeros((state.shape[0], state.shape[1]))
        action_area[int(action[1]), int(action[0])] = 1.0
        label = np.zeros((1, heightmap.shape[1], heightmap.shape[2]))
        label[0, self.padding_width:heightmap.shape[1] - self.padding_width,
              self.padding_width:heightmap.shape[2] - self.padding_width] = action_area

        return heightmap, rot_id, label
    
    # This is redundant
    def learn(self, transition): 
        """
        This is not useful since we use the heightmap_dataset and aperture_dataset to present our collected demonstrations in a form that can be
        fed into a pytorch dataloader which is used in training the modules.
        """
        
        # check if grasp is stable and successful
        if not transition['label']: 
            return
        
        self.replay_buffer.store(transition) # I am not sure this is important

        # sample from replay buffer
        state, action = self.replay_buffer.sample()

        # update resFCN
        heightmap, rot_id, label = self.get_fcn_labels(state, action)
       
        x = torch.FloatTensor(heightmap).unsqueeze(0).to(self.device)
        label = torch.FloatTensor(label).unsqueeze(0).to(self.device)
        rotations = np.array([rot_id])
        q_maps = self.fcn(x, specific_rotation=rotations)

        # compute loss in the whole scene
        loss = self.fcn_criterion(q_maps, label)
        loss = torch.sum(loss)
        logging.info('fcn_loss:', loss.detach().cpu().numpy())
        self.info['fcn_loss'].append(loss.detach().cpu().numpy())

        self.fcn_optimizer.zero_grad()
        loss.backward()
        self.fcn_optimizer.step()

        # update regression network
        aperture_img = self.preprocess_aperture_image(state, p1=np.array([action[0], action[1]]))
        x = torch.FloatTensor(aperture_img, theta=action[2]).unsqueeze(0).to(self.device)
        
        # normalize aperture to range 0-1
        normalized_aperture = utils.min_max_scale(action[3],
                                                  range=[self.aperture_limits[0], self.aperture_limits[1]],
                                                  target_range=[0, 1])
        gt_aperture = torch.FloatTensor(np.array([normalized_aperture])).unsqueeze(0).to(self.device)
        pred_aperture = self.reg(x)

        logging.info('APERTURES')
        logging.info(gt_aperture, pred_aperture)

        # compute loss
        loss = self.reg_criterion(pred_aperture, gt_aperture)
        logging.info('reg_loss:', loss.detach().cpu().numpy())
        self.info['reg_loss'].append(loss.detach().cpu().numpy())

        self.reg_optimizer.zero_grad()
        loss.backward()
        self.reg_optimizer.step()

        self.learn_step_counter += 1

    def action3d(self, action):
        """
        convert from pixels to 3d coordinates
        """

        x = -(self.pxl_size * action[0] - self.bounds[0][1])
        y = self.pxl_size * action[1] - self.bounds[1][1]
        quat = ori.Quaternion.from_rotation_matrix(np.matmul(ori.rot_y(-np.pi/2),
                                                             ori.rot_x(action[2])))
        return {'pos': np.array([x, y, self.z]),
                'quat': quat,
                'aperture': action[3],
                'push_distance': self.push_distance}
    
    def save(self, epoch):
        folder_name = os.path.join(self.params['log_dir'], 'model_' + str(epoch))
        os.mkdir(folder_name)
        torch.save(self.fcn.state_dict(), os.path.join(folder_name, 'fcn.pt'))
        torch.save(self.reg.state_dict(), os.path.join(folder_name, 'reg.pt'))

        log_data = {'params': self.params.copy()}
        pickle.dump(log_data, open(os.path.join(folder_name, 'log_data'), 'wb'))

        self.info['learn_step_counter'] = self.learn_step_counter
        pickle.dump(self.info, open(os.path.join(folder_name, 'info'), 'wb'))

    def load(self, fcn_model, reg_model):
        self.fcn.load_state_dict(torch.load(fcn_model, map_location=self.device))
        self.fcn.eval()

        self.reg.load_state_dict(torch.load(reg_model, map_location=self.device))
        self.reg.eval()

    def is_terminal(self, next_obs: ori.Quaternion):
        # check if there is only one object left in the scene TODO This won't be used for mine

        objects_above = 0
        for obj in next_obs['full_state']:
            if obj.pos[2] > 0:

                # check if there is at least one object in the scene with the axis parallel to world z
                rot_mat = obj.quat.rotation_matrix()
                angle_z = np.arccos(np.dot(np.array([0, 0, 1]), rot_mat[0:3, 2]))

                if np.abs(angle_z) < 0.5:
                    objects_above += 1
                else:
                    return True
                
        return objects_above <= 1


def plot_maps(state, out_prob):

    glob_max_prob = np.max(out_prob)
    fig, ax = plt.subplots(4, 4)
    for i in range(16):
        x = int(i / 4)
        y = i % 4

        min_prob = np.min(out_prob[i][0])
        max_prob = np.max(out_prob[i][0])

        prediction_vis = utils.min_max_scale(out_prob[i][0],
                                             range=(min_prob, max_prob),
                                             target_range=(0, 1))
        best_pt = np.unravel_index(prediction_vis.argmax(), prediction_vis.shape)
        maximum_prob = np.max(out_prob[i][0])
        #
        ax[x, y].imshow(state, cmap='gray')
        ax[x, y].imshow(prediction_vis, alpha=0.5)
        ax[x, y].set_title(str(i) + ', ' + str(format(maximum_prob, ".3f")))

        if glob_max_prob == max_prob:
            ax[x, y].plot(best_pt[1], best_pt[0], 'rx')
        else:
            ax[x, y].plot(best_pt[1], best_pt[0], 'ro')
        dx = 20 * np.cos((i / 16) * 2 * np.pi)
        dy = -20 * np.sin((i / 16) * 2 * np.pi)
        ax[x, y].arrow(best_pt[1], best_pt[0], dx, dy, width=2, color='g')
        plt.imshow(state, cmap='gray')
        plt.imshow(prediction_vis, alpha=0.5)
        # plt.savefig(os.path.join(self.params['log_dir'], 'map_' + str(i) + '.png'), dpi=720)

    plt.show()
    # plt.savefig(os.path.join(self.params['log_dir'], 'maps', 'map_' + str(self.learn_step_counter) + '.png'),
    #             dpi=720)
    # plt.close()