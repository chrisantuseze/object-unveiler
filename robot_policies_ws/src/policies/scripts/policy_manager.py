#!/usr/bin/env python3

#ros
from policy import grasping
import rospy
import cv2
import numpy as np
import os
import torch
import yaml
import argparse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from policies.msg import ObservData, ActionData
from mask_rg.object_segmenter import ObjectSegmenter
from policy.policy import Policy

from std_msgs.msg import String
from utils import general_utils

class PolicyManager:
    def __init__(self, args):
        # Initialize the ROS node
        rospy.init_node('policy_manager')

        self.args = args
        print("Running eval...")
        with open('yaml/bhand.yml', 'r') as stream:
            params = yaml.safe_load(stream)

        self.policy = Policy(args, params)
        self.policy.load(ae_model=args.ae_model, reg_model=args.reg_model, sre_model=args.sre_model)

        self.segmenter = ObjectSegmenter(is_real=True)
        self.rng = np.random.RandomState()
        self.rng.seed(args.seed)

        self.TEST_DIR = "robot_policies_ws/src/policies/scripts/images"
        if not os.path.exists(self.TEST_DIR):
            os.makedirs(self.TEST_DIR)

        self.bridge = CvBridge()
        self.rate = rospy.Rate(1)  # 1 Hz

        self.observation_sub = rospy.Subscriber("/action/obs", ObservData, self.process_observation)
        self.action_pub = rospy.Publisher('/action/data', ActionData, queue_size=1)
        while self.action_pub.get_num_connections() == 0:
            rospy.loginfo("Waiting for robot to subscribe...")
            rospy.sleep(0.5)

        # rospy.Subscriber('/robot_machine', String, self.callback)
        # self.pub = rospy.Publisher('/machine_robot', String, queue_size=10)
        # while self.pub.get_num_connections() == 0:
        #     rospy.loginfo("Waiting for robot to subscribe...")
        #     rospy.sleep(0.5)

        # rate = rospy.Rate(1)  # 1Hz
        # while not rospy.is_shutdown():
        #     self.pub.publish(String(data="Test message from machine"))
        #     print("Published message")
        #     rate.sleep()

        print("Initialized everything")
    
    def process_observation(self, obs_data):
        print("Observation subscriber callback triggered.")

        color_image = self.bridge.imgmsg_to_cv2(obs_data.color_image, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(obs_data.depth_image, desired_encoding="16UC1")

        target_mask = obs_data.target_mask
        if target_mask is not None:
            target_mask = self.bridge.imgmsg_to_cv2(target_mask, desired_encoding='mono8')

        # Convert from BGR (ROS standard) to RGB if needed
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
        # depth_image = cv2.flip(depth_image, -1)

        if target_mask is not None:
            target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(self.TEST_DIR, "color_image_data.png"), color_image)
        cv2.imwrite(os.path.join(self.TEST_DIR, "depth_image_data.png"), depth_image)

        processed_masks, pred_mask, raw_masks, bboxes = self.segmenter.from_maskrcnn(color_image, dir=self.TEST_DIR, bbox=True, dim=(480, 640))

         # get a randomly picked target mask from the segmented image
        if target_mask is None:
            target_mask, target_id = general_utils.get_target_mask(processed_masks, color_image, self.rng)
            print("Target ID:", target_id)
        cv2.imwrite(os.path.join("dofbot_pro_ws/src/dofbot_pro_info/scripts", "initial_target_mask.png"), target_mask)
        cv2.imwrite(os.path.join("dofbot_pro_ws/src/dofbot_pro_info/scripts", "initial_scene.png"), pred_mask)

        print("len(processed_masks):", len(processed_masks))

        state = self.policy.get_dmap(color_image, depth_image, intrinsics=None)
        print("Gotten the state")

        target_id, target_mask = grasping.find_target(processed_masks, target_mask)
        if target_id == -1:
            print("No target mask found.")
            action = np.array([0, 0, 0, 0])
        else:
            print("Getting actions...")
            # action = policy.exploit_unveiler(state, obs['color'], target_mask, processed_masks, bboxes)
            action = self.policy.exploit_real_robot(state, target_mask)
            print("Gotten the action:", action)

        action_data = ActionData()
        action_data.values = action.tolist()
        action_data.target_mask = self.bridge.cv2_to_imgmsg(target_mask, encoding="mono8")
        self.action_pub.publish(action_data)

        print("Action data published.", action_data.values)

    def callback(self, msg):
        print("Received message from Robot:", msg.data)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', default='ae', type=str, help='')
    
    # args for eval_agent
    parser.add_argument('--ae_model', default='save/ae/ae_model_best.pt', type=str, help='')
    parser.add_argument('--sre_model', default='save/sre/sre_model_best.pt', type=str, help='')
    parser.add_argument('--reg_model', default='downloads/reg_model.pt', type=str, help='')
    parser.add_argument('--seed', default=16, type=int, help='')
    parser.add_argument('--n_scenes', default=100, type=int, help='')
    parser.add_argument('--object_set', default='seen', type=str, help='')

    # args for trainer
    parser.add_argument('--dataset_dir', default='save/pc-ou-dataset', type=str, help='')
    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--lr', default=0.0001, type=float, help='')
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--split_ratio', default=0.9, type=float, help='')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')

    parser.add_argument('--sequence_length', default=1, type=int, help='')
    parser.add_argument('--patch_size', default=64, type=int, help='')
    parser.add_argument('--num_patches', default=10, type=int, help='This should not be less than the maximum possible number of objects in the scene, which from list Environment.nr_objects is 9')
    parser.add_argument('--step', default=500, type=int, help='')

    # args for act
    parser.add_argument('--chunk_size', default=3, action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"You are using {args.device}")

    try:
        policy_manager = PolicyManager(args)
        rospy.spin()
    except Exception as e:
        rospy.logerr(str(e))

# if __name__ == '__main__':
#     rospy.init_node('test_subscriber')

#     print("ROS_MASTER_URI:", rospy.get_master())
#     print("Node name:", rospy.get_name())
#     print("Node URI:", rospy.get_node_uri())

#     def callback(msg):
#         print("Received message from Robot:", msg.data)
    
#     sub = rospy.Subscriber('/robot_machine', String, callback)
#     pub = rospy.Publisher('/machine_robot', String, queue_size=10)
#     while pub.get_num_connections() == 0:
#         rospy.loginfo("Waiting for robot to subscribe...")
#         rospy.sleep(0.5)

#     rate = rospy.Rate(1)  # 1Hz
#     while not rospy.is_shutdown():
#         pub.publish(String(data="Test message from machine"))
#         print("Published message")
#         rate.sleep()

#     rospy.spin()