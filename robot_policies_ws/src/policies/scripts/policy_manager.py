#!/usr/bin/env python3

#ros
from dofbot_pro_ws.src.dofbot_pro_info.scripts.robot_operations import convert_numpy_masks_to_ros_image_list
import rospy
import cv2
import numpy as np
import os
import torch
import yaml
import argparse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from policies.msg import SegmentationData, ObservationData, ActionData
from mask_rg.object_segmenter import ObjectSegmenter
from policy.policy import Policy

from std_msgs.msg import String

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

        self.TEST_DIR = "robot_policies_ws/src/policies/scripts/images"
        if not os.path.exists(self.TEST_DIR):
            os.makedirs(self.TEST_DIR)

        self.bridge = CvBridge()
        self.rate = rospy.Rate(1)  # 1 Hz

        self.color_image = None
        self.depth_image = None

        # self.segmenter_pub = rospy.Publisher('/segmentation/data', SegmentationData, queue_size=1)
        self.segmenter_pub = rospy.Publisher('/segmentation/data', Image, queue_size=1)
        self.image_sub = rospy.Subscriber("/image_data", Image, self.image_sub_callback)

        self.action_pub = rospy.Publisher('/action/data', ActionData, queue_size=1)
        # self.observation_sub = rospy.Subscriber("/action/obs", ObservationData, self.process_observation)
        self.observation_sub = rospy.Subscriber("/action/obs", Image, self.process_observation_1)

        print("Initialized everything")
    
    def image_sub_callback(self, image_data):
        print("Image subscriber callback triggered.")

        # Convert ROS image message to OpenCV image (NumPy array)
        image = self.bridge.imgmsg_to_cv2(image_data, desired_encoding="bgr8")
        
        # Convert from BGR (ROS standard) to RGB if needed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        cv2.imwrite(os.path.join(self.TEST_DIR, "received_image.png"), image)

        processed_masks, pred_mask, raw_masks, bboxes = self.segmenter.from_maskrcnn(image, dir=self.TEST_DIR, bbox=True, dim=(480, 640))

        msg = SegmentationData()
        msg.header.stamp = rospy.Time.now()

        # Predicted mask
        msg.pred_mask = self.bridge.cv2_to_imgmsg(pred_mask.astype('uint8') * 255, encoding='mono8')

        # Raw and processed masks
        msg.raw_masks = convert_numpy_masks_to_ros_image_list([mask for mask in raw_masks], self.bridge)
        msg.processed_masks = convert_numpy_masks_to_ros_image_list(processed_masks, self.bridge)

        # Bounding boxes
        msg.bbox_x1 = [int(x1) for (x1, y1, x2, y2) in bboxes]
        msg.bbox_y1 = [int(y1) for (x1, y1, x2, y2) in bboxes]
        msg.bbox_x2 = [int(x2) for (x1, y1, x2, y2) in bboxes]
        msg.bbox_y2 = [int(y2) for (x1, y1, x2, y2) in bboxes]

        print("Segmentation done and now sending data to robot...")

        # Publish

        pred_mask = np.squeeze(pred_mask)  # remove singleton dim if any
        if pred_mask.ndim == 3:
            pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
        
        pred_mask = pred_mask.astype('uint8') * 255  # Ensure correct type and scale
        msg = self.bridge.cv2_to_imgmsg(pred_mask, encoding='mono8')

        self.segmenter_pub.publish(msg)

    def process_observation(self, obs_data):
        print("Observation subscriber callback triggered.")

        target_image = self.bridge.imgmsg_to_cv2(obs_data, desired_encoding='mono8')
        cv2.imwrite(os.path.join(self.TEST_DIR, "target_image.png"), target_image)
        
        segm_data = obs_data.segmentation_data
        pred_mask = self.bridge.imgmsg_to_cv2(segm_data.pred_mask, desired_encoding='mono8')
        processed_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding='mono8') for m in segm_data.processed_masks]
        bboxes = list(zip(segm_data.bbox_x1, segm_data.bbox_y1, segm_data.bbox_x2, segm_data.bbox_y2))

        color_image = self.bridge.imgmsg_to_cv2(obs_data.color_image, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(obs_data.depth_image, desired_encoding='bgr8')
        target_image = self.bridge.imgmsg_to_cv2(obs_data.target_image, desired_encoding='bgr8')

        # Convert from BGR (ROS standard) to RGB if needed
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(self.TEST_DIR, "color_image_data.png"), color_image)
        cv2.imwrite(os.path.join(self.TEST_DIR, "depth_image_data.png"), depth_image)
        cv2.imwrite(os.path.join(self.TEST_DIR, "target_image_data.png"), target_image)

        # state = self.hmap_generator.generate_heightmap(obs['color'], obs['depth'], self.intrinsics)
        state = self.policy.get_dmap(color_image, depth_image, intrinsics=None)
        # np.save(os.path.join(self.TEST_DIR, 'state.npy'), state)
        print("Gotten the state")

        print("Getting actions...")
        # action = policy.exploit_unveiler(state, obs['color'], target_mask, processed_masks, bboxes)
        action = self.policy.exploit_real_robot(state, target_image)
        print("Gotten the action:", action)

        # Publish
        action_data = ActionData()
        action_data.values = action.tolist()
        self.action_pub.publish(action_data)

        print("Action data published.", action_data.values)

    def process_observation_1(self, obs_data):
        print("Observation subscriber callback triggered.")

        target_image = self.bridge.imgmsg_to_cv2(obs_data, desired_encoding='mono8')
        cv2.imwrite(os.path.join(self.TEST_DIR, "target_image.png"), target_image)
        
        action = np.zeros((4,))
        action[0] = 0.232
        action[1] = 0.233
        action[2] = 0.234
        action[3] = 0.235

        # Publish
        action_data = ActionData()
        action_data.values = action.tolist()
        self.action_pub.publish(action_data)

        print("Action data published.", action_data.values)

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

    def callback(msg):
        print("Received message from Robot:", msg.data)
    
    pub = rospy.Publisher('/machine_robot', String, queue_size=10)
    while pub.get_num_connections() == 0:
        rospy.loginfo("Waiting for robot to subscribe...")
        rospy.sleep(0.5)

    sub = rospy.Subscriber('/robot_machine', String, callback)

#     rate = rospy.Rate(1)  # 1Hz
#     while not rospy.is_shutdown():
#         pub.publish(String(data="Test message from machine"))
#         print("Published message")
#         rate.sleep()

#     rospy.spin()