import os
import cv2
import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from trainer.memory import ReplayBuffer
from utils import general_utils

class ZeroShotCLIPRemovalPredictor:
    def __init__(self, clip_model="ViT-B/32"):
        """
        Initialize the zero-shot CLIP predictor for obstacle removal
        
        Args:
            clip_model: CLIP model version to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        
        # Define text prompts for the task
        self.text_prompts = [
            "an object blocking access to the target",
            "an obstacle preventing grasping the target",
            "the first object that should be removed",
            "an object directly obstructing the target",
            "the most important obstacle to remove"
        ]
        
        # Encode text prompts
        text = clip.tokenize(self.text_prompts).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
    
    def prepare_scene_image(self, mask_list, target_mask, canvas_size=(224, 224)):
        """
        Create scene images from masks for CLIP processing
        
        Args:
            mask_list: List of object masks
            target_mask: Target object mask
            canvas_size: Size of the output images
            
        Returns:
            list of processed images for CLIP
        """
        processed_images = []
        
        for i, obstacle_mask in enumerate(mask_list):
            # Create a 3-channel image
            scene_img = np.zeros((obstacle_mask.shape[0], obstacle_mask.shape[1], 3))
            
            # Set target as green
            scene_img[:,:,1] = target_mask
            
            # Set current obstacle as red (highlighted)
            scene_img[:,:,0] = obstacle_mask
            
            # Set other obstacles as blue (background)
            for j, other_mask in enumerate(mask_list):
                if j != i:
                    scene_img[:,:,2] += other_mask
            
            # Normalize to 0-1 range
            scene_img = np.clip(scene_img, 0, 1)
            
            # Convert to PIL Image and preprocess for CLIP
            pil_img = Image.fromarray((scene_img * 255).astype(np.uint8))
            processed_img = self.preprocess(pil_img)
            processed_images.append(processed_img)
            
        return processed_images
    
    def predict_removal(self, mask_list, target_mask):
        """
        Predict which object should be removed first
        
        Args:
            mask_list: List of object masks (numpy arrays with 0-1 values)
            target_mask: Target object mask (numpy array with 0-1 values)
            visualize: Whether to visualize the masks and prediction
            
        Returns:
            index of the object that should be removed first
        """
        if len(mask_list) == 0:
            return None
            
        # Prepare images for each obstacle+target combination
        processed_images = self.prepare_scene_image(mask_list, target_mask)
        
        # Stack images and process with CLIP
        image_tensor = torch.stack(processed_images).to(self.device)
        
        with torch.no_grad():
            # Encode images
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            
            # Average over all prompts
            avg_similarity = similarity.mean(dim=-1)
            
            # Get index of the best match
            best_match_idx = torch.argmax(avg_similarity).item()
        
        return best_match_idx
    
    def eval(self):
        dataset_dir = "real_images/seg_data"

        transition_dirs = os.listdir(dataset_dir)

        for file_ in transition_dirs:
            if not file_.startswith("transition"):
                transition_dirs.remove(file_)
                
        memory = ReplayBuffer(dataset_dir)

        for idx, transition_dir in enumerate(transition_dirs):
            print("Transition Directory:", transition_dir)
            scene_image, scene_mask, target_mask, bboxes, target_id, object_masks = memory.load_seg_data(transition_dirs, idx)
            scene_image = cv2.resize(scene_image, (400, 400)) 

            prediction = self.predict_removal(object_masks, target_mask)
            obstacle_mask = object_masks[prediction]
 
            c_target_mask = general_utils.extract_target_crop2(target_mask, scene_image)
            c_obstacle_mask = general_utils.extract_target_crop2(obstacle_mask, scene_image)

            print("Target ID:", target_id)
            print("Obstacle ID:", prediction)
            print()

            # fig, ax = plt.subplots(2, 2)
            # ax[0][0].imshow(scene_image)
            # ax[0][0].set_title("Scene - Color")
            # ax[0][0].axis("off")

            # ax[0][1].imshow(scene_mask)
            # ax[0][1].set_title("Scene - Grayscale")
            # ax[0][1].axis("off")

            # ax[1][0].imshow(c_target_mask)
            # ax[1][0].set_title("Target")
            # ax[1][0].axis("off")

            # ax[1][1].imshow(c_obstacle_mask)
            # ax[1][1].set_title("Obstacle")
            # ax[1][1].axis("off")
            # plt.show()

if __name__ == "__main__":
    
    # Initialize the predictor
    clip_predictor = ZeroShotCLIPRemovalPredictor()
    
    clip_predictor.eval()