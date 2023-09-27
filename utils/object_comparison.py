import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import utils.logger as logging

def grayscale_to_rgb_pil(image):
    # Convert the image to PIL image.
    pil_image = Image.fromarray(image)

    # Convert the PIL image to RGB.
    rgb_image = pil_image.convert('RGB')

    # Return the RGB image.
    return rgb_image

# Load and preprocess two images
def preprocess_image(preprocess, mask):
    mask = grayscale_to_rgb_pil(mask)

    img = preprocess(mask)
    img = img.unsqueeze(0)
    return img

def object_compare(mask1, mask2):
    # Load the pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer

    # Set the model to evaluation mode
    model.eval()

    # Define image preprocessing transforms
    preprocess = transforms.Compose([
        # transforms.ToPILImage(),       # Convert NumPy array to PIL Image
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img1 = preprocess_image(preprocess, mask1)
    img2 = preprocess_image(preprocess, mask2)

    # Extract features using the ResNet50 model
    with torch.no_grad():
        features1 = model(img1)
        features2 = model(img2)

    # Flatten the feature tensors
    flat_features1 = features1.view(features1.size(0), -1)
    flat_features2 = features2.view(features2.size(0), -1)

    # Calculate cosine similarity between the feature tensors
    similarity_score = nn.functional.cosine_similarity(flat_features1, flat_features2).item()

    # Define a threshold to determine if the object appears in both images
    threshold = 0.9

    # Compare the similarity score with the threshold
    is_object_similar = similarity_score > threshold
    # if is_object_similar:
    #     logging.info("The object appears in both images.", similarity_score)
    # else:
    #     logging.info("The object does not appear in both images.", similarity_score)

    return is_object_similar
