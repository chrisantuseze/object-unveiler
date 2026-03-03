

import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

def main():
    # anchor paths relative to this script
    root = os.path.dirname(__file__)
    dataset_dir1 = os.path.join(root, "real_images", "edit_iros")
    dataset_dir2 = os.path.join(root, "real_images", "edit_iros_crops")

    if not os.path.exists(dataset_dir1):
        print(f"Directory not found: {dataset_dir1}")
        return

    # filenames = ['run_1.1.png', 'run_1.2.png', 'run_2.1.png', 'run_2.2.png', 'run_2.3.png']
    filenames = [('run_1.1.png', 'scene_mask_1.1.png', 'target_1.1.png', 'obstacle_1.1.png'), 
                 ('run_1.2.png', 'scene_mask_1.2.png', 'target_1.2.png', 'obstacle_1.2.png'), 
                 ('run_2.1.png', 'scene_mask_2.1.png', 'target_2.1.png', 'obstacle_2.11.png'), 
                 ('run_2.2.png', 'scene_mask_2.2.png', 'target_2.2.png', 'obstacle_2.2.png'),
                 ('run_2.3.png', 'scene_mask_2.3.png', 'target_2.3.png', 'obstacle_2.3.png')]
    
    paths = [(os.path.join(dataset_dir1, f[0]), os.path.join(dataset_dir1, f[1]), 
              os.path.join(dataset_dir1, f[2]), os.path.join(dataset_dir1, f[3])) for f in filenames]

    for color_path, scene_mask_path, target_mask_path, obstacle_mask_path in paths:
        color_image = cv2.imread(color_path, -1)
        scene_mask = cv2.imread(scene_mask_path, -1)
        target_mask = cv2.imread(target_mask_path, -1)
        obstacle_mask = cv2.imread(obstacle_mask_path, -1)

        if color_image is None:
            print(f"Failed to load color image: {color_path}")
            continue
        if scene_mask is None:
            print(f"Failed to load scene mask: {scene_mask_path}")
            continue
        if target_mask is None:
            print(f"Failed to load target mask: {target_mask_path}")
            continue
        if obstacle_mask is None:
            print(f"Failed to load obstacle mask: {obstacle_mask_path}")
            continue

        target_mask = extract_target_crop(target_mask, color_image)
        obstacle_mask = extract_target_crop(obstacle_mask, color_image)

        # Special handling for obstacle_2.2 to use run_2.1.png as the color image
        if 'obstacle_2.2' in obstacle_mask_path:
            color_image_ = cv2.imread(os.path.join(dataset_dir1, 'run_2.1.png'), -1)
            obstacle_mask = cv2.imread(os.path.join(dataset_dir1, 'obstacle_2.2.png'), -1)

            obstacle_mask = extract_target_crop(obstacle_mask, color_image_)

        visualize(color_image, scene_mask, target_mask, obstacle_mask)

def extract_target_crop(target, scene):
    print(f"target.shape={target.shape}, scene.shape={scene.shape}")
    mask = target
    image = scene

    # Normalize mask to single channel
    if mask is None:
        raise ValueError("mask is None")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Threshold mask to binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # If sizes differ, resize the color image to match the mask size (preserve mask binary)
    if image.shape[:2] != mask.shape[:2]:
        new_w, new_h = mask.shape[1], mask.shape[0]
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a copy of the (possibly resized) image to modify
    result = image.copy()

    # Estimate background color from the edges of the (resized) image
    border_width = 10
    h, w = image.shape[:2]
    bw = min(border_width, h // 4, w // 4)
    top_border = image[:bw, :, :]
    bottom_border = image[-bw:, :, :]
    left_border = image[:, :bw, :]
    right_border = image[:, -bw:, :]

    all_borders = np.concatenate([
        top_border.reshape(-1, 3),
        bottom_border.reshape(-1, 3),
        left_border.reshape(-1, 3),
        right_border.reshape(-1, 3)
    ])

    background_color = tuple(map(int, np.mean(all_borders, axis=0)))

    # Create a background color image
    background = np.ones_like(image, dtype=np.uint8) * np.array(background_color, dtype=np.uint8)

    # Replace non-object regions with the background color; keep object pixels from image
    mask_3c = np.repeat((mask > 0)[:, :, np.newaxis], 3, axis=2)
    result = np.where(mask_3c, image, background).astype(np.uint8)

    return result

def bgr_to_rgb(image):
    if image is not None and image.ndim == 3:
        ch = image.shape[2]
        if ch == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif ch == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image

def visualize(color_image, scene_mask, target_mask, obstacle_mask):
    fig, ax = plt.subplots(2, 2)

    # Convert OpenCV BGR(A) to Matplotlib RGB(A)
    color_display = bgr_to_rgb(color_image)

    ax[0][0].imshow(color_display)
    ax[0][0].set_title("Scene - Color")
    ax[0][0].axis("off")

    # Show masks as grayscale
    ax[0][1].imshow(scene_mask, cmap='gray')
    ax[0][1].set_title("Scene - Grayscale")
    ax[0][1].axis("off")

    target_image = bgr_to_rgb(target_mask)
    obstacle_image = bgr_to_rgb(obstacle_mask)
    ax[1][0].imshow(target_image)
    ax[1][0].set_title("Target")
    ax[1][0].axis("off")

    ax[1][1].imshow(obstacle_image)
    ax[1][1].set_title("Obstacle")
    ax[1][1].axis("off")

    plt.show()



if __name__ == "__main__":
    main()