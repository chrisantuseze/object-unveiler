import numpy as np
import math

IS_REAL = False

if IS_REAL:
    WORKSPACE_LIMITS = np.asarray([[-0.227, 0.221], [-0.676, -0.228], [0.18, 0.4]])
else:
    WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.224, 0.224], [-0.0001, 0.4]])

IMAGE_SIZE = 224

IMAGE_WIDTH = 400 #640
IMAGE_HEIGHT = 400 #480

IMAGE_OBJ_CROP_SIZE = 60  # this is related to the IMAGE_SIZE and PIXEL_SIZE
IMAGE_PAD_SIZE = math.ceil(IMAGE_SIZE * math.sqrt(2) / 32) * 32  # 320
IMAGE_PAD_WIDTH = math.ceil((IMAGE_PAD_SIZE - IMAGE_SIZE) / 2)  # 48
IMAGE_PAD_DIFF = IMAGE_PAD_SIZE - IMAGE_PAD_WIDTH  # 272


background_threshold = {
    "low": np.array([0, 0, 125], np.uint8),
    "high": np.array([255, 255, 255], np.uint8),
}  # white
BG_THRESHOLD = {
    "low": np.array([0, 0, 0], np.uint8),
    "high": np.array([180, 255, 50], np.uint8),
}  # black
# colors
real_purple_lower = np.array([100, 143, 0], np.uint8)
real_purple_upper = np.array([126, 255, 255], np.uint8)
# rgb(69, 108, 149) to hsv(105 137 149)
blue_lower = np.array([95, 87, 99], np.uint8)
blue_upper = np.array([115, 187, 199], np.uint8)
# rgb(79, 143, 70) to hsv(56 130 143)
green_lower = np.array([48, 80, 87], np.uint8)
green_upper = np.array([64, 180, 187], np.uint8)
# 11  97 131
brown_lower = np.array([8, 57, 91], np.uint8)
brown_upper = np.array([14, 137, 171], np.uint8)
# 15 209 206
orange_lower = np.array([12, 159, 156], np.uint8)
orange_upper = np.array([18, 255, 255], np.uint8)
# 23 177 202
yellow_lower = np.array([20, 127, 152], np.uint8)
yellow_upper = np.array([26, 227, 252], np.uint8)
# 158, 148, 146 to 5 19 158
gray_lower = np.array([0, 0, 108], np.uint8)
gray_upper = np.array([15, 56, 208], np.uint8)
# rgb(217, 74, 76) to 0 168 217
red_lower = np.array([0, 118, 172], np.uint8)
red_upper = np.array([10, 218, 255], np.uint8)
# rgb(148, 104, 136) to 158  76 148
purple_lower = np.array([148, 26, 98], np.uint8)
purple_upper = np.array([167, 126, 198], np.uint8)
# rgb(101, 156, 151) to 87  90 156
cyan_lower = np.array([77, 40, 106], np.uint8)
cyan_upper = np.array([97, 140, 206], np.uint8)
# rgb(216, 132, 141) to 177  99 216
pink_lower = np.array([168, 49, 166], np.uint8)
pink_upper = np.array([187, 149, 255], np.uint8)
colors_lower = [
    blue_lower,
    green_lower,
    brown_lower,
    orange_lower,
    yellow_lower,
    gray_lower,
    red_lower,
    purple_lower,
    cyan_lower,
    pink_lower,
]
colors_upper = [
    blue_upper,
    green_upper,
    brown_upper,
    orange_upper,
    yellow_upper,
    gray_upper,
    red_upper,
    purple_upper,
    cyan_upper,
    pink_upper,
]


if IS_REAL:
    TARGET_LOWER = real_purple_lower
    TARGET_UPPER = real_purple_upper
else:
    TARGET_LOWER = blue_lower
    TARGET_UPPER = blue_upper