from datetime import datetime
import shutil
import os

START_TIME = datetime.now()

# # NOT FIXED DATASET
DATA_TYPE = "NOT_FIXED"
CKPT_DIR = '/home/selam/checkpoints/'
#for pretrained clip head
GELSIGHT_WEIGHTS_PATH = '/home/selam/data/camera_cage_new_notfixed/clip_models/trained/epoch_1499_gelsight_encoder.pth'
IMAGE_WEIGHTS_PATH = '/home/selam/data/camera_cage_new_notfixed/clip_models/trained/epoch_1499_vision_encoder.pth'
ENC_TYPE = 'clip'  # weights above not actually used here
DEVICE_STR = 'cuda:0'
PRED_HORIZON = 20
ABLATE_GEL = False 
GEL_ONLY = False

DATA_DIR = 'save' #'<put your data dir here>'
SIM_TASK_CONFIGS = {
    'sim_object_unveiler': {
        'dataset_dir': DATA_DIR + '/pc-ou-dataset',
        'num_episodes': 50,
        'episode_len': 3,
        'camera_names': ['top', 'front', 'target']
    },
}

