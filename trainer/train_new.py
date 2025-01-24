import os
import random
import copy
from policy.models_attn2 import Regressor, ResFCN
# from policy.models_multi_task import Regressor, ResFCN
# from policy.models_obstacle import Regressor, ResFCN
# from policy.models_obstacle_attn import Regressor, ResFCN
# from policy.models_obstacle_heuristics import Regressor, ResFCN
# from policy.models_obstacle_vit import Regressor, ResFCN

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from datasets.aperture_dataset import ApertureDataset
from datasets.unveiler_datasets import UnveilerDataset

import utils.general_utils as general_utils
import utils.logger as logging

# models_attn
def train_fcn_net(args):
    """
    Train an attention-based policy for object unveiling in cluttered environments.
    Args:
        args (Namespace): A namespace object containing the following attributes:
            - dataset_dir (str): Directory containing the dataset.
            - split_ratio (float): Ratio to split the dataset into training and validation sets.
            - batch_size (int): Batch size for training and validation.
            - device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
            - lr (float): Learning rate for the optimizer.
            - epochs (int): Number of epochs to train the model.
            - step (int): Number of steps per epoch.
    Returns:
        None
    """

    writer = SummaryWriter()
    
    save_path = 'save/fcn'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    transition_dirs = os.listdir(args.dataset_dir)
    for file_ in transition_dirs:
        if not file_.startswith("episode"):
            transition_dirs.remove(file_)

    # split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)

    # transition_dirs = transition_dirs[:10000]

    split_index = int(args.split_ratio * len(transition_dirs))
    train_ids = transition_dirs[:split_index]
    val_ids = transition_dirs[split_index:]

    # this ensures that the split is done properly without causing input mismatch error
    data_length = (len(train_ids)//args.batch_size) * args.batch_size
    train_ids = train_ids[:data_length]

    data_length = (len(val_ids)//args.batch_size) * args.batch_size
    val_ids = val_ids[:data_length]
    
    train_dataset = UnveilerDataset(args, train_ids)
    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)

    val_dataset = UnveilerDataset(args, val_ids)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    args.step = int(len(train_ids)/(4*args.batch_size))

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    logging.info('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    model = ResFCN(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.BCELoss(reduction='none')

    global_step = 0 #{'train': 0, 'val': 0}
    lowest_loss = float('inf')
    for epoch in range(args.epochs):
        
        model.train()
        for step, batch in enumerate(data_loader_train):
            x = batch[0].to(args.device) 
            target_mask = batch[1].to(args.device, dtype=torch.float32)
            object_masks = batch[2].to(args.device)
            scene_masks = batch[3].to(args.device)

            # raw_x = batch[4].to(args.device)
            # raw_target_mask = batch[5].to(args.device, dtype=torch.float32)
            # raw_object_masks = batch[6].to(args.device)
            # rotations = batch[7]
            # y = batch[8].to(args.device, dtype=torch.float32)
            # obstacle_gt = batch[9].to(args.device, dtype=torch.float32)
            # bboxes = batch[10].to(args.device, dtype=torch.float32)

            rotations = batch[4]
            y = batch[5].to(args.device, dtype=torch.float32)
            obstacle_gt = batch[6].to(args.device, dtype=torch.float32)
            bboxes = batch[7].to(args.device, dtype=torch.float32)

            obstacle_pred = model(
                x, target_mask, object_masks, scene_masks, 
                # raw_x, raw_target_mask, raw_object_masks,
                bboxes,
                rotations
            )

            loss = criterion(obstacle_pred, y)
            loss = torch.sum(loss)

            if step % (args.step * 2) == 0:
                logging.info(f"train step [{step}/{len(data_loader_train)}]\t Loss: {loss.detach().cpu().numpy()}")

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            debug_params(model)

            # grad_norm = calculate_gradient_norm(model) 

            # writer.add_scalar("norm/train", grad_norm, global_step)
            # global_step += 1

        model.eval()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        for phase in ['train', 'val']:
            for step, batch in enumerate(data_loaders[phase]):
                x = batch[0].to(args.device) 
                target_mask = batch[1].to(args.device, dtype=torch.float32)
                object_masks = batch[2].to(args.device)
                scene_masks = batch[3].to(args.device)

                # raw_x = batch[4].to(args.device)
                # raw_target_mask = batch[5].to(args.device, dtype=torch.float32)
                # raw_object_masks = batch[6].to(args.device)
                # rotations = batch[7]
                # y = batch[8].to(args.device, dtype=torch.float32)
                # obstacle_gt = batch[9].to(args.device, dtype=torch.float32)
                # bboxes = batch[10].to(args.device, dtype=torch.float32)

                rotations = batch[4]
                y = batch[5].to(args.device, dtype=torch.float32)
                obstacle_gt = batch[6].to(args.device, dtype=torch.float32)
                bboxes = batch[7].to(args.device, dtype=torch.float32)

                obstacle_pred = model(
                    x, target_mask, object_masks, scene_masks, 
                    # raw_x, raw_target_mask, raw_object_masks,
                    bboxes,
                    rotations
                )

                loss = criterion(obstacle_pred, y)
                loss = torch.sum(loss)

                epoch_loss[phase] += loss.detach().cpu().numpy()

                if step % args.step == 0:
                    logging.info(f"{phase} step [{step}/{len(data_loaders[phase])}]\t Loss: {loss.detach().cpu().numpy()}")

        logging.info('Epoch {}: training loss = {:.8f} '
              ', validation loss = {:.8f}'.format(epoch, epoch_loss['train'] / len(data_loaders['train']),
                                                  epoch_loss['val'] / len(data_loaders['val'])))
        
        writer.add_scalar("log/train", epoch_loss['train'] / len(data_loaders['train']), epoch)
        writer.add_scalar("log/val", epoch_loss['val'] / len(data_loaders['val']), epoch)

        if lowest_loss > epoch_loss['val']:
            lowest_loss = epoch_loss['val']
            torch.save(model.state_dict(), os.path.join(save_path, f'fcn_model_{epoch}.pt'))

    torch.save(model.state_dict(), os.path.join(save_path,  f'fcn_model.pt'))
    writer.close()

def calculate_gradient_norm(model):
    # Calculate gradient norm here
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is None:
            logging.info(f"{param} grad is None")
        else:
            grad_norm += param.grad.data.norm(2) ** 2

    grad_norm = grad_norm ** (1. / 2)

    return grad_norm

def debug_params(model):
    for name, param in model.named_parameters():
        if param.grad is None:  
            logging.info(name, " gradient is None!")
            module = name.split('.')[0]   
            logging.info("Checking module:", module)

            # For example, get parent module with getattr 
            parent = getattr(model, module) 
            logging.info("Parent:",parent)

def train_regressor(args):
    save_path = 'save/reg'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # transition_dirs = next(os.walk(args.dataset_dir))[1]
    transition_dirs = os.listdir(args.dataset_dir)

    # split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)
    
    split_index = int(args.split_ratio * len(transition_dirs))
    train_ids = transition_dirs[:split_index]
    val_ids = transition_dirs[split_index:]

    train_dataset = ApertureDataset(args, train_ids)
    val_dataset = ApertureDataset(args, val_ids)

    # note: the batch size is 4
    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size)

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    logging.info('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    model = Regressor().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    logging.info(model)