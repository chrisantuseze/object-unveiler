import os
import random
from policy.models_target import Regressor
from policy.obstacle_transformer import TransformerObstaclePredictor

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from datasets.aperture_dataset import ApertureDataset

from datasets.transformer_dataset import TransformerDataset

import utils.logger as logging


def train_fcn_net(args):
    """
    Trains a Fully Convolutional Network (FCN) policy model for target grasping using the provided arguments.
    Args:
        args: An object containing the following attributes:
            - dataset_dir (str): Directory containing the dataset.
            - split_ratio (float): Ratio to split the dataset into training and validation sets.
            - batch_size (int): Batch size for training and validation.
            - epochs (int): Number of epochs to train the model.
            - lr (float): Learning rate for the optimizer.
            - device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
    Returns:
        None
    """

    writer = SummaryWriter()

    save_path = 'save/fcn'

    if not os.path.exists(save_path):
        os.mkdir(save_path)


    args.dataset_dir = "/home/e_chrisantus/Projects/grasping_in_clutter/using-pointcloud/episodic-grasping/pc-ou-dataset2"
    transition_dirs = os.listdir(args.dataset_dir)
    
    for file_ in transition_dirs:
        if not file_.startswith("episode"):
            transition_dirs.remove(file_)
    
    # split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)

    split_index = int(args.split_ratio * len(transition_dirs))
    train_ids = transition_dirs[:split_index]
    val_ids = transition_dirs[split_index:]

    # this ensures that the split is done properly without causing input mismatch error
    data_length = (len(train_ids)//args.batch_size) * args.batch_size
    train_ids = train_ids[:data_length]

    data_length = (len(val_ids)//args.batch_size) * args.batch_size
    val_ids = val_ids[:data_length]
    
    train_dataset = TransformerDataset(args, train_ids)
    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, shuffle=True)

    val_dataset = TransformerDataset(args, val_ids)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    args.step = int(len(train_ids)/(4*args.batch_size))

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    logging.info('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    model = TransformerObstaclePredictor(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(data_loader_train):
            target = batch[0].to(args.device)
            object_masks = batch[1].to(args.device)

            # return processed_target_mask, processed_obj_masks, bbox, padded_objects_to_remove, raw_scene_mask, raw_target_mask, raw_object_masks
            bbox = batch[2].to(args.device)
            objects_to_remove = batch[3].to(args.device)
            raw_scene_mask = batch[4].to(args.device)
            raw_target = batch[5].to(args.device)
            raw_objects = batch[6].to(args.device)
            
            pred = model(target, object_masks, bbox, objects_to_remove, raw_scene_mask, raw_target, raw_objects)

            # Compute loss in the whole scene
            loss = criterion(pred, objects_to_remove)
            loss = torch.sum(loss)

            if step % args.step == 0:
                print("gt/pred = ", objects_to_remove, "/", torch.topk(pred, k=args.num_patches, dim=1)[1])
                logging.info(f"train step [{step}/{len(data_loader_train)}]\t Loss: {loss.detach().cpu().numpy()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            debug_params(model)

        model.eval()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        for phase in ['train', 'val']:
            for step, batch in enumerate(data_loaders[phase]):
                target = batch[0].to(args.device)
                object_masks = batch[1].to(args.device)

                # return processed_target_mask, processed_obj_masks, bbox, padded_objects_to_remove, raw_scene_mask, raw_target_mask, raw_object_masks
                bbox = batch[2].to(args.device)
                objects_to_remove = batch[3].to(args.device)

                raw_scene_mask = batch[4].to(args.device)
                raw_target = batch[5].to(args.device)
                raw_objects = batch[6].to(args.device)
                
                pred = model(target, object_masks, bbox, objects_to_remove, raw_scene_mask, raw_target, raw_objects)

                loss = criterion(pred, objects_to_remove)

                loss = torch.sum(loss)
                epoch_loss[phase] += loss.detach().cpu().numpy()

                if step % args.step == 0:
                    print("gt/pred = ", objects_to_remove, "/", torch.topk(pred, k=args.num_patches, dim=1)[1])
                    logging.info(f"{phase} step [{step}/{len(data_loaders[phase])}]\t Loss: {loss.detach().cpu().numpy()}")

        logging.info('Epoch {}: training loss = {:.6f} '
              ', validation loss = {:.6f}'.format(epoch, epoch_loss['train'] / len(data_loaders['train']),
                                                  epoch_loss['val'] / len(data_loaders['val'])))
        writer.add_scalar("log/train", epoch_loss['train'] / len(data_loaders['train']), epoch)
        writer.add_scalar("log/val", epoch_loss['val'] / len(data_loaders['val']), epoch)

        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'fcn_model.pt'))

    torch.save(model.state_dict(), os.path.join(save_path, f'fcn_model.pt'))
    writer.close()


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



def debug_params(model):
    for name, param in model.named_parameters():
        if param.grad is None:  
            logging.info(name, " gradient is None!")
            module = name.split('.')[0]   
            logging.info("Checking module:", module)

            # For example, get parent module with getattr 
            parent = getattr(model, module) 
            logging.info("Parent:",parent)