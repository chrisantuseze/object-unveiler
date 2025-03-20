from copy import deepcopy
import os
import random
# from policy.models_target import ResFCN, Regressor
from policy.models_target_new import ActionDecoder, Regressor

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datasets.aperture_dataset import ApertureDataset

from datasets.heightmap_dataset import HeightMapDataset

import utils.logger as logging

def train_ae(args):
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

    writer = SummaryWriter(comment="ae")

    save_path = 'save/ae'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    transition_dirs = os.listdir(args.dataset_dir)
    
    for file_ in transition_dirs:
        if not file_.startswith("episode"):
            transition_dirs.remove(file_)

    transition_dirs = transition_dirs[:3000]
            
    # split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)

    print(f'\nData from: {args.dataset_dir}; size: {len(transition_dirs)}\n')

    split_index = int(args.split_ratio * len(transition_dirs))
    train_ids = transition_dirs[:split_index]
    val_ids = transition_dirs[split_index:]

    # this ensures that the split is done properly without causing input mismatch error
    data_length = (len(train_ids)//args.batch_size) * args.batch_size
    train_ids = train_ids[:data_length]

    data_length = (len(val_ids)//args.batch_size) * args.batch_size
    val_ids = val_ids[:data_length]
    
    train_dataset = HeightMapDataset(args, train_ids)
    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, shuffle=True)

    val_dataset = HeightMapDataset(args, val_ids)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    args.step = int(len(train_ids)/(4*args.batch_size))

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    logging.info('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    model = ActionDecoder(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.05)

    criterion = nn.BCELoss(reduction='mean')
    lowest_loss = float('inf')
    best_ckpt_info = None
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        for step, batch in enumerate(data_loader_train):
            x = batch[0].to(args.device)
            target = batch[1].to(args.device)
            rotations = batch[2]
            y = batch[3].to(args.device, dtype=torch.float)

            pred = model(x, target, rotations)

            # Compute loss in the whole scene
            loss = criterion(pred, y)
            # loss = torch.sum(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            debug_params(model)

        model.eval()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        for phase in ['train', 'val']:
            for step, batch in enumerate(data_loaders[phase]):
                x = batch[0].to(args.device)
                target = batch[1].to(args.device)
                rotations = batch[2]
                y = batch[3].to(args.device, dtype=torch.float)

                pred = model(x, target, rotations)
                loss = criterion(pred, y)

                # loss = torch.sum(loss)
                epoch_loss[phase] += loss.detach().cpu().numpy()

                if step % args.step == 0:
                    logging.info(f"{phase} step [{step}/{len(data_loaders[phase])}]\t Loss: {loss.detach().cpu().numpy()}")

        scheduler.step()

        logging.info('Epoch {}: training loss = {:.6f} '
              ', validation loss = {:.6f}, lr = {}'.format(epoch, epoch_loss['train'] / len(data_loaders['train']),
                                                  epoch_loss['val'] / len(data_loaders['val']), scheduler.get_last_lr()))
        writer.add_scalar("log/train", epoch_loss['train'] / len(data_loaders['train']), epoch)
        writer.add_scalar("log/val", epoch_loss['val'] / len(data_loaders['val']), epoch)

        if epoch % 25 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'ae_model_{epoch}.pt'))

        if lowest_loss > epoch_loss['val']:
            lowest_loss = epoch_loss['val']
            best_ckpt_info = (epoch, lowest_loss, deepcopy(model.state_dict()))
            torch.save(model.state_dict(), os.path.join(save_path, f'ae_model_best.pt'))

    # save best checkpoint
    best_epoch, lowest_val_loss, best_state_dict = best_ckpt_info
    torch.save(best_state_dict, os.path.join(save_path, f'ae_model_best.pt'))
    print(f'Best ckpt, val loss {lowest_val_loss:.6f} @ epoch{best_epoch}')

    torch.save(model.state_dict(), os.path.join(save_path, f'ae_model_last.pt'))
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