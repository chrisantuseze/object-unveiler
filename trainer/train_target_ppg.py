import os
import random
# from policy.models_target import ResFCN, Regressor
from policy.models_target_improved import ResFCN, Regressor

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datasets.aperture_dataset import ApertureDataset

from datasets.heightmap_dataset import HeightMapDataset

import utils.logger as logging

# Loss function with focal loss components to handle class imbalance
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for addressing class imbalance in pixel-wise prediction
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    # Apply focal loss formula
    pt = torch.exp(-bce)  # Probability of being correct
    focal_weight = alpha * (1 - pt) ** gamma
    
    return (focal_weight * bce).mean()

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

    writer = SummaryWriter(comment="target_ppg_improved_focal")

    save_path = 'save/fcn-improved-focal'

    if not os.path.exists(save_path):
        os.mkdir(save_path)


    # args.dataset_dir = "/home/e_chrisantus/Projects/grasping_in_clutter/using-pointcloud/single-target-grasping/ppg-ou-dataset2"
    # args.dataset_dir = "/home/e_chrisantus/Projects/grasping_in_clutter/using-pointcloud/episodic-grasping/pc-ou-dataset2"
    transition_dirs = os.listdir(args.dataset_dir)
    
    for file_ in transition_dirs:
        if not file_.startswith("episode"):
            transition_dirs.remove(file_)

    transition_dirs = transition_dirs[:10000]
            
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

    model = ResFCN(args).to(args.device)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Learning rate scheduler with warm-up and cosine annealing
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=args.epochs,
        steps_per_epoch=len(data_loader_train),
        pct_start=0.1,  # Warm-up period
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

    # Gradient clipping value
    clip_value = 1.0
    
    # criterion = nn.BCELoss(reduction='none')
    lowest_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        for step, batch in enumerate(data_loader_train):
            x = batch[0].to(args.device)
            target = batch[1].to(args.device)
            rotations = batch[2]
            y = batch[3].to(args.device, dtype=torch.float)

            pred, aux = model(x, target, rotations)

            # Calculate losses - using focal loss for better handling of imbalanced data
            main_loss = focal_loss(pred, y)
            aux_loss = focal_loss(aux, y)
            
            # Adaptive weighting that changes over time
            # Start with more emphasis on auxiliary loss, gradually shift to main loss
            progress = min(1.0, epoch / (args.epochs * 0.7))  # Reaches 1.0 at 70% of training
            alpha = 0.5 + 0.4 * progress  # Grows from 0.5 to 0.9
            beta = 1.0 - alpha  # Decreases from 0.5 to 0.1
            
            combined_loss = alpha * main_loss + beta * aux_loss
            
            # Add a small regularization term based on attention gamma to prevent extreme values
            gamma_reg = 0.01 * torch.abs(model.gamma).mean()
            combined_loss = combined_loss + gamma_reg
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Compute loss in the whole scene
            # loss = criterion(pred, y)
            # loss = torch.sum(loss)
            # epoch_loss['train'] += loss.detach().cpu().numpy()
            epoch_loss['train'] += combined_loss.item()

            if step % args.step == 0:
                # logging.info(f"train step [{step}/{len(data_loader_train)}]\t Loss: {loss.detach().cpu().numpy()}")
                logging.info(f"train step [{step}/{len(data_loader_train)}]\t Loss: {combined_loss.item()}")

            optimizer.zero_grad()
            # loss.backward()
            combined_loss.backward()
            optimizer.step()

            # Update learning rate
            scheduler.step()

            debug_params(model)

        model.eval()
        # epoch_loss = {'train': 0.0, 'val': 0.0}
        for phase in ['val']:
            for step, batch in enumerate(data_loaders[phase]):
                x = batch[0].to(args.device)
                target = batch[1].to(args.device)
                rotations = batch[2]
                y = batch[3].to(args.device, dtype=torch.float)

                pred, aux = model(x, target, rotations)
                # loss = criterion(pred, y)
                # Calculate validation loss using only main output
                loss = F.binary_cross_entropy_with_logits(pred, y)

                # loss = torch.sum(loss)
                # epoch_loss[phase] += loss.detach().cpu().numpy()
                epoch_loss[phase] += loss.item()

                if step % args.step == 0:
                    # logging.info(f"{phase} step [{step}/{len(data_loaders[phase])}]\t Loss: {loss.detach().cpu().numpy()}")
                    logging.info(f"{phase} step [{step}/{len(data_loaders[phase])}]\t Loss: {loss.item()}")

        # Additional learning rate scheduling based on validation performance
        if epoch > 0 and epoch % 20 == 0:
            # Reduce LR on plateau for additional stability
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.8
                print(f"Reduced learning rate to {param_group['lr']}")

        logging.info('Epoch {}: training loss = {:.6f} '
              ', validation loss = {:.6f}'.format(epoch, epoch_loss['train'] / len(data_loaders['train']),
                                                  epoch_loss['val'] / len(data_loaders['val'])))
        writer.add_scalar("log/train", epoch_loss['train'] / len(data_loaders['train']), epoch)
        writer.add_scalar("log/val", epoch_loss['val'] / len(data_loaders['val']), epoch)

        if lowest_loss > epoch_loss['val']:
            lowest_loss = epoch_loss['val']
            torch.save(model.state_dict(), os.path.join(save_path, f'fcn_model_{epoch}.pt'))

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