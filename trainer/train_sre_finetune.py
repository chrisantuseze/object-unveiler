from copy import deepcopy
import os
import random
from policy.sre_model import SpatialEncoder, compute_loss
# from policy.obstacle_decoder import SpatialTransformerPredictor

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from datasets.sre_dataset import SREDataset

import utils.logger as logging


def train_sre(args):
    """
    Trains a Transformer Encoder policy model for obstacle prediction using the provided arguments.
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

    writer = SummaryWriter(comment="sre-no-edges")

    save_path = 'save/sre-no-edges'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # args.dataset_dir = "/home/e_chrisantus/Projects/grasping_in_clutter/using-pointcloud/old-episodic-grasping/pc-ou-dataset2"
    transition_dirs = os.listdir(args.dataset_dir)
    
    for file_ in transition_dirs:
        if not file_.startswith("episode"):
            transition_dirs.remove(file_)

    # transition_dirs = transition_dirs[:500]
    
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
    
    train_dataset = SREDataset(args, train_ids)
    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, shuffle=True)

    val_dataset = SREDataset(args, val_ids)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    args.step = int(len(train_ids)/(4*args.batch_size))

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    logging.info('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    model = SpatialEncoder(args).to(args.device)
    model, params_to_update = prepare_model(args, model_path=args.sre_model, model=model)
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    
    criterion = nn.CrossEntropyLoss()
    lowest_loss = float('inf')
    best_ckpt_info = None
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        for step, batch in enumerate(data_loader_train):
            target = batch[0].to(args.device)
            object_masks = batch[1].to(args.device)

            bbox = batch[2].to(args.device)
            objects_to_remove = batch[3].to(args.device)
            raw_scene_mask = batch[4].to(args.device)
            raw_target = batch[5].to(args.device)
            raw_objects = batch[6].to(args.device)
            
            pred, valid_mask = model(target, object_masks, bbox, raw_scene_mask, raw_target, raw_objects)

            # Compute loss in the whole scene
            loss = compute_loss(pred, objects_to_remove, valid_mask) 
            # loss = criterion(pred, objects_to_remove)
            
            # loss = torch.sum(loss)
            epoch_loss['train'] += loss.detach().cpu().numpy()

            if step % args.step == 0:
                # print_pred_gt(torch.topk(pred, k=args.sequence_length, dim=1)[1], objects_to_remove)
                logging.info(f"train step [{step}/{len(data_loader_train)}]\t Loss: {loss.detach().cpu().numpy()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            debug_params(model)

        model.eval()
        # epoch_loss = {'train': 0.0, 'val': 0.0}
        for phase in ['val']:
            for step, batch in enumerate(data_loaders[phase]):
                target = batch[0].to(args.device)
                object_masks = batch[1].to(args.device)

                bbox = batch[2].to(args.device)
                objects_to_remove = batch[3].to(args.device)

                raw_scene_mask = batch[4].to(args.device)
                raw_target = batch[5].to(args.device)
                raw_objects = batch[6].to(args.device)
                
                pred, valid_mask = model(target, object_masks, bbox, raw_scene_mask, raw_target, raw_objects)

                # Compute loss in the whole scene
                loss = compute_loss(pred, objects_to_remove, valid_mask) 
                # loss = criterion(pred, objects_to_remove)

                # loss = torch.sum(loss)
                epoch_loss[phase] += loss.detach().cpu().numpy()

                if step % args.step == 0:
                    # print_pred_gt(torch.topk(pred, k=args.sequence_length, dim=1)[1], objects_to_remove)
                    logging.info(f"{phase} step [{step}/{len(data_loaders[phase])}]\t Loss: {loss.detach().cpu().numpy()}")

        logging.info('Epoch {}: training loss = {:.6f} '
              ', validation loss = {:.6f}'.format(epoch, epoch_loss['train'] / len(data_loaders['train']),
                                                  epoch_loss['val'] / len(data_loaders['val'])))
        writer.add_scalar("log/train", epoch_loss['train'] / len(data_loaders['train']), epoch)
        writer.add_scalar("log/val", epoch_loss['val'] / len(data_loaders['val']), epoch)

        if epoch % 25 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'sre_model_{epoch}.pt'))

        if lowest_loss > epoch_loss['val']:
            lowest_loss = epoch_loss['val']
            best_ckpt_info = (epoch, lowest_loss, deepcopy(model.state_dict()))
            torch.save(model.state_dict(), os.path.join(save_path, f'sre_model_best.pt'))

    # save best checkpoint
    best_epoch, lowest_val_loss, best_state_dict = best_ckpt_info
    torch.save(best_state_dict, os.path.join(save_path, f'sre_model_best.pt'))
    print(f'Best ckpt, val loss {lowest_val_loss:.6f} @ epoch{best_epoch}')

    torch.save(model.state_dict(), os.path.join(save_path, f'sre_model_last.pt'))
    writer.close()

def print_pred_gt(pred, gt):
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    preds = [list(p)[0] for p in pred]
    gts = [g for g in gt]
    print("Pred:", preds)
    print("Grdt:", gts)
    print()

def debug_params(model):
    for name, param in model.named_parameters():
        if param.grad is None:  
            logging.info(name, " gradient is None!")
            module = name.split('.')[0]   
            logging.info("Checking module:", module)

            # For example, get parent module with getattr 
            parent = getattr(model, module) 
            logging.info("Parent:",parent)

def get_params_to_update(model):
    params_to_update = []

    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    return params_to_update

def prepare_model(args, model_path, model):
    # params_to_update = model.parameters()

    logging.info(f"Loading checkpoint from - {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=args.device))

    params_to_update = []
    for name, param in model.named_parameters():
        if 'layers.3' in name or 'layers.4' in name or 'layers.5' in name:
            params_to_update.append(param)
            continue

        param.requires_grad = False

    # params_to_update = get_params_to_update(model)

    return model, params_to_update