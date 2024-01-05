import os
import random
from policy.models_target import Regressor
from policy.models_target import ResFCN

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from datasets.aperture_dataset import ApertureDataset

from datasets.heightmap_dataset import HeightMapDataset

import utils.logger as logging


def train_fcn_net(args):
    writer = SummaryWriter()

    save_path = 'save/fcn'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    transition_dirs = os.listdir(args.dataset_dir)
    
    for file_ in transition_dirs:
        # if not file_.startswith("episode"):
        #     transition_dirs.remove(file_)
        if not file_.startswith("transition"):
            transition_dirs.remove(file_)
    
    # split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)

    transition_dirs = transition_dirs[:12000]

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

    model = ResFCN().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss(reduction='none')

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(data_loader_train):
            x = batch[0].to(args.device)
            target = batch[1].to(args.device)
            rotations = batch[2]
            y = batch[3].to(args.device, dtype=torch.float)

            pred = model(x, target, rotations)

            # Compute loss in the whole scene
            loss = criterion(pred, y)
            loss = torch.sum(loss)

            # logging.info(f"train step [{step}/{len(data_loader_train)}]\t Loss: {loss.detach().cpu().numpy()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

                loss = torch.sum(loss)
                epoch_loss[phase] += loss.detach().cpu().numpy()

                if step % args.step == 0:
                    logging.info(f"{phase} step [{step}/{len(data_loaders[phase])}]\t Loss: {loss.detach().cpu().numpy()}")

        logging.info('Epoch {}: training loss = {:.6f} '
              ', validation loss = {:.6f}'.format(epoch, epoch_loss['train'] / len(data_loaders['train']),
                                                  epoch_loss['val'] / len(data_loaders['val'])))
        writer.add_scalar("log/train", epoch_loss['train'] / len(data_loaders['train']), epoch)
        writer.add_scalar("log/val", epoch_loss['val'] / len(data_loaders['val']), epoch)

    torch.save(model.state_dict(), os.path.join(save_path,  f'fcn_model.pt'))
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