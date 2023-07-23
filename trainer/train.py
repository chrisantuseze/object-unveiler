import os
import random
import shutil
import argparse
from policy.models import Regressor, ResFCN

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import numpy as np
from trainer.aperture_dataset import ApertureDataset
from trainer.heightmap_dataset import HeightMapDataset

import utils.utils as utils

def train(args, model, optimizer, criterion, dataloaders, save_path, is_fcn=True):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for epoch in range(args.epochs):
        
        model.train()
        for batch in dataloaders['train']:

            if is_fcn:
                x = batch[0].to(device)
                target_mask = batch[1].to(device)
                rotations = batch[2]
                y = batch[3].to(device, dtype=torch.float)

                pred = model(x, target_mask, specific_rotation=rotations)
            else:
                x = batch[0].to(device, dtype=torch.float32)
                y = batch[1].to(device, dtype=torch.float32)

                pred = model(x)

            # compute loss in the whole scene
            loss = criterion(pred, y)
            loss = torch.sum(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        for phase in ['train', 'val']:
            for batch in dataloaders[phase]:

                if is_fcn:
                    x = batch[0].to(device)
                    target_mask = batch[1].to(device)
                    rotations = batch[2]
                    y = batch[3].to(device, dtype=torch.float)

                    pred = model(x, target_mask, specific_rotation=rotations)
                else:
                    x = batch[0].to(device, dtype=torch.float32)
                    y = batch[1].to(device, dtype=torch.float32)

                    pred = model(x)

                # compute loss
                loss = criterion(pred, y)
                loss = torch.sum(loss)
                epoch_loss[phase] += loss.detach().cpu().numpy()

        # save model
        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pt'))

        print('Epoch {}: training loss = {:.4f} '
              ', validation loss = {:.4f}'.format(epoch, epoch_loss['train'] / len(dataloaders['train']),
                                                  epoch_loss['val'] / len(dataloaders['val'])))



def train_fcn(args):
    save_path = 'save/fcn'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    transition_dirs = next(os.walk(args.dataset_dir))[1]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)
    train_ids = transition_dirs[:int(args.split_ratio * len(transition_dirs))]
    val_ids = transition_dirs[:int(args.split_ratio * len(transition_dirs))]

    train_dataset = HeightMapDataset(args.dataset_dirs, train_ids)
    val_dataset = HeightMapDataset(args.dataset_dir, val_ids)

    # note: the batch size is 1
    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size)

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    print('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    model = ResFCN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss(reduction='none')

    train(args, model, optimizer, criterion, data_loaders, save_path, is_fcn=True)

def train_regressor(args):
    save_path = 'save/fcn'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    transition_dirs = next(os.walk(args.dataset_dir))[1]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)
    train_ids = transition_dirs[:int(args.split_ratio * len(transition_dirs))]
    val_ids = transition_dirs[:int(args.split_ratio * len(transition_dirs))]

    train_dataset = ApertureDataset(args.dataset_dir, train_ids)
    val_dataset = ApertureDataset(args.dataset_dir, val_ids)

    # note: the batch size is 4
    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size)

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    print('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    model = Regressor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    train(args, model, optimizer, criterion, data_loaders, save_path, is_fcn=False)


