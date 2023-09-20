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
from policy.neural_network import ActionNet, ApertureNet

import utils.utils as utils
import utils.logger as logging

def train(args, model, optimizer, criterion, dataloaders, save_path, is_fcn=True):
    prefix = "fcn" if is_fcn else "reg"
    
    for epoch in range(args.epochs):
        logging.info('\nEpoch {}/{}'.format(epoch, args.epochs))
        logging.info('-' * 10)
        
        total_loss = 0
        model.train()
        logging.info("\nTrain mode...")
        for step, batch in enumerate(dataloaders['train']):
            if is_fcn:
                x = batch[0]
                rotations = batch[1]
                y = batch[2]
                pred = model(x, rotations)

                # x = batch[0].to(args.device)
                # rotations = batch[1].to(args.device)
                # y = batch[2].to(args.device)
                # pred = model(x, specific_rotation=rotations)

                y = utils.pad_label(args.sequence_length, y).to(args.device, dtype=torch.float32)
            else:
                x = batch[0].to(args.device, dtype=torch.float32)
                y = batch[1].to(args.device, dtype=torch.float32)

                pred = model(x)

            # compute loss in the whole scene
            loss = criterion(pred, y)
            loss = torch.sum(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 200 == 0:
                logging.info(f"Train Step [{step}/{len(dataloaders['train'])}]\t Loss: {loss.item()}")

        train_loss = total_loss / len(dataloaders['train'].dataset)
        logging.info(f'Train Loss: {train_loss}')



        model.eval()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        logging.info("\nEval mode...")
        for phase in ['val']:
            for step, batch in enumerate(dataloaders[phase]):
                if is_fcn:
                    x = batch[0]
                    rotations = batch[1]
                    y = batch[2]
                    pred = model(x, rotations)

                    # x = batch[0].to(args.device)
                    # rotations = batch[1].to(args.device)
                    # y = batch[2].to(args.device)
                    # pred = model(x, specific_rotation=rotations)

                    y = utils.pad_label(args.sequence_length, y).to(args.device, dtype=torch.float32)
                else:
                    x = batch[0].to(args.device, dtype=torch.float32)
                    y = batch[1].to(args.device, dtype=torch.float32)

                    pred = model(x)

                # compute loss
                loss = criterion(pred, y)
                loss = torch.sum(loss)
                epoch_loss[phase] += loss.item()

                if step % 200 == 0:
                    logging.info(f"{phase.capitalize()} Step [{step}/{len(dataloaders[phase])}]\t Loss: {loss.item()}")

            loss_ = total_loss / len(dataloaders[phase].dataset)
            logging.info(f'{phase.capitalize()} Loss: {loss_}')


        # save model
        # if epoch % 50 == 0:
        #     torch.save(model.state_dict(), os.path.join(save_path, f'{prefix}_model_' + str(epoch) + '.pt'))

        logging.info('Epoch {}: training loss = {}, validation loss = {}'
                     .format(epoch, train_loss, epoch_loss['val'] / len(dataloaders['val'])))
        
    torch.save(model.state_dict(), os.path.join(save_path,  f'{prefix}_model.pt'))



def train_fcn(args):
    save_path = 'save/fcn'

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

    train_dataset = HeightMapDataset(args, train_ids)
    val_dataset = HeightMapDataset(args, val_ids)

    # note: the batch size is 1
    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size)

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    logging.info('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    # model = ResFCN().to(args.device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.BCELoss(reduction='none')

    model = ActionNet(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # logging.info(model)

    train(args, model, optimizer, criterion, data_loaders, save_path, is_fcn=True)

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

    # model = Regressor().to(args.device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.SmoothL1Loss()

    model = ApertureNet(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    logging.info(model)

    train(args, model, optimizer, criterion, data_loaders, save_path, is_fcn=False)


