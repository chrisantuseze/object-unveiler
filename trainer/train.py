import os
import random
from policy.models_original import Regressor, ResFCN

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from datasets.aperture_dataset import ApertureDataset
from datasets.heightmap_dataset import HeightmapDataset

import utils.general_utils as general_utils
import utils.logger as logging

def train(args, model, optimizer, criterion, dataloaders, save_path, is_fcn=True):
    prefix = "fcn" if is_fcn else "reg"

    writer = SummaryWriter()

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(dataloaders['train']):
            if is_fcn:
                x = batch[0]
                rotations = batch[1]
                y = batch[2].to(args.device, dtype=torch.float32)
                pred = model(x, rotations)
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

            if step % args.step == 0:
                logging.info(f"Train Step [{step}/{len(dataloaders['train'])}]\t Loss: {loss.item()}\t")

        model.eval()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        for phase in ['train', 'val']:
            for step, batch in enumerate(dataloaders[phase]):
                if is_fcn:
                    x = batch[0].to(args.device, dtype=torch.float32)
                    rotations = batch[1]
                    y = batch[2].to(args.device, dtype=torch.float32)
                    pred = model(x, rotations)
                else:
                    x = batch[0].to(args.device, dtype=torch.float32)
                    y = batch[1].to(args.device, dtype=torch.float32)

                    pred = model(x)

                loss = criterion(pred, y)

                loss = torch.sum(loss)
                epoch_loss[phase] += loss.item()

                if step % args.step == 0:
                    logging.info(f"{phase} step [{step}/{len(dataloaders[phase])}]\t Loss: {loss.item()}")

        # save model
        # if epoch % 50 == 0:
        #     torch.save(model.state_dict(), os.path.join(save_path, f'{prefix}_model_' + str(epoch) + '.pt'))

        logging.info('Epoch {}: training loss = {:.6f} '
              ', validation loss = {:.6f}'.format(epoch, epoch_loss['train'] / len(dataloaders['train']),
                                                  epoch_loss['val'] / len(dataloaders['val'])))
        writer.add_scalar("log/train", epoch_loss['train'] / len(dataloaders['train']), epoch)
        writer.add_scalar("log/val", epoch_loss['val'] / len(dataloaders['val']), epoch)
        
    writer.close()
    torch.save(model.state_dict(), os.path.join(save_path,  f'{prefix}_model.pt'))



def train_fcn_net(args):
    save_path = 'save/fcn'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # transition_dirs = next(os.walk(args.dataset_dir))[1]
    transition_dirs = os.listdir(args.dataset_dir)
    
    for file_ in transition_dirs:
        if not file_.startswith("transition"):
            transition_dirs.remove(file_)

    # split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)

    # TODO: remember to remove this
    transition_dirs = transition_dirs[:4000]

    split_index = int(args.split_ratio * len(transition_dirs))
    train_ids = transition_dirs[:split_index]
    val_ids = transition_dirs[split_index:]

    # this ensures that the split is done properly without causing input mismatch error
    data_length = (len(train_ids)//args.batch_size) * args.batch_size
    train_ids = train_ids[:data_length]
    
    train_dataset = HeightmapDataset(args, train_ids)
    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)

    args.step = int(len(train_ids)/(4*args.batch_size))

    # data_length = (len(val_ids)//args.batch_size) * args.batch_size
    # val_ids = val_ids[:int(data_length)]

    val_dataset = HeightmapDataset(args, val_ids)
    data_loader_val = data.DataLoader(val_dataset, batch_size=2, num_workers=4, pin_memory=True)

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    logging.info('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    model = ResFCN().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss(reduction='none')

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

    model = Regressor().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    logging.info(model)

    train(args, model, optimizer, criterion, data_loaders, save_path, is_fcn=False)


