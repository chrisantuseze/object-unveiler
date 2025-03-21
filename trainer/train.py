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

def train_fcn_net(args):
    save_path = 'save/fcn'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # transition_dirs = next(os.walk(args.dataset_dir))[1]
    transition_dirs = os.listdir(args.dataset_dir)
    
    for file_ in transition_dirs:
        if not file_.startswith("transition"):
            transition_dirs.remove(file_)

    # Split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)
    train_ids = transition_dirs[:int(args.split_ratio * len(transition_dirs))]
    val_ids = transition_dirs[int(args.split_ratio * len(transition_dirs)):]
    # train_ids = train_ids[::4]
    # val_ids = val_ids[::4]

    train_dataset = HeightmapDataset(args.dataset_dir, train_ids)
    val_dataset = HeightmapDataset(args.dataset_dir, val_ids)

    data_loader_train = data.DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size)
    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    print('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    model = ResFCN().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.SmoothL1Loss(reduction='none')
    criterion = nn.BCELoss(reduction='none')

    for epoch in range(args.epochs):
        model.train()
        for batch in data_loader_train:
            x = batch[0]
            rotations = batch[1]
            y = batch[2].to(args.device, dtype=torch.float32)

            pred = model(x, specific_rotation=rotations)

            # Compute loss in the whole scene
            loss = criterion(pred, y)
            loss = torch.sum(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        for phase in ['train', 'val']:
            for batch in data_loaders[phase]:
                x = batch[0]
                rotations = batch[1]
                y = batch[2].to(args.device, dtype=torch.float32)

                pred = model(x, specific_rotation=rotations)

                loss = criterion(pred, y)
                loss = torch.sum(loss)
                epoch_loss[phase] += loss.detach().cpu().numpy()

        # Save model
        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pt'))

        print('Epoch {}: training loss = {:.4f} '
              ', validation loss = {:.4f}'.format(epoch, epoch_loss['train'] / len(data_loaders['train']),
                                                  epoch_loss['val'] / len(data_loaders['val'])))


def train_regressor(args):
    save_path = 'save/reg'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # transition_dirs = next(os.walk(args.dataset_dir))[1]
    transition_dirs = os.listdir(args.dataset_dir)

    # Split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)
    train_ids = transition_dirs[:int(args.split_ratio * len(transition_dirs))]
    val_ids = transition_dirs[int(args.split_ratio * len(transition_dirs)):]

    train_dataset = ApertureDataset(args.dataset_dir, train_ids)
    val_dataset = ApertureDataset(args.dataset_dir, val_ids)

    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size)
    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    print('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    # model = Classifier(n_classes=3).to('cuda')
    model = Regressor().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.SmoothL1Loss()

    for epoch in range(args.epochs):
        model.train()
        for batch in data_loader_train:
            x = batch[0].to(args.device, dtype=torch.float32)
            y = batch[1].to(args.device, dtype=torch.float32)

            pred = model(x)

            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        # accuraccy = {'train': 0.0, 'val': 0.0}
        for phase in ['train', 'val']:
            for batch in data_loaders[phase]:
                x = batch[0].to(args.device, dtype=torch.float32)
                y = batch[1].to(args.device, dtype=torch.float32)

                pred = model(x)

                loss = criterion(pred, y)
                loss = torch.sum(loss)
                epoch_loss[phase] += loss.detach().cpu().numpy()

                # Compute classification accuracy
                # max_id = torch.argmax(pred, axis=1).detach().cpu().numpy()
                # for i in range(len(max_id)):
                #     if max_id[i] == y[i].detach().cpu().numpy():
                #         accuraccy[phase] += 1

        # Save model
        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pt'))

        print('Epoch:', epoch)
        print('loss: train/val {:.4f}/{:.4f}'.format(epoch_loss['train'] / len(data_loaders['train']),
                                                      epoch_loss['val'] / len(data_loaders['val'])))

        # print('accuracy: train/val {:.4f}/{:.4f}'.format(accuraccy['train'] / len(train_dataset),
        #                                                  accuraccy['val'] / len(val_dataset)))
        print('-----------------------')