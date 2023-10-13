import os
import random
from policy.models_new import Regressor, ResFCN

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from trainer.aperture_dataset import ApertureDataset
from trainer.heightmap_dataset import HeightMapDataset
from policy.action_net_new import ActionNet

import utils.utils as utils
import utils.logger as logging


def train_fcn_net(args):
    save_path = 'save/fcn'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # transition_dirs = next(os.walk(args.dataset_dir))[1]
    transition_dirs = os.listdir(args.dataset_dir)
    
    for file_ in transition_dirs:
        if not file_.startswith("episode"):
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
    
    train_dataset = HeightMapDataset(args, train_ids)
    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)

    args.step = int(len(train_ids)/(4*args.batch_size))

    val_dataset = HeightMapDataset(args, val_ids)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    logging.info('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    model = ActionNet(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.SmoothL1Loss(reduction='none')
    criterion = nn.BCELoss(reduction='none')

    # model = ResFCN().to(args.device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # # criterion = nn.SmoothL1Loss(reduction='none')
    # criterion = nn.BCELoss(reduction='none')

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(data_loader_train):
            x = batch[0]
            rotations = batch[1]
            y = batch[2]
            y = utils.pad_label(args.sequence_length, y).to(args.device, dtype=torch.float32)

            # x = batch[0].to(args.device)
            # rotations = batch[1]
            # y = batch[2].to(args.device, dtype=torch.float)

            pred = model(x, rotations)

            # Compute loss in the whole scene
            loss = criterion(pred, y)
            loss = torch.sum(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        for phase in ['train', 'val']:
            for step, batch in enumerate(data_loaders[phase]):
                x = batch[0]
                rotations = batch[1]
                y = batch[2]
                y = utils.pad_label(args.sequence_length, y).to(args.device, dtype=torch.float32)

                # x = batch[0].to(args.device)
                # rotations = batch[1]
                # y = batch[2].to(args.device, dtype=torch.float)

                pred = model(x, rotations)

                loss = criterion(pred, y)
                loss = torch.sum(loss)
                epoch_loss[phase] += loss.detach().cpu().numpy()

                # if step % args.step == 0:
                logging.info(f"{phase.capitalize} Step [{step}/{len(data_loaders[phase])}]\t Loss: {loss.detach().cpu().numpy()}")

        # Save model
        # if epoch % 1 == 0:
        #     torch.save(model.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pt'))

        logging.info('Epoch {}: training loss = {:.4f} '
              ', validation loss = {:.4f}'.format(epoch, epoch_loss['train'] / len(data_loaders['train']),
                                                  epoch_loss['val'] / len(data_loaders['val'])))

    torch.save(model.state_dict(), os.path.join(save_path,  f'fcn_model.pt'))

def train(args, model, optimizer, scheduler, criterion, dataloaders, save_path, is_fcn=True):
    prefix = "fcn" if is_fcn else "reg"

    writer = SummaryWriter()

    
    for epoch in range(args.epochs):
        logging.info('\nEpoch {}/{}'.format(epoch, args.epochs))
        logging.info('-' * 10)
        
        total_loss = 0
        model.train()
        logging.info("\nTrain mode...")
        for step, batch in enumerate(dataloaders['train']):
            optimizer.zero_grad()
            if is_fcn:
                x = batch[0]
                rotations = batch[1]
                y = batch[2]
                pred = model(x, rotations)

                y = utils.pad_label(args.sequence_length, y).to(args.device, dtype=torch.float32)

                # print("y.shape:", y.shape)  #torch.Size([4, 1, 1, 224, 224])
            else:
                x = batch[0].to(args.device, dtype=torch.float32)
                y = batch[1].to(args.device, dtype=torch.float32)

                pred = model(x)

            # compute loss in the whole scene
            loss = criterion(pred, y)
            loss = torch.sum(loss)
            loss.backward()
            optimizer.step()

            loss_item = loss.item() * len(x)
            total_loss += loss_item
            if step % args.step == 0:
                logging.info(f"Train Step [{step}/{len(dataloaders['train'])}]\t Loss: {loss_item}\t Lr: {optimizer.param_groups[0]['lr']}")

        train_loss = total_loss / len(dataloaders['train'])

        total_loss = 0
        model.eval()
        logging.info("\nEval mode...")
        for step, batch in enumerate(dataloaders['val']):
            if is_fcn:
                x = batch[0]
                rotations = batch[1]
                y = batch[2]
                pred = model(x, rotations)

                y = utils.pad_label(args.sequence_length, y).to(args.device, dtype=torch.float32)
            else:
                x = batch[0].to(args.device, dtype=torch.float32)
                y = batch[1].to(args.device, dtype=torch.float32)

                pred = model(x)

            # compute loss
            loss = criterion(pred, y)
            loss = torch.sum(loss)
            loss_item = loss.item() * len(x)
            total_loss += loss_item

            if step % args.step == 0:
                logging.info(f"Val Step [{step}/{len(dataloaders['val'])}]\t Loss: {loss_item}")

        val_loss = total_loss / len(dataloaders['val'])


        # save model
        # if epoch % 50 == 0:
        #     torch.save(model.state_dict(), os.path.join(save_path, f'{prefix}_model_' + str(epoch) + '.pt'))

        # Decay Learning Rate
        scheduler.step()

        logging.info('Epoch {}: training loss = {}, validation loss = {}'.format(epoch, train_loss, val_loss))
        writer.add_scalar("training loss x epoch", train_loss, epoch)
        writer.add_scalar("validation loss x epoch", val_loss, epoch)
        
    writer.close()
    torch.save(model.state_dict(), os.path.join(save_path,  f'{prefix}_model.pt'))



def train_fcn(args):
    save_path = 'save/fcn'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # transition_dirs = next(os.walk(args.dataset_dir))[1]
    transition_dirs = os.listdir(args.dataset_dir)
    
    for file_ in transition_dirs:
        if not file_.startswith("episode"):
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
    
    train_dataset = HeightMapDataset(args, train_ids)
    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)

    args.step = int(len(train_ids)/(4*args.batch_size))

    # data_length = (len(val_ids)//args.batch_size) * args.batch_size
    # val_ids = val_ids[:int(data_length)]

    val_dataset = HeightMapDataset(args, val_ids)
    data_loader_val = data.DataLoader(val_dataset, batch_size=2, num_workers=4, pin_memory=True)

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    logging.info('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    # model = ResFCN().to(args.device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.BCELoss(reduction='none')

    model = ActionNet(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80])
    # criterion = nn.MSELoss()
    criterion = nn.BCELoss(reduction='none') #nn.BCEWithLogitsLoss()

    # logging.info(model)

    train(args, model, optimizer, scheduler, criterion, data_loaders, save_path, is_fcn=True)

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


