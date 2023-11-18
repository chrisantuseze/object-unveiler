import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data

from datasets.occlusion_dataset import OcclusionDataset
from policy.occlusion_model import VisionTransformer
import utils.logger as logging


def train_vit(args):
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

    split_index = int(args.split_ratio * len(transition_dirs))
    train_ids = transition_dirs[:split_index]
    val_ids = transition_dirs[split_index:]

    # this ensures that the split is done properly without causing input mismatch error
    data_length = (len(train_ids)//args.batch_size) * args.batch_size
    train_ids = train_ids[:data_length]

    data_length = (len(val_ids)//args.batch_size) * args.batch_size
    val_ids = val_ids[:data_length]
    
    train_dataset = OcclusionDataset(args, train_ids)
    data_loader_train = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True, shuffle=True)

    val_dataset = OcclusionDataset(args, val_ids)
    data_loader_val = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    args.step = int(len(train_ids)/(4*args.batch_size))

    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    logging.info('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    model = VisionTransformer(args=args).to(args.device)
    model = model.float()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(data_loader_train):
            scene_masks = batch[0].to(args.device)
            target_mask = batch[1].to(args.device)
            label = batch[2].to(args.device, dtype=torch.float)

            pred = model(scene_masks, target_mask)
            pred = pred.float()
            # print(pred)
            # # print("\n")
            # print(label)

            # Compute loss in the whole scene
            loss = criterion(pred, label)
            loss = torch.sum(loss)

            # logging.info(f"train step [{step}/{len(data_loader_train)}]\t Loss: {loss.detach().cpu().numpy()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        for phase in ['train', 'val']:
            for step, batch in enumerate(data_loaders[phase]):
                scene_masks = batch[0].to(args.device)
                target_mask = batch[1].to(args.device)
                label = batch[2].to(args.device, dtype=torch.float)

                pred = model(scene_masks, target_mask)
                loss = criterion(pred, label)

                loss = torch.sum(loss)
                epoch_loss[phase] += loss.detach().cpu().numpy()

                if step % args.step == 0:
                    logging.info(f"{phase} step [{step}/{len(data_loaders[phase])}]\t Loss: {loss.detach().cpu().numpy()}")

        logging.info('Epoch {}: training loss = {:.6f} '
              ', validation loss = {:.6f}'.format(epoch, epoch_loss['train'] / len(data_loaders['train']),
                                                  epoch_loss['val'] / len(data_loaders['val'])))

    torch.save(model.state_dict(), os.path.join(save_path,  f'vit_model.pt'))