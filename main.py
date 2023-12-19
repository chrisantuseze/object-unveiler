#!/usr/bin/env python3
import torch
import argparse
from trainer.train_new import train_fcn_net, train_regressor
from trainer.train_occlusion import train_vit
from eval_agent import eval_agent
import utils.general_utils as general_utils
import utils.logger as logging

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', default='fcn', type=str, help='')
    
    # args for eval_agent
    parser.add_argument('--fcn_model', default='', type=str, help='')
    parser.add_argument('--reg_model', default='', type=str, help='')
    parser.add_argument('--vit_model', default='save/fcn/vit_model.pt', type=str, help='')
    parser.add_argument('--seed', default=6, type=int, help='')
    parser.add_argument('--n_scenes', default=100, type=int, help='')
    parser.add_argument('--object_set', default='seen', type=str, help='')

    # args for trainer
    parser.add_argument('--dataset_dir', default='save/ppg-dataset', type=str, help='')
    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--lr', default=0.0001, type=float, help='')
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--split_ratio', default=0.9, type=float, help='')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')

    parser.add_argument('--sequence_length', default=4, type=int, help='')
    parser.add_argument('--patch_size', default=64, type=int, help='')
    parser.add_argument('--num_patches', default=12, type=int, help='This should not be less than the maximum possible number of objects in the scene, which from list Environment.nr_objects is 8')
    parser.add_argument('--step', default=200, type=int, help='')

    parser.add_argument('--log', default=1, type=int, help='1 for logging, and 0 for no logging')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.init(args.log)

    general_utils.create_dirs()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"You are using {args.device}")

    logging.info("The selected mode is:", args.mode, "and batch size is:", args.batch_size)

    if args.mode == 'fcn':
        train_fcn_net(args)
        
    elif args.mode == 'reg':
        train_regressor(args)

    elif args.mode == 'vit':
        train_vit(args)
    
    elif args.mode == 'eval':
        eval_agent(args)

    else:
        raise AssertionError
    
    logging.info("object-unveiler ended.")