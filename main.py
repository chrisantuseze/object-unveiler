#!/usr/bin/env python3
import torch
import argparse
import yaml
from trainer.train_ae import train_ae, train_fcn_net, train_regressor
from trainer.train_sre import train_sre, train_sre_multi
from trainer.train_sre_rl import train_sre_rl
# from trainer.train import train_fcn_net
# from eval_agent_target import eval_agent
from eval_agent import eval_agent
import utils.logger as logging

# logging.init()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', default='sre-rl', type=str, help='')
    
    # args for eval_agent
    parser.add_argument('--ae_model', default='save/ae/ae_model_best.pt', type=str, help='')
    parser.add_argument('--sre_model', default='save/sre/sre_model_best.pt', type=str, help='')
    parser.add_argument('--sre_rl', default='save/sre_rl/sre_rl_model_best.pt', type=str, help='')
    parser.add_argument('--fcn_model', default='save/fcn/fcn_model_best.pt', type=str, help='')
    parser.add_argument('--reg_model', default='', type=str, help='')
    parser.add_argument('--seed', default=16, type=int, help='')
    parser.add_argument('--n_scenes', default=100, type=int, help='')
    parser.add_argument('--object_set', default='seen', type=str, help='')

    # args for trainer
    parser.add_argument('--dataset_dir', default='save/pc-ou-dataset', type=str, help='')
    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--lr', default=0.0001, type=float, help='')
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--split_ratio', default=0.9, type=float, help='')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')

    parser.add_argument('--sequence_length', default=1, type=int, help='')
    parser.add_argument('--patch_size', default=64, type=int, help='')
    parser.add_argument('--num_patches', default=10, type=int, help='This should not be less than the maximum possible number of objects in the scene, which from list Environment.nr_objects is 9')
    parser.add_argument('--step', default=500, type=int, help='')

    # args for act
    parser.add_argument('--chunk_size', default=3, action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    # args for RL training
    parser.add_argument('--config', default='config.yaml', type=str, help='Path to config file for environment params')
    parser.add_argument('--rl_gamma', default=0.99, type=float, help='Discount factor for RL')
    parser.add_argument('--rl_eps_clip', default=0.2, type=float, help='PPO clip parameter')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"You are using {args.device}")

    logging.info("The selected mode is:", args.mode, "and batch size is:", args.batch_size)

    if args.mode == 'sre':
        train_sre(args)

    elif args.mode == 'sre-rl':
        # Load environment parameters
        with open(args.config, 'r') as f:
            params = yaml.safe_load(f)
        train_sre_rl(args, params)

    elif args.mode == 'sre-multi':
        train_sre_multi(args)

    elif args.mode == 'ae':
        train_ae(args)

    elif args.mode == 'fcn':
        train_fcn_net(args)
        
    elif args.mode == 'reg':
        train_regressor(args)

    elif args.mode == 'eval':
        eval_agent(args)

    else:
        raise AssertionError
    
    logging.info("object-unveiler ended.")