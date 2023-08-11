import argparse
from trainer.train import train_fcn, train_regressor
from eval_agent import eval_agent
import utils.utils as utils

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', default='fcn', type=str, help='')
    
    # args for eval_agent
    parser.add_argument('--fcn_model', default='', type=str, help='')
    parser.add_argument('--reg_model', default='', type=str, help='')
    parser.add_argument('--seed', default=6, type=int, help='')
    parser.add_argument('--n_scenes', default=100, type=int, help='')
    parser.add_argument('--object_set', default='seen', type=str, help='')

    # args for trainer
    parser.add_argument('--dataset_dir', default='logs/ppg-dataset', type=str, help='')
    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--lr', default=0.0001, type=float, help='')
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--split_ratio', default=0.9, type=float, help='')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.create_dirs()

    print("The selected mode is: ", args.mode)

    if args.mode == 'fcn':
        train_fcn(args)
        
    elif args.mode == 'reg':
        train_regressor(args)
    
    elif args.mode == 'eval':
        eval_agent(args)

    else:
        raise AssertionError