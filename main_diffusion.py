import argparse
from diffusion.train_policy import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--enc_type', action='store', type=str, help='enc_type', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # # for ACT
    # parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    # parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    # parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    # parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    # parser.add_argument('--temporal_agg', action='store_true')

    main(vars(parser.parse_args()))