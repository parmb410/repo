# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import random
import numpy as np
import torch
import sys
import os
import argparse
import torchvision
import PIL


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Tee(object):
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.file = open(fname, mode)

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def print_args(args, print_list):
    s = "==========================================\n"
    for name, val in vars(args).items():
        if (not print_list) or (name in print_list):
            s += "{}: {}\n".format(name, val)
    s += "=========================================="
    return s


def act_param_init(args):
    parser = argparse.ArgumentParser(description="DIVERSIFY Utility Arguments")
    parser.add_argument('--model', type=str, default="diversify", help="Model name")
    parser.add_argument('--dataset', type=str, default="ucla", help="Dataset name")
    parser.add_argument('--data_dir', type=str, default="./data", help="Data directory")
    parser.add_argument('--data_file', type=str, default="", help="Data file prefix")
    parser.add_argument('--gpu_id', type=str, default="0", help="CUDA_VISIBLE_DEVICES setting")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini-batch size")
    parser.add_argument('--num_epoch', type=int, default=50, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loader workers")
    parser.add_argument('--feature_dim', type=int, default=128, help="Dimension of feature embeddings")
    parser.add_argument('--print_freq', type=int, default=10, help="Print frequency during training")
    parser.add_argument('--eval_freq', type=int, default=1, help="Evaluation frequency (epochs)")
    parser.add_argument('--save_freq', type=int, default=5, help="Model save frequency (epochs)")
    parser.add_argument('--old', action='store_true')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=str, default="cross_people")
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output', type=str, default="train_output")
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # ─── Added for Automated K Estimation ──────────────────────────────────
    parser.add_argument(
        '--k_min',
        type=int,
        default=2,
        help="Minimum K to try when running automated K estimation"
    )
    parser.add_argument(
        '--k_max',
        type=int,
        default=10,
        help="Maximum K to try when running automated K estimation"
    )
    # ──────────────────────────────────────────────────────────────────────

    args = parser.parse_args()
    args.steps_per_epoch = 10000000000
    args.data_dir = args.data_file + args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = act_param_init(args)
    return args


def get_args():
    return act_param_init(None)
