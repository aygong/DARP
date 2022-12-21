import argparse

from dataset import *
from supervision import *
from evaluation import *


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Arguments of dataset.py
    parser.add_argument('--dataset', type=bool, default=False)
    parser.add_argument('--train_index', type=int, default=11)
    parser.add_argument('--num_train_subsets', type=int, default=20)
    parser.add_argument('--num_train_instances', type=int, default=500)
    parser.add_argument('--wait_time', type=int, default=7)

    # Arguments of supervision.py
    parser.add_argument('--supervision', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)

    # Arguments of evaluation.py
    parser.add_argument('--evaluation', type=bool, default=False)
    parser.add_argument('--test_index', type=int, default=8)
    parser.add_argument('--num_test_instances', type=int, default=10)

    # Arguments of transformer.py
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    if args.dataset:
        dataset(args)

    if args.supervision:
        supervision(args)

    if args.evaluation:
        evaluation(args)


if __name__ == '__main__':
    main()
