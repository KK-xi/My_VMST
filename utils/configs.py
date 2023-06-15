import argparse
import re
import os
from os.path import dirname

def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / evaluate dataset
    parser.add_argument("--train_dataset", default="./data/N-Caltech101/train/", required=True)
    parser.add_argument("--test_dataset", default="./data/N-Caltech101/test/", required=True)

    # logging options
    parser.add_argument("--log_dir", default="./results")
    parser.add_argument("--save_dir", default="./save_model")
    parser.add_argument("--arch_name", default="VMST-Net_N-Cal", required=True)

    # training/testing settings
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device_ids", type=list, default=[0, 1])
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--save_every_n_epochs", type=int, default=20)

    # base settings, varies with dataset
    parser.add_argument('--num_classes', type=int, default=101, required=True)
    parser.add_argument('--voxel_num', type=int, default=1024, required=True)

    # transformer settings
    parser.add_argument('--embed_dim', type=list, default=[32, 128, 256, 512])
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument("--in_chan", type=int, default=3)

    flags = parser.parse_args()

    assert os.path.isdir(dirname(flags.log_dir)), "Log directory root {dirname(flags.log_dir)} not found."

    print("----------------------------\n"
          "Starting training with \n"
          "num_epochs: {}\n"
          "batch_size: {}\n"
          "log_dir: {}\n"
          "dataset: {}\n"
          .format(flags.epochs, flags.batch_size, flags.log_dir, flags.train_dataset))

    return flags