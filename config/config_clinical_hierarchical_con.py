import math
import argparse
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn as nn

def parse_option():
    parser = argparse.ArgumentParser(description='Training/finetuning on OLIVES Dataset')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset, the superset of train/val')
    parser.add_argument('--save_freq', type=int, default=2,
                        help='save frequency')
    parser.add_argument('--root_data', type=str, default='/mnt/HDD/Chau_Truong/Source_test/data/Datasets',
                        help='root of image')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train-listfile', default='', type=str,
                        help='training file with annotation')
    parser.add_argument('--val-listfile', default='', type=str,
                        help='validation file with annotation')
    parser.add_argument('--class-map-file', default='', type=str,
                        help='class mapping between str and int')
    parser.add_argument('--class-seen-file', default='', type=str,
                        help='seen classes text file. Used for seen/unseen split experiments.')
    parser.add_argument('--class-unseen-file', default='', type=str,
                        help='unseen classes text file. Used for seen/unseen split experiments.')

    parser.add_argument('--mode', default='train', type=str,
                        help='Train or val')
    parser.add_argument('--input-size', default=224, type=int,
                        help='input size')
    parser.add_argument('--scale-size', default=256, type=int,
                        help='scale size in validation')
    parser.add_argument('--crop-size', default=224, type=int,
                        help='crop size')
    parser.add_argument('--num-classes', type=int,
                        help='number of classes')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N', help='mini-batch size (default: 512)')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                        help='use pre-trained model')
    parser.add_argument('--feature-extract', action='store_false',
                        help='When flase, finetune the whole model; else only update the reshaped layer para')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    #other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--loss', type=str, default='hmce',
                        help='loss type', choices=['hmc', 'hce', 'hmce'])
    parser.add_argument('--tag', type=str, default='',
                        help='tag for model name')
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    # warm-up for large-batch training,
    if args.batch_size >= 256:
        args.warm = True
    if args.warm:
        args.model_name = '{}_warm'.format(args.model)
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    return args