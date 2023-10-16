'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import os
import sys
# from data_processing.generate_dataset import DatasetCategory
# from data_processing.hierarchical_dataset import DeepFashionHierarchihcalDataset, HierarchicalBatchSampler
from datasets.hierarchical_OLIVES import OLIVES_HierarchihcalDataset, OLIVES_HierarchicalBatchSampler
from utils.utils_hierarchical import adjust_learning_rate, warmup_learning_rate, TwoCropTransform
from losses.losses import HMLC
# from network import resnet_modified
# from network.model import LinearClassifier, build_model
from config.config_clinical_hierarchical_con import parse_option
import tensorboard_logger as tb_logger
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist
# import time
import shutil
import builtins

from utils.utils_hierarchical import load_olives_hierarchical, set_model
from models.resnet_modified import LinearClassifier
from training_hierarchical_con.training_one_epoch import train_Combined
best_prec1 = 0
# torch.cuda.empty_cache()
def main():
    print("in main function")
    global args, best_prec1
    args = parse_option()
    # print(f"args.model:{args.model}")
    # exit()
    args.save_folder = './trained_model'
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    args.tb_folder = './tensorboard'
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    args.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_loss_{}_trial_{}'.\
        format('hmlc', 'dataset', args.model, args.learning_rate,
               args.lr_decay_rate, args.batch_size, args.loss, 5)
    if args.tag:
        args.model_name = args.model_name + '_tag_' + args.tag
    args.tb_folder = os.path.join(args.tb_folder, args.model_name)
    args.save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    # distributed training
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        print("Adopting distributed multi processing training")
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    

def main_worker(gpu, ngpus_per_node, args):
    print("GPU in main worker is {}".format(gpu))
    torch.cuda.empty_cache()
    args.gpu = gpu
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            print("In the process of multi processing with rank as {}".format(args.rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

                                
    # create model
    print("=> creating model '{}'".format(args.model))
    model, criterion = set_model(ngpus_per_node, args)
    args.classifier = LinearClassifier(name=args.model, num_classes=args.num_classes).cuda(args.gpu)
    set_parameter_requires_grad(model, args.feature_extract)
    # if args.model == "resnet50":
    #     set_parameter_requires_grad_for_resnet50(model, args.feature_extract)
    # else:
    #     set_parameter_requires_grad(model, args.feature_extract)

    optimizer = setup_optimizer(model, args.learning_rate,
                                   args.momentum, args.weight_decay,
                                   args.feature_extract)
    cudnn.benchmark = True
    print(f'args.data:{args.data}')
    # exit()
    args.data = "data_OLIVES/"
    dataloaders_dict, sampler = load_olives_hierarchical(args.data, args.train_listfile, args.class_map_file,args)

    train_sampler, val_sampler = sampler['train'], sampler['val']
    for epoch in range(1, args.epochs + 1):
        print('Epoch {}/{}'.format(epoch, args.epochs + 1))
        print('-' * 50)
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        loss = train_Combined(dataloaders_dict, model, criterion, optimizer, epoch, args, logger)
        output_file = args.save_folder + '/checkpoint_loss_{}_epoch_{:04d}_.pth.tar'.format(loss, epoch)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if epoch % args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False,
                    filename=output_file)


def setup_optimizer(model_ft, lr, momentum, weight_decay, feature_extract):
    # Send the model to GPU
    # model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    print(f"model_ft:{model_ft}")
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    print(f"params to update: {params_to_update}")
    print(f"lr:{lr} momentum:{momentum} weight_decay:{weight_decay}")
    optimizer_ft = torch.optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer_ft

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        # Select which params to finetune
        # for param in model.parameters():
        #     param.requires_grad = True
        
        for name, param in model.module.named_parameters():
            # print(name)
            if name.startswith('encoder.layer4') or name.startswith('body.layer4'):
                param.requires_grad = True
            elif name.startswith('encoder.layer3') or name.startswith('body.layer3') or name.startswith('layers.3'):
                param.requires_grad = True
            elif name.startswith('head'):
                param.requires_grad = True
            else:
                param.requires_grad = False

def set_parameter_requires_grad_for_resnet50(model, feature_extracting):
    if feature_extracting:
        # Select which params to finetune
        # for param in model.parameters():
        #     param.requires_grad = True
        
        for name, param in model.named_parameters():
            # print(name)
            if name.startswith('encoder.layer4') or name.startswith('body.layer4'):
                param.requires_grad = True
            elif name.startswith('encoder.layer3') or name.startswith('body.layer3') or name.startswith('layers.3'):
                param.requires_grad = True
            elif name.startswith('head'):
                print(f"head name:{name}")
                param.requires_grad = True
            else:
                param.requires_grad = False


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == "__main__":
    main()
