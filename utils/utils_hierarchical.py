from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import os
from models import resnet_modified
from losses.losses import HMLC
from datasets.hierarchical_OLIVES import OLIVES_HierarchihcalDataset, OLIVES_HierarchicalBatchSampler

def set_loader_baseline(opt):
    # construct data loader
    if opt.dataset == 'OLIVES' or opt.dataset == 'RECOVERY':
        mean = (.1706)
        std = (.2112)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),

        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])


    if opt.dataset =='OLIVES':
        csv_path_train = opt.train_csv_path
        csv_path_test = opt.test_csv_path
        data_path_train = opt.train_image_path
        data_path_test = opt.test_image_path
        train_dataset = OLIVES(csv_path_train,data_path_train,transforms = train_transform)
        test_dataset = RECOVERY(csv_path_test,data_path_test,transforms = val_transform)
    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True,drop_last=False)

    return train_loader, test_loader




def load_olives_hierarchical(root_dir, train_list_file, class_map_file, opt):
    transform_rgb = TransformRGB()
    mean = (.1706)
    std = (.2112)
    
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),

        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transform_rgb,
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transform_rgb,
        normalize,
    ])
    
    
    print(f"root_dir:{root_dir}")
    print(f"os.path.join(root_dir, train_list_file):{os.path.join(root_dir, train_list_file)}")
    # exit()
    train_dataset = OLIVES_HierarchihcalDataset(os.path.join(root_dir, train_list_file),
                                                os.path.join(root_dir, class_map_file),
                                                opt,
                                                transform=TwoCropTransform(train_transform))

    val_dataset = OLIVES_HierarchihcalDataset(os.path.join(root_dir, train_list_file),
                                              os.path.join(root_dir, class_map_file),
                                              opt,
                                              transform=TwoCropTransform(val_transform))
    print('LENGTH TRAIN', len(train_dataset))
    image_datasets = {'train': train_dataset,
                      'val': val_dataset}
    train_sampler = OLIVES_HierarchicalBatchSampler(batch_size=opt.batch_size,
                                       drop_last=False,
                                       dataset=train_dataset)
    val_sampler = OLIVES_HierarchicalBatchSampler(batch_size=opt.batch_size,
                                           drop_last=False,
                                           dataset=val_dataset)
    sampler = {'train': train_sampler,
               'val': val_sampler}
    print(opt.workers, "workers")
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], sampler=sampler[x],
                                       num_workers=opt.workers, batch_size=1,
                                       pin_memory=True)
        for x in ['train', 'val']}
    return dataloaders_dict, sampler



def set_model(ngpus_per_node, args):
    model = resnet_modified.MyResNet(name='resnet50')
    criterion = HMLC(temperature=args.temp, loss_type=args.loss, layer_penalty=torch.exp)

    # This part is to load a pretrained model
    # ckpt = torch.load(args.ckpt, map_location='cpu')
    path = "/content/drive/MyDrive/Hierarchical_ConLoss_OCT_Labels/pretrained_models/resnet50-19c8e357.pth"
    ckpt = torch.load(path, map_location='cpu')
    # state_dict = ckpt['state_dict']
    state_dict = ckpt
    model_dict = model.state_dict()
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if not k.startswith('fc'):
            new_key = 'encoder.' + k
            new_state_dict[new_key] = v

    state_dict = new_state_dict
    model_dict.update(state_dict)

    # for key, value in model_dict.items():
    #     print(f"Key: {key}")
    #     print(f"Value: {value}")
    #     print("--------------------")

    # for k, v in new_state_dict.items():
    #     print(f"key in new state dict:{k}")
    #     print(f"value in new state dict:{v}")
    print(f"args.distributed:{args.distributed}")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        print("GPU setting", args.gpu)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            print("Updated batch size is {}".format(args.batch_size))
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # There is memory issue in data loader
            # args.workers = 0
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
            print("check model data parallel")
            for key, value in model_dict.items():
                print(f"Key: {key}")
                # print(f"Value: {value}")
                print("--------------------")
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print('Loading state dict from ckpt')
            model.load_state_dict(state_dict)
    elif args.gpu is not None:
        print("not using args distributed!")
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    criterion = criterion.cuda(args.gpu)

    return model, criterion




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def set_optimizer(opt, model):
    # backbone = model.backbone
    params_to_update = model.parameters()
    print("Params to learn:")
    if opt.feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

class TransformRGB(object):
    def __call__(self, img):

        if isinstance(img, torch.Tensor):
            img = transforms.functional.to_pil_image(img)
        img_rgb = img.convert("RGB")


        # Chuyển đổi lại thành tensor PyTorch
        img_rgb_tensor = transforms.functional.to_tensor(img_rgb)

        return img_rgb_tensor
