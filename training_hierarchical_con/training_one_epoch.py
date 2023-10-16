import torch
import time
from utils.utils_hierarchical import AverageMeter, ProgressMeter
from utils.utils_hierarchical import warmup_learning_rate
import sys

def train_Combined(dataloaders, model, criterion, optimizer, epoch, args, logger):
    """
    one epoch training
    """

    classifier = args.classifier
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    end = time.time()

    # Each epoch has a training and/or validation phase
    for phase in ['train']:
        print(len(dataloaders[phase]))
        # exit()
        if phase == 'train':
            # print(phase)
            progress = ProgressMeter(len(dataloaders['train']),
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode
        classifier.eval()

        # Iterate over data.
        for idx, (images, labels) in enumerate(dataloaders[phase]):
            data_time.update(time.time() - end)
            labels = labels.squeeze()
            images = torch.cat([images[0].squeeze(), images[1].squeeze()], dim=0)
            images = images.cuda(non_blocking=True)
            labels = labels.squeeze().cuda(non_blocking=True)
            bsz = labels.shape[0] #batch size
            if phase == 'train':
                warmup_learning_rate(args, epoch, idx, len(dataloaders[phase]), optimizer)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                features = model(images)
                if not isinstance(features, torch.Tensor):
                    features = torch.cat(features[0], dim=0)

                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = criterion(features, labels)
                losses.update(loss.item(), bsz)
                # backward + optimize only if in training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print(idx)
            # print(args.print_freq)
            # exit()

            if idx % args.print_freq == 0:
                progress.display(idx)
            
                sys.stdout.flush()
        logger.log_value('loss', losses.avg, epoch)
    return losses.avg