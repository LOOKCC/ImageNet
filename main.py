import argparse
import os
import random
import shutil
import time
import warnings
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from dataset import Dataset
from utils import save_checkpoint, AverageMeter, ProgressMeter_train, ProgressMeter_test, adjust_learning_rate, accuracy
import net as models
import yaml


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-c', '--config', default='config.yaml', type=str, help='path to config file.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save', type=str, help='save_dir')


device = torch.device("cuda:0")


def main():
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config.update(vars(args))
    args = Namespace(**config)
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    print('Saveing configs to ' + args.save + '/' +'used_config...')
    with open(args.save + '/' +'used_config', 'w') as f:
        yaml.dump(config, f)
    # parpaer tensorboard
    writer = SummaryWriter(args.save)
    if not args.resume:
        fout = open(os.path.join(args.save, args.log), 'w')
        fout.write('iter train_loss train_acc test_loss test_acc lr\n')
        fout.close()

    model = models.__dict__[args.arch](num_classes=args.num_classes)
    if args.fix_fc:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)
    
    num = torch.cuda.device_count()
    print('device:', num)
    model = torch.nn.DataParallel(model, device_ids=args.device_ids)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    start_iter = 0
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            start_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    train_dataset = Dataset(
        args.train_set,
        transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        Dataset(args.test_set, 
            transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ]), train=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        acc1, loss, result = validate(val_loader, model, criterion, args)
        with open(os.path.join(args.save, 'evaluate.txt'), 'w') as f:
            f.writelines(result)
        return
    
    for epoch in range(start_epoch, args.max_epoch):
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        start_iter = train(train_loader, val_loader, model, criterion, optimizer, epoch, args, writer, start_iter)
        if start_iter > args.max_iteration:
            print('Training over at', args.max_epoch, 'iteration.')
            break
    print('Training over at', epoch, 'iteration.')

def train(train_loader, val_loader, model, criterion, optimizer, epoch, args, writer, start_iter):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter_train(
        len(train_loader), args.max_iteration,
        [batch_time, data_time, losses, top1, top2],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)
        # compute output
        output = model(images)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc2 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top2.update(acc2[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, start_iter + i)
        if start_iter + i > args.max_iteration:
            return start_iter + i
            
        if i > 0 and (start_iter + i)%args.validation_freq == 0:
            acc1, loss, result = validate(val_loader, model, criterion, args)
            writer.add_scalar('Test/Loss', loss, start_iter + i)
            writer.add_scalar('Test/ACC', acc1, start_iter + i)
            writer.flush()
            for param_group in optimizer.param_groups:
                now_lr = param_group['lr'] 
                break
            with open(os.path.join(args.save, args.log), 'a+') as f:
                f.write(str(start_iter + i) + ' ' + str(losses.avg) +  ' ' + str(float(top1.avg)) + ' ' + str(loss) + ' ' + str(float(acc1)) + ' ' + str(now_lr) +'\n')

            with open(os.path.join(args.save, 'result_' + str(start_iter + i) + '.txt'), 'w') as f:
                f.writelines(result)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'iter':start_iter
            }, os.path.join(args.save, 'ckpt_'), start_iter + i)
            
            writer.add_scalar('Train/Loss', losses.avg, start_iter + i)
            writer.add_scalar('Train/ACC', top1.avg, start_iter + i)
            writer.flush()
    

    return start_iter+len(train_loader)

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter_test(
        len(val_loader),
        [batch_time, losses, top1, top2],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    result = []

    with torch.no_grad():
        end = time.time()
        for i, (image_path, images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            softmax = F.softmax(output, dim=1)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top2.update(acc2[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            for j in range(len(image_path)):
                result.append(image_path[j] + ' ' +str(int(target[j].item())) + ' ' + str(float(softmax[j][1].item())) + '\n')

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@2 {top5.avg:.3f} loss {losses.avg:.3f}'
              .format(top1=top1, top5=top2, losses=losses))

    return top1.avg, losses.avg, result

if __name__ == '__main__':
    main()
