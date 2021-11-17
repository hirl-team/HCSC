import argparse
import os
import time
import logging
from logging import getLogger
import urllib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils.options import parse_semisup_args
from utils.utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    accuracy,
)

logger = getLogger()

def main():
    global args, best_acc
    args = parse_semisup_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    if args.rank==0:
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val"
    )

    # build data
    train_data_path = os.path.join(args.data_path, "train")
    train_dataset = datasets.ImageFolder(train_data_path)
    # take either 1% or 10% of images
    subset_file = "{}percent.txt".format(args.labels_perc)
    with open(subset_file, "r") as f:
        list_imgs = f.readlines()
        list_imgs = [x.split("\n")[0] for x in list_imgs]
    
    train_dataset.samples = [(
        os.path.join(train_data_path, li.split('_')[0], li),
        train_dataset.class_to_idx[li.split('_')[0]]
    ) for li in list_imgs]
    val_dataset = datasets.ImageFolder(os.path.join(args.data_path, "val"))
    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )
    train_dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = models.__dict__[args.arch](num_classes=1000)

    # convert batch norm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda:" + str(args.gpu))
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}
        
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training from random weights")

    # model to gpu
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        find_unused_parameters=True,
    )

    # set optimizer
    trunk_parameters = []
    head_parameters = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            head_parameters.append(param)
        else:
            trunk_parameters.append(param)
    optimizer = torch.optim.SGD(
        [{'params': trunk_parameters},
         {'params': head_parameters, 'lr': args.lr_last_layer}],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0,
    )
    # set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.decay_epochs, gamma=args.gamma
    )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": (0., 0.)}
    restart_from_checkpoint(
        os.path.join(args.exp_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set samplers
        train_loader.sampler.set_epoch(epoch)

        scores = train(model, optimizer, train_loader, epoch)
        scores_val = validate_network(val_loader, model)
        training_stats.update(scores + scores_val)

        scheduler.step()

        # save checkpoint
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.exp_dir, "checkpoint.pth.tar"))
    logger.info("Fine-tuning with {}% of labels completed.\n"
                "Test accuracies: top-1 {acc1:.1f}, top-5 {acc5:.1f}".format(
                args.labels_perc, acc1=best_acc[0], acc5=best_acc[1]))


def train(model, optimizer, loader, epoch):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter("time", ":.2f")
    data_time = AverageMeter("data time", ":.2f")

    # training statistics
    top1 = AverageMeter("top1", ":.3f")
    top5 = AverageMeter("top5", ":.3f")
    losses = AverageMeter("loss", ":.3e")
    end = time.perf_counter()

    model.train()
    criterion = nn.CrossEntropyLoss().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        output = model(inp)

        # compute cross entropy loss
        loss = criterion(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # update stats
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(acc1[0], inp.size(0))
        top5.update(acc5[0], inp.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if args.rank == 0 and iter_epoch % 50 == 0:
            logger.info(
                "Epoch[{0}] - Iter: [{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec {top1.val:.3f} ({top1.avg:.3f})\t"
                "LR trunk {lr}\t"
                "LR head {lr_W}".format(
                    epoch,
                    iter_epoch,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    lr=optimizer.param_groups[0]["lr"],
                    lr_W=optimizer.param_groups[1]["lr"],
                )
            )
    return epoch, losses.avg, top1.avg.item(), top5.avg.item()

def validate_network(val_loader, model):
    batch_time = AverageMeter("time", ":.2f")
    losses = AverageMeter("loss", ":.3e")
    top1 = AverageMeter("top1", ":.2f")
    top5 = AverageMeter("top5", ":.2f")
    global best_acc

    # switch to evaluate mode
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target) in enumerate(val_loader):

            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(inp)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            top5.update(acc5[0], inp.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    if top1.avg.item() > best_acc[0]:
        best_acc = (top1.avg.item(), top5.avg.item())

    if args.rank == 0:
        logger.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=top1, acc=best_acc[0]))

    return losses.avg, top1.avg.item(), top5.avg.item()


if __name__ == "__main__":
    main()
