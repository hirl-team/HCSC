import argparse
import builtins
import logging
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from datetime import timedelta

import math
import os, sys
import numpy as np
import torch
import hcsc.loader
import hcsc
from hcsc.hcsc import HCSC

from hcsc.logger import EasyLogger
from utils.options import parse_args_main
from utils.utils import init_distributed_mode


def build_model(args, logger):
    backbone = models.__dict__[args.arch]
    model = HCSC(
        backbone,
        args.dim,
        args.queue_length,
        args.m,
        args.T,
        args.mlp,
        args.multi_crop,
        args.instance_selection,
        args.proto_selection,
        args.selection_on_local,
        logger)
    return model

def build_dataloaders(args):
    return getattr(hcsc.loader, args.dataset)(args)


def build_optimizer(args, model):
    total_batch_size = args.batch_size * dist.get_world_size()
    ## scale up the batch size
    args.lr = args.lr * total_batch_size / 256
    print("total batch size is {}, lr is scaled up to {}".format(total_batch_size, args.lr))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    return optimizer

def main():
    args = parse_args_main()
    init_distributed_mode(args)
    

    args.num_cluster = args.num_cluster.split(',')
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir, exist_ok=True)
    
    logger = EasyLogger(args.exp_dir, 0, args.rank)
    # create dataset
    train_loader, eval_loader, train_dataset, eval_dataset, train_sampler = build_dataloaders(args)
    dist.barrier()
    args.dataset_size = len(train_dataset)
    # create model
    if args.rank == 0:
        print("=> creating model '{}'".format(args.arch))
    model = build_model(args, logger)
    model = model.to(args.device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = build_optimizer(args, model)
    scheduler = adjust_learning_rate
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print("No optimizer state!")
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    for epoch in range(args.start_epoch, args.epochs):
        logger.set_epoch(epoch)
        if hasattr(model.module, "set_epoch"):
            model.module.set_epoch(epoch)
        cluster_result = None
        if epoch >= args.warmup_epoch:
            # compute momentum features for center-cropped images
            features = compute_features(eval_loader, model, args)         
            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[], 'cluster2cluster': [], 'logits': []}
            for i, num_cluster in enumerate(args.num_cluster):
                cluster_result['im2cluster'].append(torch.zeros(len(eval_dataset),dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),args.dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())
                if i < (len(args.num_cluster) - 1):
                    cluster_result['cluster2cluster'].append(torch.zeros(int(num_cluster), dtype=torch.long).cuda())
                    cluster_result['logits'].append(torch.zeros([int(num_cluster), int(args.num_cluster[i+1])]).cuda())
            if dist.get_rank() == 0:
                features[torch.norm(features,dim=1)>1.5] /= 2 
                features = features.numpy()
                cluster_result = run_hkmeans(features,args)  
                # save the clustering result
                try:
                    torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))  
                except:
                    pass
            dist.barrier()  
            # broadcast clustering result
            for k, data_list in cluster_result.items():
                for data_tensor in data_list:                
                    dist.broadcast(data_tensor, 0, async_op=False)   
            
        train_sampler.set_epoch(epoch)
        scheduler(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, cluster_result)

        if (epoch+1)%5==0 and dist.get_rank()==0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir,epoch))

def train(train_loader, model, criterion, optimizer, epoch, args, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = dict()
    acc_inst = dict()
    losses['InsLoss_sum'] = AverageMeter('InsLoss_sum', ':.4e')
    acc_inst['Acc@Inst_avg'] = AverageMeter('Acc@Inst_avg', ':6.2f')
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')
    buffer_meter = dict()
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst, acc_proto, buffer_meter],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    for i, (images, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = [image.to(args.device) for image in images]
                
        # compute output
        output, target, output_proto, target_proto, local_logits, local_labels, local_proto_logits, local_proto_targets = model(images, 
                                                                                cluster_result=cluster_result, 
                                                                                index=index)

        # InfoNCE loss
        loss = 0.
        if isinstance(target, list):
            loss_total = 0.
            for k, (out, tar) in enumerate(zip(output, target)):
                loss = criterion(out, tar)
                loss_total += loss
                if f'InsLoss_{k}' not in buffer_meter:
                    buffer_meter[f'InsLoss_{k}'] = AverageMeter(f'InsLoss_{k}', ':.4e')
                buffer_meter[f'InsLoss_{k}'].update(loss.item(), images[0].size(0))
                acc = accuracy(out, tar)[0] 
                if f'Acc@Inst{k}' not in buffer_meter:
                    buffer_meter[f'Acc@Inst{k}'] = AverageMeter(f'Acc@Inst{k}', ":6.2f")
                buffer_meter[f'Acc@Inst{k}'].update(acc[0], images[0].size(0))
                losses['InsLoss_sum'].update(loss.item(), images[0].size(0))
                acc_inst['Acc@Inst_avg'].update(acc[0], images[0].size(0))
                loss = loss_total
        else:
            loss = criterion(output, target)  
            losses['InsLoss_sum'].update(loss.item(), images[0].size(0))
            acc = accuracy(output, target)[0] 
            acc_inst['Acc@Inst_avg'].update(acc[0], images[0].size(0))
        
        # InfoNCE Loss on local views with multi-crop
        if local_logits is not None:
            # print("local nce")
            loss_local = 0
            for vid, (local_logit, local_target) in enumerate(zip(local_logits, local_labels)):
                loss_local += criterion(local_logit, local_target)
                acc_local = accuracy(local_logit, local_target)[0]
                if f"acc_local{vid}" not in buffer_meter:
                    buffer_meter[f"acc_local{vid}"] = AverageMeter(f"acc_local{vid}", ":6.4f")
                
                buffer_meter[f"acc_local{vid}"].update(acc_local[0], images[0].size(0))
            loss += loss_local

        # HProtoNCE loss
        if output_proto is not None:
            loss_proto = 0
            for proto_out,proto_target in zip(output_proto, target_proto):
                loss_proto += criterion(proto_out, proto_target)  
                accp = accuracy(proto_out, proto_target)[0] 
                acc_proto.update(accp[0], images[0].size(0))
                
            # average loss across all sets of prototypes
            loss_proto /= len(args.num_cluster) 
            loss += loss_proto

        # HProtoNCE Loss on local views
        if local_proto_logits is not None:
            loss_local_proto = 0
            for vid, (proto_logits, proto_targets) in enumerate(zip(local_proto_logits, local_proto_targets)):
                for logit, target in zip(proto_logits, proto_targets):
                    loss_local_proto += criterion(logit, target)
                    accp = accuracy(logit, target)[0]
                    if f"proto_acc_local{vid}" not in buffer_meter:
                        buffer_meter[f"proto_acc_local{vid}"] = AverageMeter(f"proto_acc_local{vid}", ":6.4f")
                    buffer_meter[f"proto_acc_local{vid}"].update(accp[0], images[0].size(0))
            loss_local_proto /= len(args.num_cluster)
            loss += loss_local_proto
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if dist.get_rank() == 0:
                progress.display(i)

def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),args.dim).cuda()
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images, is_eval=True)
            features[index] = feat
    dist.barrier()        
    dist.all_reduce(features, op=dist.ReduceOp.SUM)
    return features.cpu()

def run_hkmeans(x, args):
    """
    This function is a hierarchical 
    k-means: the centroids of current hierarchy is used
    to perform k-means in next step
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[], 'cluster2cluster':[], 'logits':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.local_rank  
        index = faiss.GpuIndexFlatL2(res, d, cfg)  
        if seed==0: # the first hierarchy from instance directly
            clus.train(x, index)   
            D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        else:
            # the input of higher hierarchy is the centorid of lower one
            clus.train(results['centroids'][seed - 1].cpu().numpy(), index)
            D, I = index.search(results['centroids'][seed - 1].cpu().numpy(), 1)
        
        im2cluster = [int(n[0]) for n in I]
        # sample-to-centroid distances for each cluster 
        ## centroid in lower level to higher level
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

       # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)

        if seed>0: # the im2cluster of higher hierarchy is the index of previous hierachy
            im2cluster = np.array(im2cluster) # enable batch indexing
            results['cluster2cluster'].append(torch.LongTensor(im2cluster).cuda())
            im2cluster = im2cluster[results['im2cluster'][seed - 1].cpu().numpy()]
            im2cluster = list(im2cluster)
    
        if len(set(im2cluster))==1:
            print("Warning! All samples are assigned to one cluster")

        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) 
        density = args.T*density/density.mean() 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    
        if seed > 0: #maintain a logits from lower prototypes to higher
            proto_logits = torch.mm(results['centroids'][-1], centroids.t())
            results['logits'].append(proto_logits.cuda())


        density = torch.Tensor(density).cuda()
        im2cluster = torch.LongTensor(im2cluster).cuda()    
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
        for meter in self.meters:
            if isinstance(meter, AverageMeter):
                entries += [str(meter)]
            elif isinstance(meter, dict):
                entries += [str(v) for (k, v) in meter.items()]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr = args.lr_final + 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (args.lr - args.lr_final)
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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


if __name__ == '__main__':
    main()
