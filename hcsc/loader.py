from PIL import ImageFilter
import random
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import os

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
        
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class MultiCropsTransform(object):
    """
    The code is modified from 
    https://github.com/maple-research-lab/AdCo/blob/b8f749db3e8e075f77ec17f859e1b2793844f5d3/data_processing/MultiCrop_Transform.py
    """
    def __init__(
            self,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,normalize,init_size=224):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        trans=[]
        #image_k
        weak = transforms.Compose([
            transforms.RandomResizedCrop(init_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        trans.append(weak)
        trans_weak=[]

        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )


            weak=transforms.Compose([
            randomresizedcrop,
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
            trans_weak.extend([weak]*nmb_crops[i])

        trans.extend(trans_weak)
        self.trans=trans
        print("in total we have %d transforms"%(len(self.trans)))
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops


    

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index


class CIFAR10Instance(datasets.CIFAR10):
    def __init__(self, root="./", train=True, transform=None, target_transform=None, download=True):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, index

class CIFAR100Instance(datasets.CIFAR100):
    def __init__(self, root="./", train=True, transform=None, target_transform=None, download=True):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, index


def build_augmentation(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if args.multi_crop:
        augmentation = MultiCropsTransform(args.size_crops,
                                           args.nmb_crops,
                                           args.min_scale_crops,
                                           args.max_scale_crops,
                                           normalize)
    else:
        if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        else:
        # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
            augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        augmentation = TwoCropsTransform(transforms.Compose(augmentation))
        
    # center-crop augmentation 
    eval_augmentation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
        ])    
    return augmentation, eval_augmentation

def cifar10(args):
    augmentation, eval_augmentation = build_augmentation(args)
    train_dataset = CIFAR10Instance(root="./", transform=augmentation, download=True)
    eval_dataset = CIFAR10Instance(root="./", transform=eval_augmentation, download=True)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size*5, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler

def cifar100(args):
    augmentation, eval_augmentation = build_augmentation(args)
    train_dataset = CIFAR100Instance(root="./", transform=augmentation, download=True)
    eval_dataset = CIFAR100Instance(root="./", transform=eval_augmentation, download=True)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size*5, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler

def imagenet(args):
        # Data loading code
    if args.debug:
        traindir = os.path.join(args.data, 'val')
    else:
        traindir = os.path.join(args.data, 'train')
    
    augmentation, eval_augmentation = build_augmentation(args)

    train_dataset = ImageFolderInstance(
        traindir,
        augmentation)
    eval_dataset = ImageFolderInstance(
        traindir,
        eval_augmentation)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size*5, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler
