import argparse

def parse_args_main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Pre-training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar10', 'cifar100', 'coco'],
                        help='which dataset should be used to pretrain the model')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        help='model architecture')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr_final', default=0., type=float)
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default='', type=str,
                        help='pretrained backbone')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10034', type=str,
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

    parser.add_argument('--dim', default=128, type=int,
                        help='feature dimension')
    parser.add_argument('--queue_length', default=16384, type=int,
                        help='queue size; number of negative pairs')
    parser.add_argument('--m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--T', default=0.2, type=float,
                        help='temperature')

    parser.add_argument('--mlp', type=int, default=1,
                        help='use mlp head')
    parser.add_argument('--aug-plus', type=int, default=1,
                        help='use moco-v2/SimCLR data augmentation')
    parser.add_argument('--cos', type=int, default=1,
                        help='use cosine lr schedule')
    parser.add_argument('--num-cluster', default='3000,2000,1000', type=str, 
                        help='number of clusters')
    parser.add_argument('--warmup-epoch', default=20, type=int,
                        help='number of warm-up epochs to only train with InfoNCE loss')

    parser.add_argument('--multi_crop', action='store_true',
                        default=False,
                        help='Whether to enable multi-crop transformation')
    parser.add_argument("--nmb_crops", type=int, default=[1, 1, 1, 1, 1], nargs="+",
                        help="list of number of crops (example: [2, 6])") 
    parser.add_argument("--size_crops", type=int, default=[224, 192, 160, 128, 96], nargs="+",
                        help="crops resolutions (example: [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, default=[0.2, 0.172, 0.143, 0.114, 0.086], nargs="+",
                        help="argument in RandomResizedCrop (example: [0.14, 0.05])")
    parser.add_argument("--max_scale_crops", type=float, default=[1.0, 0.86, 0.715, 0.571, 0.429], nargs="+",
                        help="argument in RandomResizedCrop (example: [1., 0.14])")

    ## Selection configs
    parser.add_argument("--selection_on_local", action="store_true", default=False,
                        help="whether enable mining on local views")

    parser.add_argument("--instance_selection", type=int, default=1,
                        help="Whether enable instance selection")
    parser.add_argument("--proto_selection", type=int, default=1,
                        help="Whether enable prototype selection")              

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--exp-dir', default='experiment_pcl', type=str,
                        help='experiment directory')

    args = parser.parse_args()
    return args


def parse_args_lincls_imagenet():
    parser = argparse.ArgumentParser(description='ImageNet Linear Classification')
    parser.add_argument('--data', type=str, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=5., type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[60,80], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by a ratio)') 
    parser.add_argument('--cos', type=int, default=0,
                        help='use cosine lr schedule')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10028', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=int, default=1,
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    parser.add_argument('--pretrained', default='', type=str,
                        help='path to moco pretrained checkpoint')
    parser.add_argument('--choose', type=str, default=None, help="choose gpu for training")
    parser.add_argument("--dataset", type=str, default="ImageNet", help="which dataset is used to finetune")
    parser.add_argument("--final_lr", type=float, default=0.0, help="ending learning rate for training")
    parser.add_argument('--save_path', default="", type=str, help="model and record save path")
    parser.add_argument('--log_path', type=str, default="train_log", help="log path for saving models")
    parser.add_argument("--ngpu", type=int, default=8, help="number of gpus per node")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="addr for master node")
    parser.add_argument("--master_port", type=str, default="1234", help="port for master node")

    args = parser.parse_args()
    return args


def parse_args_lincls_places():
    parser = argparse.ArgumentParser(description='Places205 Linear Classification')
    parser.add_argument('--data', type=str, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=3., type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by a ratio)')  # default is for places205
    parser.add_argument('--cos', type=int, default=1,
                        help='use cosine lr schedule')
    parser.add_argument("--sgdr", type=int, default=2)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10028', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=int, default=1,
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    parser.add_argument('--pretrained', default='', type=str,
                        help='path to moco pretrained checkpoint')
    parser.add_argument('--choose', type=str, default=None, help="choose gpu for training")
    parser.add_argument("--dataset", type=str, default="ImageNet", help="which dataset is used to finetune")
    parser.add_argument("--strong", type=int, default=0, help="use strong augmentation or not")
    parser.add_argument("--final_lr", type=float, default=0.01, help="ending learning rate for training")
    parser.add_argument('--save_path', default="", type=str, help="model and record save path")
    parser.add_argument('--log_path', type=str, default="train_log", help="log path for saving models")
    parser.add_argument("--nodes_num", type=int, default=1, help="number of nodes to use")
    parser.add_argument("--ngpu", type=int, default=8, help="number of gpus per node")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="addr for master node")
    parser.add_argument("--master_port", type=str, default="1234", help="port for master node")
    parser.add_argument('--node_rank', type=int, default=0, help='rank of machine, 0 to nodes_num-1')
    parser.add_argument("--avg_pool", default=1, type=int, help="average pool output size")
    parser.add_argument("--crop_scale", type=float, default=[0.2, 1.0], nargs="+",
                        help="argument in RandomResizedCrop (example: [1., 0.14])")
    parser.add_argument("--train_strong", type=int, default=1, help="training use stronger augmentation or not")
    parser.add_argument("--sgdr_t0", type=int, default=10, help="sgdr t0")
    parser.add_argument("--sgdr_t_mult", type=int, default=1, help="sgdr t mult")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout layer settings")
    parser.add_argument("--randcrop", type=int, default=1, help="use random crop or not")

    args = parser.parse_args()
    return args

def parse_semisup_args():
    
    parser = argparse.ArgumentParser(description="ImageNet Semi-supervised Learning Evaluation")

    parser.add_argument("--labels_perc", type=str, default="10", choices=["1", "10"],
                        help="fine-tune on either 1% or 10% of labels")
    parser.add_argument("--exp_dir", type=str, default=".",
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("data_path", type=str, default="",
                        help="path to imagenet")
    parser.add_argument("--workers", default=10, type=int,
                        help="number of data loading workers")
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")
    parser.add_argument("--epochs", default=70, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--lr", default=0.005, type=float, help="initial learning rate - trunk")
    parser.add_argument("--lr_last_layer", default=0.05, type=float, help="initial learning rate - head")
    parser.add_argument("--decay_epochs", type=int, nargs="+", default=[30, 60],
                        help="Epochs at which to decay learning rate.")
    parser.add_argument("--gamma", type=float, default=0.1, help="lr decay factor")

    parser.add_argument("--dist_url", default="env://", type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--world_size", default=-1, type=int, help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")

    args = parser.parse_args()
    return args

def parse_args_knn():
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=1, type=int,
        help="Store features in GPU.")
    parser.add_argument('--arch', default='resnet50', type=str, help='Architecture')
    parser.add_argument("--checkpoint_key", default="state_dict", type=str,
        help='Key to use in the checkpoint')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; """)
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument('--data', type=str)
    args = parser.parse_args()
    return args
