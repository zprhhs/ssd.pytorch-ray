# horovod
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import math
from tqdm import tqdm
# SSD
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# Training settings
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch & Horovod')
parser.add_argument('--log-dir', default='./logs', help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
# ssd
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=1e-3,
                    help='learning rate for a single GPU')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.base_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * hvd.size() * args.batches_per_allreduce

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break
    
    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.

    cfg = voc
    train_dataset = VOCDetection(
        root = args.dataset_root,
        transform = SSDAugmentation(cfg['min_dim'],MEANS)
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas = hvd.size(),
        rank=hvd.rank()
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size = allreduce_batch_size,
        sampler = train_sampler,
        collate_fn=detection_collate,
        **kwargs
    )

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    model = ssd_net
    vgg_weights = torch.load("./weights/vgg16_reducedfc.pth")
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        model.extras.apply(weights_init)
        model.loc.apply(weights_init)
        model.conf.apply(weights_init)
    model.vgg.load_state_dict(vgg_weights)

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce # * hvd.size()

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(
        model.parameters(),
        lr = (args.base_lr*lr_scaler),
        momentum = args.momentum,
        weight_decay=args.weight_decay
    )

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression = compression,
        backward_passes_per_step = args.batches_per_allreduce,
        op = hvd.Average,
        gradient_predivide_factor = args.gradient_predivide_factor
    )
    criterion = MultiBoxLoss(
        cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda
    )

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    iter_sum = 0
    for epoch in range(resume_from_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = Metric("train_loss")
        with tqdm(total=len(train_loader), desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:

            batch_iterator = iter(train_loader)
            for iteration in range(len(train_loader)):
                iter_sum += 1
                if iter_sum in cfg['lr_steps']:
                    step_index += 1
                    adjust_learning_rate(optimizer, args.gamma, step_index)
    
                # load train data
                images, targets = next(batch_iterator)
    
                if args.cuda:
                    images = Variable(images.cuda())
                    targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
                else:
                    images = Variable(images)
                    targets = [Variable(ann, volatile=True) for ann in targets]
                # forward
                t0 = time.time()
                out = model(images)
                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                train_loss.update(loss)
                loss.backward()
                optimizer.step()
                t1 = time.time()
                lr = None
                for param_group in optimizer.param_groups:
                    lr = param_group["lr"]
                    break
                t.set_postfix({'loss': train_loss.avg.item(), 'lr': lr})
                t.update(1)

