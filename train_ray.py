from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime
import torch.utils.data
import torchvision

import ray
from ray.util.sgd.torch.examples.segmentation.coco_utils import get_coco
import ray.util.sgd.torch.examples.segmentation.transforms as T
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd import TorchTrainer

try:
    from apex import amp
except ImportError:
    amp = None


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
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
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument(
    "--address",
    required=False,
    default=None,
    help="the address to use for connecting to a Ray cluster.")
parser.add_argument("--model", default="fcn_resnet101", help="model")
parser.add_argument(
    "--aux-loss", action="store_true", help="auxiliar loss")
parser.add_argument("--device", default="cuda", help="device")
parser.add_argument("-b", "--batch-size", default=8, type=int)
parser.add_argument(
    "-n", "--num-workers", default=1, type=int, help="GPU parallelism")
parser.add_argument(
    "--epochs",
    default=30,
    type=int,
    metavar="N",
    help="number of total epochs to run")
parser.add_argument(
    "--data-workers",
    default=16,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 16)")
parser.add_argument("--output-dir", default=".", help="path where to save")
parser.add_argument(
    "--pretrained",
    dest="pretrained",
    help="Use pre-trained models from the modelzoo",
    action="store_true",
)

args = parser.parse_args()


def get_dataset(config, name):
    args = config['args']
    cfg = config['cfg']
    if name == "train":
        dataset = VOCDetection(
            root=args.dataset_root,
            transform=SSDAugmentation(cfg['min_dim'],MEANS)
        )
        return dataset
    else:
        dataset = VOCDetection(
            args.dataset_root,
            [('2007', 'test')],
            None,
            VOCAnnotationTransform()
        )
        return dataset

def data_creator(config):
    # Within a machine, this code runs synchronously.
    args = config["args"]
    cfg = config["args"]
    dataset = get_dataset(config, "train")
    # cfg["num_classes"] = num_classes
    data_loader = data.DataLoader(
            dataset,
            args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=detection_collate,
            pin_memory=True
    )
    data_loader_test = get_dataset(config, "val")
    return data_loader, data_loader_test 



def model_creator(config):
    args = config["args"]
    cfg = config["cfg"]
    model = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    if config["num_workers"] > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def criterion(outputs, targets, device):
    criterion_ = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5,
                             False, False)
    # outputs = [Variable(ann.cuda(device)) for ann in outputs]
    # targets = [Variable(ann.cuda(device)) for ann in targets]
    return criterion_(outputs, targets, device)


def optimizer_creator(model, config):
    args = config["args"]
    return optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
 

class SegOperator(TrainingOperator):
    def train_batch(self, batch, batch_info):
        image, target = batch
        image = Variable(image.cuda(self.device))
        target = [Variable(ann.cuda(self.device)) for ann in target]
        output = self.model(image)
        output = [ann.cuda(self.device) for ann in output]
        loss_l, loss_c = criterion(output, target, self.device)
        loss = loss_l + loss_c
        self.optimizer.zero_grad()
        if self.use_fp16 and amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()
        lr = self.optimizer.param_groups[0]["lr"]
        return {"loss": loss.item(), "lr": lr, "num_samples": len(batch)}

    #TODO

def main():
    os.makedirs(args.output_dir, exist_ok=True)
    print(args)
    if args.dataset_root == COCO_ROOT:
        parser.error('Must specify dataset if specifying dataset_root')
    cfg = voc
        
    start_time = time.time()
    config = {"args": args, "num_workers": args.num_workers, "cfg": cfg}
    trainer = TorchTrainer(
        model_creator=model_creator,
        data_creator=data_creator,
        optimizer_creator=optimizer_creator,
        training_operator_cls=SegOperator,
        use_tqdm=True,
        use_fp16=False,
        num_workers=config["num_workers"],
        config=config,
        use_gpu=torch.cuda.is_available()
    )
    for epoch in range(args.epochs):
        trainer.train()
        state_dict = trainer.state_dict()
        state_dict.update(epoch=epoch, args=args)
        torch.save(state_dict,
                   os.path.join(args.output_dir, "model_{}.pth".format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

if __name__ == "__main__":
    ray.init(address=args.address)
    main()
