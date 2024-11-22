
import sys
import os
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from transformers import CLIPModel
from torchvision import datasets, transforms
import torchvision.transforms as transforms




def set_val_loader(args, preprocess=None):
    root = args.root_dir
    if args.in_dataset == "cifar100":
        valset = datasets.ImageFolder(os.path.join(root,'OOD', 'cifar100', 'test'), transform=preprocess)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=512,
                                                shuffle=False, num_workers=4)
        trainset = datasets.ImageFolder(os.path.join(root,'OOD', 'cifar100', 'train'), transform=preprocess)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                                shuffle=False, num_workers=4)
    return val_loader,train_loader


def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):

    if out_dataset == 'ImageNetr':
        testsetout = datasets.ImageFolder(root=os.path.join(root,'OOD', 'imagenet-r'), transform=preprocess)
    elif out_dataset == 'cifar10':
        testsetout = datasets.ImageFolder(root=os.path.join(root,'OOD', 'cifar10','test'), transform=preprocess)
    elif out_dataset == 'LSUN':
        testsetout = datasets.ImageFolder(root=os.path.join(root,'OOD', 'LSUN_C'), transform=preprocess)
    elif out_dataset == 'LSUN_resize':
        testsetout = datasets.ImageFolder(root=os.path.join(root,'OOD', 'LSUN_R'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=512,
                                            shuffle=False, num_workers=4)
    return testloaderOut
