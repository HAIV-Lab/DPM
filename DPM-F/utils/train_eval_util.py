
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



def set_model_clip(args):
    '''
    load Huggingface CLIP
    '''
    ckpt_mapping = {"ViT-B/16":"/data/hdd/xz2002/xz2002/CLIP-OOD/DualCoOp-main/MCM/pretrained_model/clip-vit-base-patch16/",
                    "ViT-B/32":"/data/hdd/xz2002/xz2002/CLIP-OOD/DualCoOp-main/MCM/pretrained_model/clip-vit-base-patch32/",
                    "ViT-L/14":"openai/clip-vit-large-patch14"}
    args.ckpt = ckpt_mapping[args.CLIP_ckpt]
    model =  CLIPModel.from_pretrained(args.ckpt)
    if args.model == 'CLIP-Linear':
        model.load_state_dict(torch.load(args.finetune_ckpt, map_location=torch.device(args.gpu)))
    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    return model, val_preprocess

def set_train_loader(args, preprocess=None, batch_size=None, shuffle=False, subset=False):
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if batch_size is None:  # normal case: used for trainign
        batch_size = args.batch_size
        shuffle = True
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
        dataset = datasets.ImageFolder(path, transform=preprocess)
        if subset:
            from collections import defaultdict
            classwise_count = defaultdict(int)
            indices = []
            for i, label in enumerate(dataset.targets):
                if classwise_count[label] < args.max_count:
                    indices.append(i)
                    classwise_count[label] += 1
            dataset = torch.utils.data.Subset(dataset, indices)
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100"]:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'train'), transform=preprocess),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    return train_loader


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # 调用父类的 __getitem__ 方法获取图像和标签
        original_data = super().__getitem__(index)


        return original_data[0], original_data[1],index

def set_val_loader(args, preprocess=None):
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "ImageNet":
        path = "/new_data/datasets/OOD_dataset/ImageNet/val/"
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        path = "/new_data/datasets/OOD_dataset/ImageNet/train/"
        train_loader = torch.utils.data.DataLoader(
            CustomImageFolder(path, transform=preprocess),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100"]:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        train_loader = torch.utils.data.DataLoader(
            CustomImageFolder(os.path.join(
                root, args.in_dataset, 'train'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)

    elif args.in_dataset == "CUB":
        valset = datasets.ImageFolder(os.path.join(
                root, args.in_dataset,'test'), transform=preprocess)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=4)

        train_loader = torch.utils.data.DataLoader(
            CustomImageFolder(os.path.join(
                root, args.in_dataset,'train'), transform=preprocess),
            batch_size=args.batch_size, shuffle=True, **kwargs)


    elif args.in_dataset == "cifar100":
        valset = datasets.ImageFolder(os.path.join('/data/hdd/data_xz/OOD/', 'cifar100', 'test'), transform=preprocess)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=4)
        trainset = datasets.ImageFolder(os.path.join('/data/hdd/data_xz/OOD/', 'cifar100', 'train'), transform=preprocess)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=4)
    elif args.in_dataset == "Cal":
        valset = datasets.ImageFolder(os.path.join(
            root, args.in_dataset, 'test'), transform=preprocess)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=4)

        train_loader = torch.utils.data.DataLoader(
            CustomImageFolder(os.path.join(
                root, args.in_dataset, 'train'), transform=preprocess),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    return val_loader,train_loader


def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365': # filtered places
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'Places'),transform=preprocess)  
    elif out_dataset == 'placesbg': 
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'placesbg'),transform=preprocess)  
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                        transform=preprocess)
        # testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Textures'),
        #                                 transform=preprocess)
    elif out_dataset == 'ImageNet10':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'ImageNet10', 'train'), transform=preprocess)
    elif out_dataset == 'ImageNet20':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'ImageNet20', 'val'), transform=preprocess)

    elif out_dataset == 'ImageNetr':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'imagenet-r'), transform=preprocess)
    elif out_dataset == 'ImageNeto':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'imagenet-o'), transform=preprocess)
    elif out_dataset == 'ImageNeta':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'imagenet-a'), transform=preprocess)
    elif out_dataset == 'ImageNets':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'sketch'), transform=preprocess)
    elif out_dataset == 'ImageNetv2':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'imagenetv2-matched-frequency-format-val'), transform=preprocess)
    elif out_dataset == 'cifar10':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'cifar10','test'), transform=preprocess)

    elif out_dataset == 'CUBez':
        testsetout = datasets.ImageFolder(root="/new_data/xz2002/CLIP/CUBEZ/uk/", transform=preprocess)
    elif out_dataset == 'NINCO':
        testsetout = datasets.ImageFolder(root="/new_data/dataset/OOD/NINCO/NINCO_OOD_classes/", transform=preprocess)
    elif out_dataset == 'CUBhd':
        testsetout = datasets.ImageFolder(root="/new_data/xz2002/CLIP/CUBHD/uk/", transform=preprocess)
    elif out_dataset == 'CUB50uk':
        testsetout = datasets.ImageFolder(root="/new_data/dataset/OOD/CUB_200_2011/OOD-50uk/", transform=preprocess)
    elif out_dataset == 'AWA10uk':
        testsetout = datasets.ImageFolder(root="/new_data/xz2002/CLIP/AWA/uk/", transform=preprocess)

    elif out_dataset == 'LADA10uk':
        testsetout = datasets.ImageFolder(root="/new_data/dataset/OOD/LAD/AOOD-10uk/", transform=preprocess)

    elif out_dataset == 'LADV10uk':
        testsetout = datasets.ImageFolder(root="/new_data/xz2002/CLIP/LAD/uk/" ,transform=preprocess)

    elif out_dataset == 'LADHVuk':
        testsetout = datasets.ImageFolder(root="/new_data/dataset/OOD/LAD/LADHV4/val/" ,transform=preprocess)
    elif out_dataset == 'LADHAuk':
        testsetout = datasets.ImageFolder(root="/new_data/dataset/OOD/LAD/LADHA4/val/" ,transform=preprocess)

    elif out_dataset == 'LSUN':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'LSUN'), transform=preprocess)

    elif out_dataset == 'FGVCez':
        testsetout = datasets.ImageFolder(root="/new_data/xz2002/CLIP/FGVC_EZ/uk/", transform=preprocess)
    elif out_dataset == 'FGVChd':
        testsetout = datasets.ImageFolder(root="/new_data/xz2002/CLIP/FGVC_HD/uk/", transform=preprocess)
    elif out_dataset == 'Scarez':
        testsetout = datasets.ImageFolder(root="/new_data/xz2002/CLIP/ScarEZ/uk/", transform=preprocess)
    elif out_dataset == 'Scarhd':
        testsetout = datasets.ImageFolder(root="/new_data/xz2002/CLIP/ScarHD/uk/", transform=preprocess)
    elif out_dataset == 'LSUN_resize':

        testsetout = datasets.ImageFolder(root=os.path.join(root, 'LSUN_resize'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
    return testloaderOut

def set_ood_loader_ImageNet_my(out_dataset, preprocess, root ,bs=64):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365': # filtered places
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'Places'),transform=preprocess)
    elif out_dataset == 'placesbg':
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'placesbg'),transform=preprocess)
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                        transform=preprocess)
        # testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Textures'),
        #                                 transform=preprocess)
    # elif out_dataset == 'ImageNet10':
    #     testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet10', 'train'), transform=preprocess)
    # elif out_dataset == 'ImageNet20':
    #     testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet20', 'val'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=bs,
                                            shuffle=False, num_workers=4)
    return testloaderOut
