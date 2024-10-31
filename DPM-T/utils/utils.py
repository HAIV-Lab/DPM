import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import pickle
from PIL import Image
import torchvision
import os
class CosineClassifier(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(CosineClassifier, self).__init__()
        self.class_vectors = nn.Parameter(torch.randn(num_classes, input_dim))

    def forward(self, x):
        x = F.normalize(x, dim=-1)  
        class_vectors = F.normalize(self.class_vectors, dim=1)  
        similarities = torch.mul(x, class_vectors)
        similarities = torch.sum(similarities, dim=2)  
        return similarities

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_path, transform):
        self.transform = transform
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        # print(self.data.keys())
        x = self.data['train'][idx].impath
        y = self.data['train'][idx].label
        img = Image.open(x)
        img = self.transform(img)
        return img, y

    def __len__(self):
        # print(self.data['train'])
        return len(self.data['train'])

def set_val_loader(args, preprocess=None):
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "ImageNet":
        path = os.path.join(args.root, 'imagenet','val')  
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        path = os.path.join(args.root, 'imagenet','train')  
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    return val_loader, train_loader


def set_ood_loader_ImageNet(out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':  # filtered places
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'places365'), transform=preprocess)
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                                      transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=64,
                                                shuffle=False, num_workers=4)
    return testloaderOut

