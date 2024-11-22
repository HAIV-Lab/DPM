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




