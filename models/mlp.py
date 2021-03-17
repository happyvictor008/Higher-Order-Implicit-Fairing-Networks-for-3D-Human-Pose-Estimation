import torch
import torch.utils.data
from torch import optim, nn
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self,in_features,out_features):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.model = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, out_features),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)

        return x
