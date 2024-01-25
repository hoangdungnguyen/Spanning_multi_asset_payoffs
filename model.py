import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# NN_SingleAsset, NN_Longonly, LS_GD, NN
def maxout(x, nb_sub=1):
    return x * (x >= torch.sort(x, dim=0,
                                descending=True)[0][nb_sub - 1:nb_sub, :])

class NN_SingleAsset(nn.Module):
    def __init__(self, input_dim, init_basket):
        super(NN_SingleAsset, self).__init__()
        self.name = 'NN-Single asset' 
        self.units = init_basket.shape[1]
        init_basket = torch.as_tensor(init_basket, dtype=torch.float32)

        self.w = nn.Parameter(init_basket)
        self.k = nn.Parameter(torch.zeros((1, self.units)))
        self.delta = nn.Parameter(
            torch.normal(mean=0.,
                         std=math.sqrt(2 / (1 + self.units)),
                         size=(1, self.units)))
        self.v = nn.Parameter(
            torch.normal(mean=0.,
                         std=math.sqrt(2 / (1 + self.units)),
                         size=(self.units, 1)))

        self.activation = F.relu

        self.linear = nn.Linear(input_dim, 1)
        torch.nn.init.xavier_normal_(self.linear.weight)
        self.linear.bias.data.fill_(0.001)

        self.batchnorm = nn.BatchNorm1d(self.units)
        self.batchnorm2 = nn.BatchNorm1d(1)

    def forward(self, x):
        self.y = torch.exp(self.w)
        self.y = self.y / self.y.sum(dim=0, keepdim=True)
        self.y = maxout(self.y, 1)

        xx = self.activation(self.delta *
                             (torch.matmul(x, self.y) - torch.exp(self.k)))
        xx = self.batchnorm(xx)
        xx = torch.matmul(xx, self.v) + self.batchnorm2(self.linear(x))
        return xx


class NN_Longonly(nn.Module):
    def __init__(self, input_dim, init_basket):
        super(NN_Longonly, self).__init__()
        self.name = 'NN-Long only'
        self.units = init_basket.shape[1]
        init_basket = torch.as_tensor(init_basket, dtype=torch.float32)

        self.w = nn.Parameter(init_basket)
        self.k = nn.Parameter(torch.zeros((1, self.units)))
        self.delta = nn.Parameter(
            torch.normal(mean=0.,
                         std=math.sqrt(2 / (1 + self.units)),
                         size=(1, self.units)))
        self.v = nn.Parameter(
            torch.normal(mean=0.,
                         std=math.sqrt(2 / (1 + self.units)),
                         size=(self.units, 1)))

        self.activation = F.relu

        self.linear = nn.Linear(input_dim, 1)
        torch.nn.init.xavier_normal_(self.linear.weight)
        self.linear.bias.data.fill_(0.001)

        self.batchnorm = nn.BatchNorm1d(self.units)
        self.batchnorm2 = nn.BatchNorm1d(1)

    def forward(self, x):
        self.y = torch.abs(self.w)

        xx = self.activation(self.delta *
                             (torch.matmul(x, self.y) - torch.exp(self.k)))
        xx = self.batchnorm(xx)
        xx = torch.matmul(xx, self.v) + self.batchnorm2(self.linear(x))
        return xx


class LS_GD(nn.Module):
    def __init__(self, input_dim, init_basket):
        super(LS_GD, self).__init__()
        self.name = 'LS-GD'
        self.units = init_basket.shape[1]
        self.register_buffer('y',
                             torch.as_tensor(init_basket, dtype=torch.float32))

        self.k = 1.
        self.v = nn.Parameter(
            torch.normal(mean=0.,
                         std=math.sqrt(2 / (1 + self.units)),
                         size=(self.units, 1)))

        self.activation = F.relu

        self.linear = nn.Linear(input_dim, 1)
        torch.nn.init.xavier_normal_(self.linear.weight)
        self.linear.bias.data.fill_(0.001)

        self.batchnorm = nn.BatchNorm1d(self.units)
        self.batchnorm2 = nn.BatchNorm1d(1)

    def forward(self, x):
        xx = self.activation(torch.matmul(x, self.y) - self.k)
        xx = self.batchnorm(xx)
        xx = torch.matmul(xx, self.v) + self.batchnorm2(self.linear(x))
        return xx


class NN(nn.Module):
    def __init__(self, input_dim, init_basket):
        super(NN, self).__init__()
        self.name = 'NN'
        self.units = init_basket.shape[1]
        init_basket = torch.as_tensor(init_basket, dtype=torch.float32)

        self.w = nn.Parameter(init_basket)
        self.k = nn.Parameter(torch.zeros((1, self.units)))
        self.delta = nn.Parameter(
            torch.normal(mean=0.,
                         std=math.sqrt(2 / (1 + self.units)),
                         size=(1, self.units)))
        self.v = nn.Parameter(
            torch.normal(mean=0.,
                         std=math.sqrt(2 / (1 + self.units)),
                         size=(self.units, 1)))

        self.activation = F.relu

        self.linear = nn.Linear(input_dim, 1)
        torch.nn.init.xavier_normal_(self.linear.weight)
        self.linear.bias.data.fill_(0.001)

        self.batchnorm = nn.BatchNorm1d(self.units)
        self.batchnorm2 = nn.BatchNorm1d(1)

    def forward(self, x):
        xx = self.activation(self.delta *
                             (torch.matmul(x, self.w) - torch.exp(self.k)))
        xx = self.batchnorm(xx)

        xx = torch.matmul(xx, self.v) + self.batchnorm2(self.linear(x))
        return xx