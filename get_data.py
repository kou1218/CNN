import torch
import torchvision
import torchvision.transforms as transforms


# 前処理
trans = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))])
traindata = torchvision.datasets.MNIST(root = '/home/kou/study/CNN/datasets', train = True, download = True, transform = trans)