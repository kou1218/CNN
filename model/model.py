import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy

class torch_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 畳み込み層
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)

        # 全結合層
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.relu(F.max_pool2d(self.conv2(x), 2))

        # テンソルを平らにする処理
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x