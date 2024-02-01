from model import torch_Net

from get_data import traindata
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np
from time import time

def fix_seed(seed=0):
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    fix_seed()
    batch_size = 128
    epochs = 15
    net = torch_Net()
    train_loader = DataLoader(dataset=traindata, batch_size=batch_size, shuffle=True)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    y_axis_list = []
    alltime = 0

    for epoch in range(epochs):
        start = time()
        for batch, label in train_loader:

            optimizer.zero_grad()

            outputs = net(batch)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()
        end = time() - start
        alltime += end
        print("epoch: %d  loss: %f  time: %f" % (epoch+1, float(loss), end))

        y_axis_list.append(loss.detach().numpy())
    
    x_axis_list = [num for num in range(15)]

    print("alltime: %f" % (alltime))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x_axis_list, y_axis_list)
    plt.savefig("outputs/loss.png")


if __name__ == "__main__":
    main()