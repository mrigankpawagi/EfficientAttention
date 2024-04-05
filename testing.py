import numpy as np
# import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

#Load MNIST
transform = transforms.Compose([transforms.ToTensor()])
batch_size = 64
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)
train_x = train_dataset.train_data.numpy()
train_y = train_dataset.train_labels.numpy()
test_x = test_dataset.train_data.numpy()
train_y = test_dataset.train_labels.numpy()
print(train_x)
print(train_y)