import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data.distributed import (
    DistributedSampler,
) 


def data_loader(data_dir,batch_size,shuffle=False,test=False):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
    ])

    if test:
        dataset = datasets.CIFAR100(
          root=data_dir, train=False,
          download=True, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,sampler=DistributedSampler(dataset)
        )

        return data_loader

    # load the dataset
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=DistributedSampler(train_dataset))

    return train_loader
