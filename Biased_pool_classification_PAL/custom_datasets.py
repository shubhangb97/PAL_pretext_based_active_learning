import os
import random


import numpy as np


import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from skimage import transform


def cifar10_transformer():
    return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.247, 0.243, 0.261]),
        ])
def cifar100_transformer():
    return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
        ])


def svhn_transformer():
    return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
        ])





class CIFAR10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class rot_CIFAR10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer())

    def rotate_img(self,img, rot):
        if rot == 0: # 0 degrees rotation
            return img
        elif rot == 90: # 90 degrees rotation
            return np.flipud(np.transpose(img, (1,0,2)))
        elif rot == 180: # 90 degrees rotation
            return np.fliplr(np.flipud(img))
        elif rot == 270: # 270 degrees rotation / or -90
            return np.transpose(np.flipud(img), (1,0,2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.cifar10[index]
        data=np.asarray(data)

        data=np.moveaxis(data,0,-1)
        data0=data
        data90=self.rotate_img(data,90)
        data180=self.rotate_img(data,180)
        data270=self.rotate_img(data,270)

        data=np.moveaxis(data,2,0)
        data0=np.moveaxis(data0,2,0)
        data90=np.moveaxis(data90,2,0)
        data180=np.moveaxis(data180,2,0)
        data270=np.moveaxis(data270,2,0)


        data0=torch.from_numpy(data0.copy()).float()
        data90=torch.from_numpy(data90.copy()).float()
        data180=torch.from_numpy(data180.copy()).float()
        data270=torch.from_numpy(data270.copy()).float()

        target0=torch.from_numpy(np.array(0)).long()
        target90=torch.from_numpy(np.array(1)).long()
        target180=torch.from_numpy(np.array(2)).long()
        target270=torch.from_numpy(np.array(3)).long()


        dataFull=torch.stack([data0,data90,data180,data270],dim=0)

        targetsRot=torch.stack([target0,target90,target180,target270],dim=0)

        return dataFull,target,targetsRot,index


    def __len__(self):
        return len(self.cifar10)

class rot_CIFAR100(Dataset):
    def __init__(self, path):
        self.cifar100 = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar100_transformer())

    def rotate_img(self,img, rot):
        if rot == 0: # 0 degrees rotation
            return img
        elif rot == 90: # 90 degrees rotation
            return np.flipud(np.transpose(img, (1,0,2)))
        elif rot == 180: # 90 degrees rotation
            return np.fliplr(np.flipud(img))
        elif rot == 270: # 270 degrees rotation / or -90
            return np.transpose(np.flipud(img), (1,0,2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')



    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.cifar100[index]
        data=np.asarray(data)


        data=np.moveaxis(data,0,-1)
        data0=data
        data90=self.rotate_img(data,90)
        data180=self.rotate_img(data,180)
        data270=self.rotate_img(data,270)

        data=np.moveaxis(data,2,0)
        data0=np.moveaxis(data0,2,0)
        data90=np.moveaxis(data90,2,0)
        data180=np.moveaxis(data180,2,0)
        data270=np.moveaxis(data270,2,0)


        data0=torch.from_numpy(data0.copy()).float()
        data90=torch.from_numpy(data90.copy()).float()
        data180=torch.from_numpy(data180.copy()).float()
        data270=torch.from_numpy(data270.copy()).float()

        target0=torch.from_numpy(np.array(0)).long()
        target90=torch.from_numpy(np.array(1)).long()
        target180=torch.from_numpy(np.array(2)).long()
        target270=torch.from_numpy(np.array(3)).long()


        dataFull=torch.stack([data0,data90,data180,data270],dim=0)

        targetsRot=torch.stack([target0,target90,target180,target270],dim=0)

        return dataFull,target,targetsRot,index

    def __len__(self):
        return len(self.cifar100)





class CIFAR100(Dataset):
    def __init__(self, path):
        self.cifar100 = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar100_transformer())

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.cifar100[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar100)



class SVHN(Dataset):
    def __init__(self, path):
        self.svhn = datasets.SVHN(root=path,
                                        download=True,
                                        split='train',
                                        transform=svhn_transformer())

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.svhn[index]

        return data, target, index

    def __len__(self):
        return len(self.svhn)


class rot_SVHN(Dataset):
    def __init__(self, path):
        self.svhn = datasets.SVHN(root=path,
                                        download=True,
                                        split='train',
                                        transform=svhn_transformer())

    def rotate_img(self,img, rot):
        if rot == 0: # 0 degrees rotation
            return img
        elif rot == 90: # 90 degrees rotation
            return np.flipud(np.transpose(img, (1,0,2)))
        elif rot == 180: # 90 degrees rotation
            return np.fliplr(np.flipud(img))
        elif rot == 270: # 270 degrees rotation / or -90
            return np.transpose(np.flipud(img), (1,0,2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.svhn[index]
        data=np.asarray(data)


        data=np.moveaxis(data,0,-1)
        data0=data
        data90=self.rotate_img(data,90)
        data180=self.rotate_img(data,180)
        data270=self.rotate_img(data,270)

        data=np.moveaxis(data,2,0)
        data0=np.moveaxis(data0,2,0)
        data90=np.moveaxis(data90,2,0)
        data180=np.moveaxis(data180,2,0)
        data270=np.moveaxis(data270,2,0)

        data0=torch.from_numpy(data0.copy()).float()
        data90=torch.from_numpy(data90.copy()).float()
        data180=torch.from_numpy(data180.copy()).float()
        data270=torch.from_numpy(data270.copy()).float()

        target0=torch.from_numpy(np.array(0)).long()
        target90=torch.from_numpy(np.array(1)).long()
        target180=torch.from_numpy(np.array(2)).long()
        target270=torch.from_numpy(np.array(3)).long()

        dataFull=torch.stack([data0,data90,data180,data270],dim=0)

        targetsRot=torch.stack([target0,target90,target180,target270],dim=0)

        return dataFull,target,targetsRot,index


    def __len__(self):
        return len(self.svhn)
