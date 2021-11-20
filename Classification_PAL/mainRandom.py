#Run this in place of main.py to run the Random sampling baseline with hyperparameters in arguments
import os
os.environ['PYTHONHASHSEED']=str(101)

import random
random.seed(101)

import numpy as np
np.random.seed(101)

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(101)
torch.cuda.manual_seed(101)
torch.cuda.manual_seed_all(101)

from torchvision import datasets, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data
from RotNetModel1 import RotNetMulti
from RotNetModel1 import RotNetMultiPretrained
import numpy as np
import argparse
import torch.nn as nn
import vggcifar

import samplerMulti2
from custom_datasets import *
import vggcifarpretrained
from solverMulti import Solver
import arguments

def cifar10_transformer():
    return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465,],
                                std=[0.247, 0.243, 0.261,]),
        ])


def cifar100_transformer():
    return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
        ])


def caltech256_transformer():
    return transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB") ),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224)),
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
def caltech101_transformer():
    return transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB") ),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
        ])

def main(args):

    print("Seed 101")

    if args.dataset == 'cifar10':
        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar10_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False, num_workers=0)

        train_dataset = CIFAR10(args.data_path)
        rot_train_dataset = rot_CIFAR10(args.data_path)
        rot_test_dataset = rot_CIFAR10(args.data_path)

        args.num_images = 50000
        args.num_val = 5000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar100_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR100(args.data_path)
        rot_train_dataset = rot_CIFAR100(args.data_path)
        rot_test_dataset = rot_CIFAR100(args.data_path)

        args.num_val = 5000
        args.num_images = 50000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 100

    elif args.dataset == 'imagenet':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = ImageNet(args.data_path)

        args.num_val = 128120
        args.num_images = 1281167
        args.budget = 64060
        args.initial_budget = 128120
        args.num_classes = 1000

    elif args.dataset == 'caltech256':

        args.num_val = 3000
        args.num_images = 27607
        args.budget = 1530
        args.initial_budget = 3060
        args.num_classes = 257

        all_indices = set(np.arange(args.num_images))
        test_indices=random.sample(list(all_indices),1530)
        test_sampler = data.sampler.SubsetRandomSampler(test_indices)
        all_indices = np.setdiff1d(list(all_indices), test_indices)
        train_dataset = Caltech256(args.data_path)

        test_dataloader=data.DataLoader(train_dataset,sampler=test_sampler,batch_size=args.batch_size,drop_last=False,num_workers=0)
        rot_train_dataset = rot_Caltech256(args.data_path)

    elif args.dataset == 'caltech101':

        args.num_val = 914
        args.num_images = 8232#9146
        args.budget = 411#1530
        args.initial_budget = 822#3060
        args.num_classes = 102#256

        all_indices = set(np.arange(args.num_images))
        test_indices=random.sample(list(all_indices),822)
        test_sampler = data.sampler.SubsetRandomSampler(test_indices)
        all_indices = np.setdiff1d(list(all_indices), test_indices)
        train_dataset = Caltech101(args.data_path)

        test_dataloader=data.DataLoader(train_dataset,sampler=test_sampler,batch_size=args.batch_size,drop_last=False,num_workers=0)
        rot_train_dataset = rot_Caltech101(args.data_path)

    elif args.dataset == 'svhn' :
        test_dataloader = data.DataLoader(datasets.SVHN(args.data_path,download=True,transform=svhn_transformer(), split='test'),batch_size=args.batch_size,drop_last=False,num_workers=0)
        train_dataset= SVHN(args.data_path)
        rot_train_dataset=rot_SVHN(args.data_path)

        args.num_images=73257
        args.num_val=7325
        args.budget=3660
        args.initial_budget=7325
        args.num_classes=10


    else:
        raise NotImplementedError
    print("Random sampling only")
    print("Batch Size is"+str(args.batch_size))
    print("100 epochs only ")
    if not(args.dataset == 'caltech256' or args.dataset=='caltech101'):
        all_indices = set(np.arange(args.num_images))

    if not(args.dataset=="tinyImageNet"):
        val_indices = random.sample(list(all_indices), args.num_val)
        all_indices = np.setdiff1d(list(all_indices), val_indices)

    initial_indices = random.sample(list(all_indices), args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    if not(args.dataset=="tinyImageNet"):
        val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler,
            batch_size=args.batch_size, drop_last=True,num_workers=0)
    if not(args.dataset=="tinyImageNet"):
        val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler,
                batch_size=args.batch_size, drop_last=False,num_workers=0)
    rot_dataloader=data.DataLoader(rot_train_dataset,sampler=sampler,batch_size=args.batch_size,drop_last=True,num_workers=0)
    if not(args.dataset=="tinyImageNet"):
        rot_val_dataloader=data.DataLoader(rot_train_dataset,sampler=val_sampler,batch_size=args.batch_size,drop_last=True,num_workers=0)


    print("Running on cuda")


    solver = Solver(args, test_dataloader)
    samplerRot=samplerMulti2.RotSampler(args.budget,args)
    splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    current_indices = list(initial_indices)
    num_img1=len(current_indices)

    accuracies = []

    for split in splits:
        if args.dataset=="caltech256":
            task_model=vggcifarpretrained.vgg16_pretrained(num_classes=args.num_classes)
        else:
            task_model = vggcifar.vgg16_bn(num_classes=args.num_classes)

        if args.dataset=="caltech256":
            rotNet1=RotNetMultiPretrained(num_classes=args.num_classes,num_rotations=4)
        else:
            rotNet1=RotNetMulti(num_classes=args.num_classes,num_rotations=4)

        rotNet1.cuda()
        rotNet1=nn.DataParallel(rotNet1)

        task_model=task_model.cuda()
        task_model=nn.DataParallel(task_model)

        #Get unlabeleled indice dataloader

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        remain_indices=np.setdiff1d(list(all_indices),current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset,
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False,num_workers=0)
        rot_unlabeled_dataloader=data.DataLoader(rot_train_dataset,sampler=unlabeled_sampler,batch_size=args.batch_size,drop_last=False,num_workers=0)


        # train task model for this iteration
        acc = solver.train(querry_dataloader,
                                               val_dataloader,
                                               task_model,
                                               unlabeled_dataloader,num_img1)

        print('Final accuracy of Task network with {}% of data is: {:.2f}'.format(int(split*100), acc))

        accuracies.append(acc)

        #sample randomly

        new_random=random.sample(list(remain_indices),args.budget)
        current_indices = list(current_indices) + list(new_random)
        sampler = data.sampler.SubsetRandomSampler(current_indices)

        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler,batch_size=args.batch_size, drop_last=True,num_workers=0)
        rot_dataloader=data.DataLoader(rot_train_dataset,sampler=sampler,batch_size=args.batch_size,drop_last=True,num_workers=0)

        num_img1=len(current_indices)
        torch.save(accuracies, os.path.join(args.out_path, args.log_name))

    torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)
