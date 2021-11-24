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


def svhn_transformer():
    return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
        ])


def main(args):


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

    print("Random sampling")
    all_indices = set(np.arange(args.num_images))

    val_indices = random.sample(list(all_indices), args.num_val)
    all_indices = np.setdiff1d(list(all_indices), val_indices)

    indices1=[]
    indices2=[]

    allclass=set(np.arange(args.num_classes))
    allclass=list(allclass)
    classbiased=random.sample(allclass,args.biased)
    print("Biasing"+str(classbiased))
    torch.save(classbiased,"biased1.log")
    allclass=np.setdiff1d(allclass,classbiased)

    num1=len(all_indices)
    n1=0
    while(n1<num1):
        _,label1,_,_=rot_train_dataset[all_indices[n1]]
        if(label1 not in classbiased):
            indices1.append(all_indices[n1])
        else:
            indices2.append(all_indices[n1])
        n1=n1+1

    initial_indices=random.sample(list(indices1),args.initial_budget)

    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler,
            batch_size=args.batch_size, drop_last=True,num_workers=0)
    val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler,
            batch_size=args.batch_size, drop_last=False,num_workers=0)
    rot_dataloader=data.DataLoader(rot_train_dataset,sampler=sampler,batch_size=args.batch_size,drop_last=True,num_workers=0)
    rot_val_dataloader=data.DataLoader(rot_train_dataset,sampler=val_sampler,batch_size=args.batch_size,drop_last=True,num_workers=0)


    solver = Solver(args, test_dataloader)
    samplerRot=samplerMulti2.RotSampler(args.budget,args)
    splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    current_indices = list(initial_indices)
    num_img1=len(current_indices)

    accuracies = []

    for split in splits:
        task_model = vggcifar.vgg16_bn(num_classes=args.num_classes)
        rotNet1=RotNetMulti(num_classes=args.num_classes,num_rotations=4)
        rotNet1.cuda()

        task_model=task_model.cuda()


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
