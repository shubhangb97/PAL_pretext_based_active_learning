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

import numpy as np
import argparse
from PIL import Image
import torch.nn as nn
from deeplabv3 import DeepLabMobile
from deeplabv3 import DeepLabScoringMobile

import samplerMulti2
from custom_datasets import *
from solverMulti import Solver
import arguments
import ext_transforms as et


def main(args):

    print("Seed 101")
    print(args)

    if args.dataset == "cityscapes":
        transform_seg=et.ExtCompose([
            #et.ExtResize( 512 ),
            #et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        transform_seg2=et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dataset_ss=Cityscapes_ss(str(args.data_path)+"/cityscapes", split='train',transform=transform_seg2)
        train_dataset=Cityscapes(str(args.data_path)+"/cityscapes",split='train',transform=transform_seg)
        test_dataset=Cityscapes(str(args.data_path)+"/cityscapes",split='val',transform=transform_seg2)
        val_dataset=Cityscapes(str(args.data_path)+"/cityscapes",split='val',transform=transform_seg2)
        val_dataset_ss=Cityscapes_ss(str(args.data_path)+"/cityscapes",split='val',transform=transform_seg2)
        test_dataloader=data.DataLoader(test_dataset,batch_size=args.batch_size,drop_last=True,num_workers=0)

        args.num_images=2975
        args.num_val=500
        args.budget=30
        args.initial_budget=150
        args.num_classes=19

    else:
        raise NotImplementedError

    all_indices = set(np.arange(args.num_images))

    if(args.dataset != "cityscapes"):
        val_indices = random.sample(list(all_indices), args.num_val)
        all_indices = np.setdiff1d(list(all_indices), val_indices)

    initial_indices = random.sample(list(all_indices), args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    if(args.dataset != "cityscapes"):
        val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    # -------------------- initialize dataloader --------------------------------
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler,
            batch_size=args.batch_size, drop_last=True,num_workers=0)
    if(args.dataset != "cityscapes"):
        val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler,batch_size=args.batch_size, drop_last=False,num_workers=0)
    else:
        val_dataloader=data.DataLoader(val_dataset,batch_size=args.batch_size, drop_last=False,num_workers=0)


    rot_dataloader=data.DataLoader(train_dataset_ss,sampler=sampler,batch_size=args.batch_size,drop_last=True,num_workers=0)
    rot_val_dataloader=data.DataLoader(val_dataset_ss,batch_size=args.batch_size,drop_last=False,num_workers=0)

    solver = Solver(args, test_dataloader)
    samplerRot=samplerMulti2.RotSampler(args.budget,args)
    splits = [0.05,0.06,0.07,0.09,0.1]

    current_indices = list(initial_indices)
    num_img1=len(current_indices)

    accuracies = []

    for split in splits:

        task_model=DeepLabMobile(num_classes=args.num_classes)
        task_model=task_model.cuda()
        task_model=nn.DataParallel(task_model)


        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset,
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=True,num_workers=0)
        rot_unlabeled_dataloader=data.DataLoader(train_dataset_ss,sampler=unlabeled_sampler,batch_size=args.batch_size,drop_last=True,num_workers=0)

        #--------------train task model for this iteration------------------------
        acc = solver.train(querry_dataloader,
                                               val_dataloader,
                                               task_model,
                                               num_img1)
        rotNet1=DeepLabScoringMobile(num_classes=args.num_classes)
        rotNet1=rotNet1.cuda()
        rotNet1=nn.DataParallel(rotNet1)


        #--------------train scoring network for this iteration------------------------
        rotNet1=solver.rot_net_train(rot_dataloader,rotNet1,rot_val_dataloader,num_img1)

        print('Final mIoU of Task Network with {}% of data is: {:.2f}'.format(int(split*100), acc))

        accuracies.append(acc)

        #----------Sampling----------------------------
        sampled_indices=samplerRot.sample_query2(rotNet1,rot_unlabeled_dataloader)
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        #---------------initialize dataloaders again----------------
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler,batch_size=args.batch_size, drop_last=True,num_workers=0)
        rot_dataloader=data.DataLoader(train_dataset_ss,sampler=sampler,batch_size=args.batch_size,drop_last=True,num_workers=0)

        num_img1=len(current_indices)
        torch.save(accuracies, os.path.join(args.out_path, args.log_name))

    torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)
