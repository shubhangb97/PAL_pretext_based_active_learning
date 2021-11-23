# PAL - Pretext based Active learning
 We propose a new method to measure novelty of an unlabeled point by introducing pretext-based active learning (PAL) to develop effective sampling strategies for active learning. The proposed method uses the difficulty in solving a self-supervised task as a proxy measure for the novelty of an unlabeled sample.
 This directory contains code to reproduce our experiments reported on 4 different datasets.
### Requirements
- torch==1.5.0
- torchvision==0.6.0
- numpy==1.19.2
- skimage==0.15.0
- sklearn==0.21.2
- PIL==8.1.0
- OpenCV==4.5.1

### Instructions
Firstly, download and extract the dataset on which the code is to be run by in a folder `./data` for classification in this directory or in `./data/cityscapes` in this directory for segmentation.
For running PAL on one of the implemented datasets (CIFAR-10, Cityscapes, SVHN, Caltech-101)-
- choose the appropriate arguments by uncommenting them in arguments.py file as command parameter as specified by us in the technical appendix including
 - --dataset from cifar10, cityscapes, caltech101, svhn
 - --lr_task and --lr_rot which are the learning rates for task and scoring network
 -  --batch_size
 - --data_path the path to where the datasets are already present or should be downloaded to
-  Run the experiment by using the command-
`python Classification_PAL/main.py` or `python Segmentation_PAL/main.py`
- The accuracies of the task model trained after each query will be saved in a log file in the results folder

**Noisy Labels** - For reproducing PAL results with noisy labels on classification (CIFAR-10 or SVHN) or segmentation (Cityscapes) run the following command-

`python Noisy_label_classification_PAL/main.py`

or

`python Noisy_label_segmentation_PAL/main.py`

**Biased initial pool**- For reproducing results of PAL using a biased pool (with classes missing), on classification (SVHN) or segmentation (Cityscapes) run the following command-

`python Biased_pool_classification_PAL/main.py`

or

`python Biased_pool_segmentation_PAL/main.py`
