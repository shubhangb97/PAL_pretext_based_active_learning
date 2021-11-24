import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from custom_datasets import *
import copy
from helper import *
import torch.nn.functional as F
from metrics import StreamSegMetrics

class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.n_views=2
        self.temperature=0.07
        self.test_dataloader = test_dataloader
        self.metrics = StreamSegMetrics(self.args.num_classes)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.criterion=nn.CrossEntropyLoss()


    def rot_net_train(self,scoring_dataloader,scoring_model,val_dataloader,num_img):
        #--------- Trains the scoring network -----------------------

        if self.args.optim_Rot == "adam":
            optimizer = optim.Adam(scoring_model.parameters(), lr=0.001,weight_decay=10e-4)
        elif self.args.optim_Rot == "sgd":
            optimizer = torch.optim.SGD(params=[
                {'params': scoring_model.module.model.backbone.parameters(), 'lr': 0.1*self.args.lr_rot},
                {'params': scoring_model.module.model.classifier.parameters(), 'lr': self.args.lr_rot},
            ], lr=self.args.lr_rot, momentum=0.9, weight_decay=10e-4)

        test_scores = []

        for epoch in range(self.args.rot_train_epochs):
            running_loss = 0.0
            for i, data in enumerate(scoring_dataloader, 0):
                scoring_model.train()
                image0,images1, mask, index = data
                image0=image0.cuda()
                images1 = torch.cat(images1, dim=0)
                images1=images1.cuda()
                mask = mask.cuda()
                optimizer.zero_grad()

                output1,output2 = scoring_model(image0,images1)
                output1=output1['out']
                output2=output2['out']
                loss = self.ce_loss(output1, mask)
                logits,labels=self.info_nce_loss(output2)
                ss_loss=self.criterion(logits,labels)

                w1=self.args.train_loss_weightClassif
                w2=self.args.train_loss_weightRotation

                if self.args.train_loss_weightRotation == 0:
                    lossTotal=loss
                elif self.args.train_loss_weightClassif == 0:
                    lossTotal=ss_loss
                else:
                    lossTotal = w1 * loss +w2 * ss_loss
                lossTotal.backward()
                optimizer.step()

                _, preds = torch.max(output1, 1)
                targets_mask = mask != 255
                test_scores.append(np.mean((preds == mask)[targets_mask].data.cpu().numpy()))
                running_loss += loss.item()

                if i % 5 == 0 and i > 0:
                    print("Scoring training loss {} and accuracy {} in iteration {} and epoch {}".format(running_loss / i,
                                                                                            np.mean(test_scores), i, epoch))
                    running_loss = 0.0
                    test_scores = []
                    #print("Loss segmentation is "+str(loss.item())+" ss loss is "+str(lossTotal.item()- loss.item())+"\n")
            if(epoch%2==0):
                val_accuracy=self.validate_rot_net(scoring_model,val_dataloader)
                #print("Epoch "+str(epoch)+" mou here "+str(val_accuracy))

        return scoring_model



    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        features=torch.reshape(features,(features.shape[0],-1))
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0])#.to(self.args.device)
        labels=labels.long()
        labels=labels.cuda()

        logits = logits / self.temperature
        return logits, labels

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img


    def train(self, querry_dataloader, val_dataloader, task_model, num_img):
        print("Task Model training begins \n Number of iters to train= "+str(self.args.train_epochs)+" images-"+str(num_img))

        test_scores = []
        if self.args.optim_task == "adam":
            optimizer = optim.Adam(task_model.parameters(), lr=0.001,weight_decay=10e-4)
        elif self.args.optim_task == "sgd":
            optimizer = torch.optim.SGD(params=[
                {'params': task_model.module.model.backbone.parameters(), 'lr': 0.1*self.args.lr_task},
                {'params': task_model.module.model.classifier.parameters(), 'lr': self.args.lr_task},
            ], lr=self.args.lr_task, momentum=0.9, weight_decay=10e-4)
        best_model=task_model
        best_accuracy=0
        for epoch in range(self.args.train_epochs):
            running_loss = 0.0
            i=0
            for data in querry_dataloader:
                task_model.train()
                inputs, mask = data
                inputs = inputs.cuda()
                mask = mask.cuda()
                optimizer.zero_grad()

                output = task_model(inputs)
                output=output['out']
                loss = self.ce_loss(output, mask)
                loss.backward()
                optimizer.step()


                _, preds = torch.max(output, 1)
                targets_mask = mask != 255
                test_scores.append(np.mean((preds == mask)[targets_mask].data.cpu().numpy()))
                running_loss += loss.item()

                i=i+1
                if i % 10 == 0 and i > 0:
                    print("loss {} and accuracy {} in iteration {} and epoch {}".format(running_loss / i,
                                                                                            np.mean(test_scores), i, epoch))
                    running_loss = 0.0
                    test_scores = []

            if(epoch%2==0):
                val_accuracy=self.validate(task_model,val_dataloader)
                #print("Epoch "+str(epoch)+" miou here "+str(val_accuracy))
                if(val_accuracy>best_accuracy):
                    best_model=copy.deepcopy(task_model)
                    best_model.cuda()
                    best_accuracy=val_accuracy

        accuracy1=self.test(best_model)
        return accuracy1

    def validate(self, task_model, loader):
        task_model.eval()
        self.metrics.reset()
        for (imgs, label) in loader:
            with torch.no_grad():
                imgs = imgs.cuda()
                out = task_model(imgs)
                out= out['out']
                preds = out.detach().max(dim=1)[1].cpu().numpy()
                targets = label.cpu().numpy()
                self.metrics.update(targets, preds)

        val1= self.metrics.get_results()
        miou=val1['Mean IoU']
        print("Validation miou is "+str(miou)+"\n")
        return miou


    def validate_rot_net(self, scoring_model, loader):
        scoring_model.eval()
        self.metrics.reset()
        for (num1,data) in enumerate(loader):
            with torch.no_grad():
                image0,images1, mask, index = data
                image0=image0.cuda()
                images1 = torch.cat(images1, dim=0)
                images1=images1.cuda()
                mask = mask.cuda()
                output1,output2 = scoring_model(image0,images1)
                output1= output1['out']
                preds = output1.detach().max(dim=1)[1].cpu().numpy()
                targets = mask.cpu().numpy()
                self.metrics.update(targets, preds)

        val1= self.metrics.get_results()
        miou=val1['Mean IoU']
        print("Validation miou is "+str(miou)+"\n")
        return miou


    def kl_div(self,d1, d2):
        # """
        # Compute KL-Divergence between d1 and d2.
        # """
        dirty_logs = d1 * torch.log2(d1 / d2)
        return torch.sum(torch.where(d1 != 0, dirty_logs, torch.zeros_like(d1)), dim=1)


    def test(self, task_model):
        task_model.eval()
        self.metrics.reset()
        for (imgs, label) in self.test_dataloader:
            with torch.no_grad():
                imgs = imgs.cuda()
                out = task_model(imgs)
                out= out['out']
                preds = out.detach().max(dim=1)[1].cpu().numpy()
                targets = label.cpu().numpy()
                self.metrics.update(targets, preds)
        miou1= self.metrics.get_results()
        miou=miou1['Mean IoU']
        print("Test miou is "+str(miou)+"\n")
        return miou
