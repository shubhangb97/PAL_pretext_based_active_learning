import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from custom_datasets import *
import copy





class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader

        self.ce_loss = nn.CrossEntropyLoss()
        self.criterion=nn.CrossEntropyLoss()


    def rot_net_train(self,rot_querry_dataloader,rot_net,val_dataloader,split):
        state= {}
        rot_learning_rate = self.args.lr_rot
        rot_epochs = self.args.rot_train_epochs
        params1=rot_net.parameters()

        if self.args.optim_Rot == "sgd":
            if (self.args.dataset=="caltech256"):
                optim_rot_net = optim.SGD([{'params': rot_net.features.parameters()},{'params': rot_net.layer1.parameters(),'lr':0.01},{'params':rot_net.layer2.parameters(),'lr':0.01}], lr=rot_learning_rate, momentum=0.9,weight_decay = 5e-4,nesterov=True)
            else:
                optim_rot_net = optim.SGD(params1, lr=rot_learning_rate, momentum=0.9,weight_decay = 5e-4,nesterov=True)
        elif self.args.optim_Rot == "adam":
            optim_rot_net=torch.optim.Adam(params1,lr=rot_learning_rate)

        if self.args.scheduler_Rot == "cosine":
            scheduler_rot_model = optim.lr_scheduler.CosineAnnealingLR(optim_rot_net,
                                                                        T_max=rot_epochs,
                                                                        eta_min=1e-5,
                                                                        last_epoch=-1)
        elif self.args.scheduler_Rot == "decay_step":
            lr_change = rot_epochs // 4

        rot_net.cuda()
        if self.args.valtype ==  "loss":
            acc1=np.inf
            best_acc1=np.inf
        elif self.args.valtype == "accuracy":
            acc1=0
            best_acc1=0

        for epoch in range(0,rot_epochs):

            #Scheduling for SGD only
            if( (self.args.scheduler_Rot == "decay_step") and not(self.args.optim_Rot == 'adam')):
                if epoch is not 0 and epoch % lr_change == 0:
                    for param in optim_rot_net.param_groups:
                        param['lr'] = param['lr'] / 10
            rot_net.train()
            for data,labelclass,label,index in rot_querry_dataloader:

                data=data.cuda()
                label=label.cuda()
                labelclass=labelclass.cuda()
                data=data.view(data.shape[0]*4,data.shape[2],data.shape[3],data.shape[4])
                label=label.view(label.shape[0]*4)

                data2=data[0:-1:4,:,:,:]
                _,predClasses=rot_net(data2)

                preds,_=rot_net(data)
                lossT=self.criterion(preds,label)


                lossClass=self.criterion(predClasses,labelclass)

                w1=self.args.train_loss_weightRotation
                w2=self.args.train_loss_weightClassif

                if self.args.train_loss_weightRotation == 0:
                    lossTotal=lossClass
                elif self.args.train_loss_weightClassif == 0:
                    lossTotal=lossT
                else:
                    lossTotal = w1 * lossT +w2 * lossClass

                optim_rot_net.zero_grad()
                lossTotal.backward()

                optim_rot_net.step()
                if self.args.scheduler_Rot == "cosine":
                    scheduler_rot_model.step()

            print("Epoch number "+str(epoch)+" Loss "+str(lossTotal.item())+" loss rotation "+str(lossTotal.item()-lossClass.item())+" loss classification "+str(lossClass.item()))
            if epoch%5 == 0:
                acc1=self.validate_rot_net(rot_net,val_dataloader)
                print("Loss at epoch "+str(epoch)+" is "+str(acc1.item()))
                if self.args.valtype == "loss":
                    if acc1 <= best_acc1:
                        best_acc1=acc1
                        print("Saving weights at epoch "+str(epoch)+" best Validation loss value is "+str(best_acc1.item()))
                        best_model=copy.deepcopy(rot_net)
                        best_model.cuda()
                elif self.args.valtype == "accuracy":
                    if acc1 >= best_acc1:
                        best_acc1=acc1
                        print("Saving weights at epoch "+str(epoch)+" best Validation loss value is "+str(best_acc1.item()))
                        best_model=copy.deepcopy(rot_net)
                        best_model.cuda()
        return best_model


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img
    def read_data1(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _,_ in dataloader:
                    img=img.view(img.shape[0]*4,img.shape[2],img.shape[3],img.shape[4])
                    img2=img[0:-1:4,:,:,:]
                    yield img2, label
        else:
            while True:
                for img, _, _,_ in dataloader:
                    img=img.view(img.shape[0]*4,img.shape[2],img.shape[3],img.shape[4])
                    img2=img[0:-1:4,:,:,:]
                    yield img2


    def train(self, querry_dataloader, val_dataloader, task_model, unlabeled_dataloader,num_img):
        #trains task network to help evaluate labeled dataset
        self.args.train_iterations = (num_img * self.args.train_epochs) // self.args.batch_size
        self.iters_per_epoch = num_img//self.args.batch_size
        print("Task Model training begins \n Number of iters to train= "+str(self.args.train_iterations)+" images-"+str(num_img))


        task_model_learning=self.args.lr_task
        params1=task_model.parameters()
        if self.args.optim_task == "sgd":
            if (self.args.dataset=="caltech256"):
                optim_task_model = optim.SGD([{'params': task_model.vgg16_1.features.parameters()},{'params': task_model.classifier.parameters(),'lr':0.01}], lr=task_model_learning, weight_decay=5e-4, momentum=0.9)
            else:
                optim_task_model = optim.SGD(task_model.parameters(), lr=task_model_learning, weight_decay=5e-4, momentum=0.9)
        elif self.args.optim_task == "adam":
            optim_task_model=torch.optim.Adam(params1,lr=task_model_learning,weight_decay=10e-4)

        if self.args.scheduler_task == "cosine":
            scheduler_task_model = optim.lr_scheduler.CosineAnnealingLR(optim_task_model,
                                                                        T_max=self.args.train_epochs,
                                                                        eta_min=1e-5,
                                                                        last_epoch=-1)
        elif self.args.scheduler_task == "decay_step":
            lr_change = self.args.train_iterations // 4

        labeled_data = self.read_data1(querry_dataloader)
        unlabeled_data = self.read_data1(unlabeled_dataloader, labels=False)

        task_model.train()

        task_model = task_model.cuda()

        best_acc = 0
        for iter_count in range(self.args.train_iterations):
            if self.args.scheduler_task == "decay_step":
                if iter_count is not 0 and iter_count % lr_change == 0:
                    for param in optim_task_model.param_groups:
                        param['lr'] = param['lr'] / 10
            labeled_imgs, labels= next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()
            labels = labels.cuda()

            # task_model step
            preds = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()
            if self.args.scheduler_task == "cosine":
                if (iter_count+1)%self.iters_per_epoch == 0:
                    scheduler_task_model.step()

            if iter_count % 100 == 0:
                print('Current training iteration: {}'.format(iter_count))
                print('Current task model loss: {:.4f}'.format(task_loss.item()))


            if iter_count % 500 == 0:
                acc = self.validate(task_model, val_dataloader)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(task_model)

                print('current step: {} acc: {}'.format(iter_count, acc))
                print('best acc: ', best_acc)
        best_model = best_model.cuda()

        final_accuracy = self.test(best_model)
        return final_accuracy



    def validate(self, task_model, loader):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels, _,_ in loader:
            imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
            if(self.args.dataset != 'svhn'):
                task_model.train()
        return correct / total * 100


    def validate_rot_net(self, rot_net, loader):
        rot_net.eval()
        total, correct = 0, 0

        totalclass=0
        correctclass=0

        lossTotal=0
        lossClTot=0
        lossRotTot=0

        for data,labelclass,label,index in loader:
            data=data.view(data.shape[0]*4,data.shape[2],data.shape[3],data.shape[4])
            label=label.view(label.shape[0]*4)


            data=data.cuda()
            label=label.cuda()
            labelclass=labelclass.cuda()

            with torch.no_grad():
                preds,_=rot_net(data)
                data2=data[0:-1:4,:,:,:]
                _,predClasses=rot_net(data2)

            lossT=self.criterion(preds,label)

            lossRotTot+=lossT

            lossClass=self.criterion(predClasses,labelclass)

            w1=self.args.val_loss_weightRotation
            w2=self.args.val_loss_weightClassif
            lossClTot+=lossClass
            lossTotal+=self.args.val_loss_weightRotation*lossT + self.args.val_loss_weightClassif*lossClass #+0.05*klDiv1


            preds=torch.argmax(preds,dim=1).cpu()
            predClasses=torch.argmax(predClasses,dim=1).cpu()


            preds=preds.numpy()
            predClasses=predClasses.numpy()


            label=label.cpu().numpy()
            labelclass=labelclass.cpu().numpy()


            correct+= accuracy_score(label,preds,normalize=False)
            correctclass+= accuracy_score(labelclass,predClasses,normalize=False)

            totalclass += data2.size(0)
            total += data.size(0)
        print("Test accuracy is "+str((correctclass / totalclass) * 100))
        print("Rotation accuracy is "+str((correct / total) * 100))
        print("Validation Loss "+str(lossTotal.item())+" validation loss rotation "+str(lossRotTot.item())+"validation loss classification "+str(lossClTot.item()))
        if self.args.valtype == "loss":
            rot_net.train()
            return lossTotal
        elif self.args.valtype == "accuracy":
            rot_net.train()
            return ((correct / total) * 100)
        else:
            print("Give proper valtype arg")
            rot_net.train()
            return lossTotal


    def kl_div(self,d1, d2):
        # """
        # Compute KL-Divergence between d1 and d2.
        # """
        dirty_logs = d1 * torch.log2(d1 / d2)
        return torch.sum(torch.where(d1 != 0, dirty_logs, torch.zeros_like(d1)), dim=1)


    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        if (self.args.dataset=="caltech256" or self.args.dataset=="caltech101"):
            for imgs, labels,_ in self.test_dataloader:
                imgs = imgs.cuda()

                with torch.no_grad():
                    preds = task_model(imgs)

                preds = torch.argmax(preds, dim=1).cpu().numpy()
                correct += accuracy_score(labels, preds, normalize=False)
                total += imgs.size(0)
        else:
            for imgs, labels in self.test_dataloader:
                imgs = imgs.cuda()

                with torch.no_grad():
                    preds = task_model(imgs)

                preds = torch.argmax(preds, dim=1).cpu().numpy()
                correct += accuracy_score(labels, preds, normalize=False)
                total += imgs.size(0)
        return correct / total * 100
