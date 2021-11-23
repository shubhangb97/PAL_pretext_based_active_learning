import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as datautil
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from custom_datasets import *
import copy
from custom_datasets import *


class RotSampler:
    def __init__(self, budget,args):
        self.budget = budget
        self.args=args
        self.ce_loss = nn.CrossEntropyLoss()
        self.criterion=nn.CrossEntropyLoss()


    def sample1(self, rot_net, data_loader, samplebatch,scores1):
        #sample with all 3 components in all iterations except first
        all_preds = []
        all_indices = []
        scores=[]
        indexnum=[]
        with torch.no_grad():
            rot_net.eval()

            for data,labelclass,label,index in data_loader:

                data=data.cuda()
                label=label.cuda()
                data=data.view(data.shape[0]*4,data.shape[2],data.shape[3],data.shape[4])
                label=label.view(label.shape[0]*4)


                with torch.no_grad():
                    preds,_=rot_net(data)
                    data2=data[0:-1:4,:,:,:]
                    _,predClasses=rot_net(data2)

                preds=F.softmax(preds,dim=1)

                predClasses=F.softmax(predClasses,dim=1)

                mult_fact=1.0/self.args.num_classes
                targetClass=torch.ones_like(predClasses)*mult_fact

                scoreClasses=self.kl_div(targetClass,predClasses)
                scoreClasses1=self.kl_div(predClasses,targetClass)

                scoreClasses=scoreClasses*-1
                scoreClasses1=scoreClasses1*-1

                targets=torch.zeros_like(preds)

                s1=targets.size()
                s1=s1[0]
                for i in range(s1):
                    if i%4==0:
                        targets[i][0]=1
                    elif i%4==1:
                        targets[i][1]=1
                    elif i%4==2:
                        targets[i][2]=1
                    elif i%4==3:
                        targets[i][3]=1

                scoreT=torch.zeros_like(scoreClasses)
                for i in range(scoreT.shape[0]):
                    scoreT[i]=preds[4*i,0]+preds[4*i+1,1]+preds[4*i+2,2]+preds[4*i+3,3]
                score1=scoreT

                score1=score1 * -1

                if self.args.lambda_kl == 0:
                    score2=score1
                elif self.args.lambda_rot == 0:
                    score2=scoreClasses
                else:
                    w1=self.args.lambda_rot
                    w2=self.args.lambda_kl
                    w3=self.args.lambda_div
                    scoresn=[]
                    index2=index.tolist()
                    for key in index2:
                        scoresn.append(scores1.get(key))
                    scoresn=torch.cuda.FloatTensor(scoresn)
                    score2=w3*score1 + scoresn

                scores.extend(score2.tolist())
                indexnum.extend(index.tolist())

            _,indice1=torch.topk(torch.FloatTensor(scores),samplebatch)
            arr1=np.asarray(indexnum)
            indexfull=arr1[indice1]
            rot_net.train()
            return indexfull

    def sample2(self, rot_net, data_loader, samplebatch):
        #sample without diversity score in first iteration
        all_preds = []
        all_indices = []

        scores=[]
        indexnum=[]
        with torch.no_grad():
            rot_net.eval()

            for data,labelclass,label,index in data_loader:


                data=data.cuda()
                label=label.cuda()

                data=data.view(data.shape[0]*4,data.shape[2],data.shape[3],data.shape[4])
                label=label.view(label.shape[0]*4)


                with torch.no_grad():
                    preds,_=rot_net(data)
                    data2=data[0:-1:4,:,:,:]
                    _,predClasses=rot_net(data2)

                preds=F.softmax(preds,dim=1)
                predClasses=F.softmax(predClasses,dim=1)

                mult_fact=1.0/self.args.num_classes
                targetClass=torch.ones_like(predClasses)*mult_fact

                scoreClasses=self.kl_div(targetClass,predClasses)
                scoreClasses1=self.kl_div(predClasses,targetClass)

                scoreClasses=scoreClasses*-1
                scoreClasses1=scoreClasses1*-1

                targets=torch.zeros_like(preds)

                s1=targets.size()
                s1=s1[0]
                for i in range(s1):
                    if i%4==0:
                        targets[i][0]=1
                    elif i%4==1:
                        targets[i][1]=1
                    elif i%4==2:
                        targets[i][2]=1
                    elif i%4==3:
                        targets[i][3]=1

                scoreT=torch.zeros_like(scoreClasses)
                for i in range(scoreT.shape[0]):
                    scoreT[i]=preds[4*i,0]+preds[4*i+1,1]+preds[4*i+2,2]+preds[4*i+3,3]
                score1=scoreT

                score1=score1 * -1

                if self.args.lambda_kl == 0:
                    score2=score1
                elif self.args.lambda_rot == 0:
                    score2=scoreClasses
                else:
                    w1=self.args.lambda_rot
                    w2=self.args.lambda_kl
                    score2=w1 * score1+ w2 * scoreClasses

                scores.extend(score2.tolist())
                indexnum.extend(index.tolist())

            _,indice1=torch.topk(torch.FloatTensor(scores),samplebatch)
            arr1=np.asarray(indexnum)
            indexfull=arr1[indice1]
            scores1=dict(zip(indexnum,scores))

            rot_net.train()
            return [indexfull,scores1]


    def sample_query(self, rot_net,unlabeled_indices,current_indices,dataset1,val_dataloader):
        samplebatch=self.args.samplebatch_size
        iters=self.budget//samplebatch
        resi=self.budget-iters*samplebatch
        sampled_indices=[]
        sampled2=[]
        num1=0
        rot_net0=copy.deepcopy(rot_net)
        rot_net1=copy.deepcopy(rot_net)
        print("\n \n Sampling")
        for iter1 in range(0,iters):
            print("Sub Query is \n"+str(num1))
            current_unlabeled_indices=np.setdiff1d(unlabeled_indices,sampled_indices)
            current_unlabeled_indices=list(current_unlabeled_indices)


            sampler_repeat=datautil.sampler.SubsetRandomSampler(current_unlabeled_indices)
            dataloader_repeat=datautil.DataLoader(dataset1,sampler=sampler_repeat,batch_size=self.args.batch_size,drop_last=False,num_workers=0)
            if(iter1==0):
                [sampled1,scores1]=self.sample2(rot_net1,dataloader_repeat,samplebatch)
            else:
                sampled1=self.sample1(rot_net1,dataloader_repeat,samplebatch,scores1)
            sampled2=list(sampled2)+list(sampled1)

            sampled_indices=list(sampled_indices)+list(sampled1)
            rot_net1=copy.deepcopy(rot_net)
            rot_net1=self.finetunesmall(rot_net1,sampled_indices,dataset1,val_dataloader,rot_net0)
            num1=num1+1

        current_unlabeled_indices=np.setdiff1d(unlabeled_indices,sampled_indices)
        current_unlabeled_indices=list(current_unlabeled_indices)
        if(resi>0):
            sampler_repeat=datautil.sampler.SubsetRandomSampler(current_unlabeled_indices)
            dataloader_repeat=datautil.DataLoader(dataset1,sampler=sampler_repeat,batch_size=self.args.batch_size,drop_last=False,num_workers=0)
            sampled1=self.sample1(rot_net1,dataloader_repeat,resi,scores1)
            sampled_indices=list(sampled_indices)+list(sampled1)

        return sampled_indices



    def finetunesmall(self,rot_net,sampled,dataset1,val_dataloader,rot_net0):
        #Fine tune only self-supervsion head for diversity score generation
        sampler=datautil.sampler.SubsetRandomSampler(sampled)
        dataloader_repeat=datautil.DataLoader(dataset1,sampler=sampler,batch_size=self.args.batch_size,drop_last=False,num_workers=0)
        rot_learning_rate=self.args.lr_rot*0.1
        rot_epochs=5
        params1=rot_net.parameters()

        if self.args.optim_Rot == "sgd":
            if self.args.dataset=="caltech256":
                optim_rot_net = optim.SGD(filter(lambda x: x.requires_grad,params1), lr=rot_learning_rate)
            else:
                optim_rot_net = optim.SGD(filter(lambda x: x.requires_grad,params1), lr=rot_learning_rate)
        elif self.args.optim_Rot == "adam":
            if self.args.dataset=="caltech256":
                optim_rot_net = optim.Adam(filter(lambda x: x.requires_grad,params1), lr=rot_learning_rate)
            else:
                optim_rot_net=torch.optim.Adam(params1,lr=rot_learning_rate)

        rot_net.cuda()
        rot_net.train()

        for epoch in range(0,rot_epochs):
            for data,labelclass,label,index in dataloader_repeat:

                data=data.cuda()
                label=label.cuda()
                data=data.view(data.shape[0]*4,data.shape[2],data.shape[3],data.shape[4])
                label=label.view(label.shape[0]*4)


                preds,_=rot_net(data)
                lossT=self.criterion(preds,label)

                data2=data[0:-1:4,:,:,:]
                _,predClasses=rot_net(data2)
                lossTotal=lossT
                optim_rot_net.zero_grad()
                lossTotal.backward()

                optim_rot_net.step()

            #print("Epoch number "+str(epoch)+" Loss "+str(lossTotal.item())+" loss rotation "+str(lossTotal.item()))

            if epoch%1 == 0:
                acc1=self.validate_rot_net(rot_net,val_dataloader)
                #print("Loss at epoch "+str(epoch)+" is "+str(acc1.item()))

        best_model=copy.deepcopy(rot_net)
        best_model.cuda()
        return best_model

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
            lossTotal+=self.args.val_loss_weightRotation*lossT + self.args.val_loss_weightClassif*lossClass

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

        #print("Test accuracy "+str((correctclass / totalclass) * 100))
        #print("Rotation accuracy is "+str((correct / total) * 100))
        print("Validation Loss "+str(lossTotal.item())+" validation loss rotation "+str(lossRotTot.item())+"validation loss classificatio "+str(lossClTot.item()))
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
