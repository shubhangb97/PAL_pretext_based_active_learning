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


class RotSampler:
    def __init__(self, budget,args):
        self.budget = budget
        self.args=args
        self.n_views=2
        self.temperature=0.07
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.criterion=nn.CrossEntropyLoss()


    def sample1(self, rot_net, data_loader, cuda,samplebatch,scores1):
        #Not used for segmentation final result
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

                #scoreClasses=scoreClasses+scoreClasses1

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

                if self.args.scoretype=="kl_based":

                    scoreT=self.kl_div(targets,preds)
                    scoreT1=torch.zeros_like(scoreClasses)

                    for i in range(scoreT1.shape[0]):
                        scoreT1[i]=scoreT[4*i]+scoreT[4*i+1]+scoreT[4*i+2]+scoreT[4*i+3]
                    score1=scoreT1

                elif self.args.scoretype=="probability":

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
            print("Sampling done of "+ str(len(indexfull)))
            rot_net.train()
            return indexfull

    def sample_query2(self, scoring_model, data_loader):
        all_preds = []
        all_indices = []

        scores=[]
        indexnum=[]
        with torch.no_grad():
            scoring_model.eval()

            for image0, images1, mask,index in data_loader:


                image0=image0.cuda()
                images1 = torch.cat(images1, dim=0)
                images1=images1.cuda()
                mask=mask.cuda()

                with torch.no_grad():
                    out1,out2=scoring_model(image0,images1)
                    out1=out1['out']
                    out2=out2['out']

                out1=F.softmax(out1,dim=1)
                logits,labels = self.info_nce_loss(out2)
                score_ss1=torch.zeros(self.args.batch_size)
                num2=0
                while(num2<self.args.batch_size):
                    score_ss1[num2]=logits[num2,0]
                    num2=num2+1
                mult_fact=1.0/self.args.num_classes
                targetClass=torch.ones_like(out1)*mult_fact

                scoreClasses=self.kl_div(targetClass,out1)

                scoreClasses=scoreClasses*-1
                scoreClasses=torch.mean(scoreClasses,dim=[1,2])
                score_ss=torch.zeros_like(scoreClasses)
                score_ss[:]=score_ss1[:]

                if self.args.lambda_kl == 0:
                    score2=score_ss
                elif self.args.lambda_rot == 0:
                    score2=scoreClasses
                else:
                    w1=self.args.lambda_rot
                    w2=self.args.lambda_kl
                    score2=w1 * score_ss+ w2 * scoreClasses
                scores.extend(score2.tolist())
                indexnum.extend(index.tolist())
            _,indice1=torch.topk(torch.FloatTensor(scores),self.budget)
            arr1=np.asarray(indexnum)
            indexfull=arr1[indice1]

            scoring_model.train()
            return indexfull

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        #labels = labels.cuda()#.to(self.args.device)
        #breakpoint()
        features=torch.reshape(features,(features.shape[0],-1))
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0])
        labels=labels.long()
        labels=labels.cuda()

        logits = logits / self.temperature
        return logits, labels


    def sample_query(self, rot_net,unlabeled_indices,current_indices,dataset1,val_dataloader):
        samplebatch=self.args.samplebatch_size
        iters=self.budget//samplebatch
        resi=self.budget-iters*samplebatch
        sampled_indices=[]
        sampled2=[]
        num1=0
        rot_net0=copy.deepcopy(rot_net)
        rot_net1=copy.deepcopy(rot_net)
        print("\n \n sampling")
        for iter1 in range(0,iters):
            print("small batch is \n"+str(num1))
            current_unlabeled_indices=np.setdiff1d(unlabeled_indices,sampled_indices)
            current_unlabeled_indices=list(current_unlabeled_indices)


            sampler_repeat=datautil.sampler.SubsetRandomSampler(current_unlabeled_indices)
            dataloader_repeat=datautil.DataLoader(dataset1,sampler=sampler_repeat,batch_size=self.args.batch_size,drop_last=False,num_workers=0)
            if(iter1==0):
                [sampled1,scores1]=self.sample2(rot_net1,dataloader_repeat,self.args.cuda,samplebatch)
            else:
                sampled1=self.sample1(rot_net1,dataloader_repeat,self.args.cuda,samplebatch,scores1)
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
            sampled1=self.sample1(rot_net1,dataloader_repeat,self.args.cuda,resi,scores1)
            sampled_indices=list(sampled_indices)+list(sampled1)

        return sampled_indices




    def kl_div(self,d1, d2):
        # """
        # Compute KL-Divergence between d1 and d2.
        # """
        dirty_logs = d1 * torch.log2(d1 / d2)
        return torch.sum(torch.where(d1 != 0, dirty_logs, torch.zeros_like(d1)), dim=1)# check !!
