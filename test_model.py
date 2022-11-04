# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 15:12:16 2022

@author: Schwab Matthias
"""

import astra
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
import random
import torch
import os
from torch.utils.data import  DataLoader
from u_net import *
from datetime import datetime
from functions import *
from dataloader import Dataset, Collator

    ##################################
    #test the model on example data  #
    ##################################

device = "cuda" if torch.cuda.is_available() else "cpu"

models=os.listdir('models/')
for i in range(7,0,-1):
    net=UNet(1,1).to(device)

    net.load_state_dict(torch.load('models/'+models[i-1]))
    net.eval()
    
    cd_test =Dataset(difficulty=i,train=False)
    collator = Collator()
    
    dataloader_test = DataLoader(cd_test, batch_size=1, collate_fn=collator, shuffle=False)    
    
    k=1
    for inp,outp in dataloader_test:
        with torch.no_grad():
            discs=create_discs(inp.shape[0]).to(device)
            pred=net(inp)*discs
            pred=(pred>0.5)*1
            ground_truth=outp[0,0,...].cpu().detach().numpy()
            prediction=pred[0,0,...].cpu().detach().numpy()
            tv_rec=inp[0,0,...].cpu().detach().numpy()
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(tv_rec)
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(prediction)
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(ground_truth)
            plt.axis('off')
            percentage_false_pixels=np.sum(abs(prediction.astype('int16')-ground_truth.astype('int16')))/(512**2)
            plt.title('Percentage of wrong pixels: ' +str(round(percentage_false_pixels*100,2))+ '%')
            plt.savefig('results/Difficulty_'+str(i)+'_example_'+str(k)+'.png', dpi=300,bbox_inches=None, pad_inches=0.1)
            plt.show()
            k=k+1
