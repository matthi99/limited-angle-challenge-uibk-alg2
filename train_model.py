# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 15:03:33 2022

@author: Schwab Matthias
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import DataLoader
from u_net import *
from datetime import datetime
from functions import *
from dataloader import *






    ######################    
    # Train first model  #
    ######################


device = "cuda" if torch.cuda.is_available() else "cpu"
LR=1e-4



for i in range(7,0,-1):
    cd_train = Dataset(difficulty=i)
    cd_test =Dataset(difficulty=i,train=False)
    collator = Collator()
    
    dataloader_train = DataLoader(cd_train, batch_size=4, collate_fn=collator, shuffle=True)
    dataloader_test = DataLoader(cd_test, batch_size=1, collate_fn=collator, shuffle=False)    
    net=UNet(1,1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    
    dice=DiceLoss()
    bce=nn.BCELoss()
    
    N_epochs=20
    best_loss=float('inf')
    print(len(dataloader_train))
    for epoch in range(N_epochs):
        net.train()
        print(f"Epoch {epoch}\n-------------------------------")
        count=0
        l=0
        for inp,outp in dataloader_train:
            count=count+1
            optimizer.zero_grad()
            
            discs=create_discs(inp.shape[0]).to(device)
            inp=inp+0.05*torch.randn_like(inp)
            
            pred=net(inp)*discs
                
            loss=dice(pred,outp)
            l+=loss.item()
            loss.backward()
                
            optimizer.step()
                
        plt.figure()
        plt.subplot(131)
        plt.imshow(inp[0,0,...].detach().cpu().numpy())
        plt.subplot(132)
        plt.imshow(pred[0,0,...].detach().cpu().numpy())
        plt.subplot(133)
        plt.imshow(outp[0,0,...].detach().cpu().numpy())
        plt.show() 
        print('loss:', l/count)
        l=0
        count=0
        for inp,outp in dataloader_test:
            with torch.no_grad():
                discs=create_discs(inp.shape[0]).to(device)
                pred=net(inp)*discs
                pred=(pred>0.5)*1
                loss=loss=dice(pred,outp)
                l+=loss.item()
                count+=1
                plt.figure()
                plt.subplot(131)
                plt.imshow(inp[0,0,...].detach().cpu().numpy())
                plt.subplot(132)
                plt.imshow(pred[0,0,...].detach().cpu().numpy())
                plt.subplot(133)
                plt.imshow(outp[0,0,...].detach().cpu().numpy())
                plt.show() 
        if (l/count)<best_loss:
            print("New best loss on testset: Dicecoeff=",l/count)
            best_loss=l/count
            torch.save(net.state_dict(), 'model_diff_'+str(i)+'_best.pt')
                
            
        
                
    # dt = datetime.now()        
    # index = '%d%d%d_%d%d'%(dt.year,dt.month,dt.day,dt.hour,dt.minute)
    # torch.save(net.state_dict(), 'model_diff_'+str(i)+'_%s.pt'%index)
    
    

