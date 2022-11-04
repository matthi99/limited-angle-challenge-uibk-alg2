# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 08:47:19 2022

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
import random
from scipy import ndimage
import argparse
import matplotlib.image


parser = argparse.ArgumentParser(description= 'Define paths and difficulty ')

parser.add_argument('inputFolder', type=str)
parser.add_argument('outputFolder', type=str)
parser.add_argument('difficulty', type=int)

args = parser.parse_args()


def main(inputFolder,outputFolder,difficulty):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    net=UNet(1,1).to(device)
    net.load_state_dict(torch.load('models/model_diff_'+str(difficulty)+'.pt'))
    net.eval()
    
    tv_disc=np.load('Data/tv_solid_disc_0.npy')
    scale=np.mean(tv_disc[tv_disc>0.0015])
    
    inputfiles=os.listdir(inputFolder)
    for file in inputfiles:
        CtData = load(inputFolder+'/'+file)
        
        A=create_operator(CtData, 512,0)
        sinogram=CtData['sinogram']
        angles= CtData['parameters']['angles']
        starting_angle = CtData['parameters']['angles'][0]
        
        A_limited=create_operator(CtData, 512, difficulty)
        
        x0 = np.zeros((512,512))
        alpha = 0.001
        L = 500
        Niter = 30
        
        tv_rec, _ = tv(x0, A_limited, sinogram, alpha, L, Niter, tv_disc)
        
        
        tv_rec=(tv_rec-np.min(tv_rec))/(np.max(tv_rec)-np.min(tv_rec))
        tv_rec=np.expand_dims(tv_rec,0)
        tv_rec=np.expand_dims(tv_rec,0)
        tv_rec = torch.from_numpy(tv_rec.astype("float32")).to(device)
        
        disc=create_discs(tv_rec.shape[0]).to(device)
        
        net_rec=net(tv_rec)*disc
        net_rec=((net_rec>0.5)*1)[0,0,...].cpu().detach().numpy()
        
        net_rec= scipy.ndimage.rotate(net_rec, starting_angle, reshape=False, cval=0)
        scaled_rec=tv_disc.copy()
        scaled_rec[net_rec==0]=0
        sin_rec=np.reshape(A*scaled_rec,(720,560))
        sin_rec[int(starting_angle):int(starting_angle+len(angles))]=sinogram
        
        final_rec,_=tv(x0,A,sin_rec,0.001,500,300,tv_disc)
        final_rec = final_rec/scale
        final_rec[final_rec<0.5] = 0
        final_rec[final_rec>=0.5] = 1
        
        matplotlib.image.imsave(outputFolder+'/'+file[:-4]+'.png',final_rec, cmap="gray")
        

main(args.inputFolder, args.outputFolder, args.difficulty)