# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:47:55 2022

@author: matth
"""
import astra
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
import random
#import imageio
# time
#import pywt
#import cv2
from functions import *


    ######################    
    # calculate TV-reconstruction for given examples and difficulties #
    ######################

examples = ['solid_disc', 'ta', 'tb', 'tc', 'td']

alpha = 0.0001
L = 500
Niter = 500

#dont run this it tankes very long!
for i in range(8):
    for example in examples:
        CtData, fbp, mask = load_data(example)
        angles=np.arange(0, 360, 0.5) #starting angle=0 for all difficulties
        if i==0:
            anglesReduced=angles
        elif i==1:
            anglesReduced=angles[0:180]
        elif i==2:
            anglesReduced=angles[0:160]
        elif i==3:
            anglesReduced=angles[0:140]
        elif i==4:
            anglesReduced=angles[0:120]
        elif i==5:
            anglesReduced=angles[0:100]
        elif i==6:
            anglesReduced=angles[0:80]
        else:
            anglesReduced=angles[0:60]
            
        CtDataLimited = subsample_sinogram(CtData, anglesReduced)
        A_limited = create_ct_operator(CtDataLimited, 512)
        
        x0 = np.zeros((512,512))
        tv_rec, _ = tv(x0, A_limited, CtDataLimited['sinogram'], alpha, L, Niter, fbp)
        plt.figure()
        plt.imshow(tv_rec)
        plt.colorbar()
        plt.show()
        np.save('Data/tv_'+example+'_'+str(i)+'.npy', tv_rec)

#%%

    ######################    
    # Test performance of TV Reconstruction in combination with otsu thresholding #
    ######################

n = 512
startingAngle = 0
for exmaple in examples:
    for i in range(7,8):
        rec=np.load('Data/tv_'+example+'_'+str(i)+'.npy')
        rec=(rec-np.min(rec))/np.max(rec)*255
        otsu_threshold, rec_mask = cv2.threshold(rec.astype("uint8"), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        
        plt.figure()
        plt.imshow(rec)
        plt.colorbar()
        plt.title('TV-Reconstruction for difficulty '+str(i)+'.')
        plt.show()
        
        percentage_false_pixels=np.sum(abs(rec_mask.astype('int16')-mask.astype('int16')))/(n**2)
        print('Percentage of wrong pixels:', round(percentage_false_pixels*100,2), '%')

