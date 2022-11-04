# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:41:47 2022

@author: matth
"""
import astra
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
import random
from functions import *
from scipy import ndimage
import os

    ######################    
    # calculate right rescaling factor 
    ######################

#


CtData, fbp, mask_solid_disk = load_data('solid_disc')
sinogram=CtData['sinogram']

A = create_ct_operator(CtData, 512, 0,0)

#load tv rec of solid disk after 500 iterations (alpha=0.0001, L=500)
tv_rec=np.load('Data/tv_solid_disc_0.npy') 
tv_sinogram=np.reshape(A*tv_rec, sinogram.shape)

scale=np.mean(tv_rec[tv_rec>0.0015])
scaled_solid_disk=mask_solid_disk*scale



print('Error tv: ', np.mean((tv_sinogram-sinogram))**2)


np.save('Data/scaled_mask.npy', scaled_solid_disk)

#%%
    ######################    
    # Generate masks#
    ######################

#maybe copy code fom https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib

#generate masks with random circles in them

def random_circles(tv_rec,n):
    number=np.random.randint(1,n)
    mask=tv_rec.copy()
    binary_mask=np.zeros((512,512))
    binary_mask[mask>0.0015]=1
    
    for i in range(number):
        radius=np.random.randint(15,60)
        
        r=np.random.randint(0,240-radius)
        phi=np.random.uniform(0,2*np.pi)
        
        center_x=256+r*np.cos(phi)
        center_y=256+r*np.sin(phi)
        
        X, Y = np.ogrid[:512, :512]
        
        dist_from_center = (X - center_x)**2 + (Y - center_y)**2
    
        circular_mask = (dist_from_center > radius**2)
        mask=mask*circular_mask
        binary_mask=binary_mask*circular_mask
    return mask, binary_mask


#generate masks with random ellipses in them

def random_ellipses(tv_rec,n):
    number=np.random.randint(1,n)
    mask=tv_rec.copy()
    binary_mask=np.zeros((512,512))
    binary_mask[mask>0.0015]=1
    
    for i in range(number):
        a=np.random.randint(15,50)
        b=np.random.randint(15,50)
        
        r=np.random.randint(0,240-((a+b)/2))
        phi=np.random.uniform(0,2*np.pi)
        
        center_x=256+r*np.cos(phi)
        center_y=256+r*np.sin(phi)
        
        X, Y = np.ogrid[:512, :512]
        
        dist_from_center = (X - center_x)**2/(a**2) + (Y - center_y)**2/(b**2)
    
        elliptical_mask = (dist_from_center > 1)*1
        angle=random.randint(-90,90)
        elliptical_mask = scipy.ndimage.rotate(elliptical_mask, angle, reshape=False, cval=1)
        mask=mask*elliptical_mask
        binary_mask=binary_mask*elliptical_mask
    return mask, binary_mask
    

#generate masks with random rectangles in them

def random_rectangles(tv_rec,n):
    number=np.random.randint(1,n)
    mask=tv_rec.copy()
    binary_mask=np.zeros((512,512))
    binary_mask[mask>0.0015]=1
    
    for i in range(number):
        a=np.random.randint(10,50)
        b=np.random.randint(10,50)
        
        r=np.random.randint(0,240-((a+b)/2))
        phi=np.random.uniform(0,2*np.pi)
        
        center_x=256+r*np.cos(phi)
        center_y=256+r*np.sin(phi)
        
        X, Y = np.ogrid[:512, :512]
        
        dist_from_X = abs(X - center_x)
        dist_from_Y = abs(Y - center_y)
    
        rectangular_mask = ((dist_from_X >a)+(dist_from_Y >b))*1
        angle=random.randint(-90,90)
        rectangular_mask = scipy.ndimage.rotate(rectangular_mask, angle, reshape=False, cval=1)
        mask=mask*rectangular_mask
        binary_mask=binary_mask*rectangular_mask
    return mask, binary_mask


#%%


######################    
# Generate testing data#
######################
folder= "D:/Daten Holzkopf/Data-limited-angle/test/"


x0 = np.zeros((512,512))
alpha = 0.001
L = 500
Niter = 30

examples = ['solid_disc', 'ta', 'tb', 'tc', 'td']

for example in examples:
    data={}
    CtData, fbp, gt = load_data(example)
    noisy_sin=CtData['sinogram']
    data['gt']=gt
    data['sinogram']=noisy_sin
    for difficulty in range(8):
        starting_angle=0
        data['starting_angle']=starting_angle
        A_limited=create_ct_operator(CtData, 512, difficulty, starting_angle)
        rec, error = tv(x0, A_limited, subsample(noisy_sin, difficulty, starting_angle), alpha, L, Niter, gt)
        data['tv_rec_diff_'+str(difficulty)]=rec
        fbp=filtered_backprojection(subsample(noisy_sin, difficulty, starting_angle), A_limited)
        data['fbp_diff_'+str(difficulty)]=fbp
        
        plt.figure()
        plt.imshow(data['gt'])
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.imshow(data['tv_rec_diff_'+str(difficulty)])
        plt.colorbar()
        plt.show()
        
    np.save(folder+example+'.npy',data)
######################    
# Generate training data#
######################

folder= "D:/Daten Holzkopf/Data-limited-angle/train/"

Niter = 30

#Calculate noise level
sigma=np.sqrt(np.mean((tv_sinogram-sinogram))**2)

n_circle=500
for i in range(n_circle):
    data={}
    mask, gt=random_circles(tv_rec,8)
    sin=np.reshape(A*mask,sinogram.shape)
    noisy_sin=sin+np.random.normal(0,sigma,sin.shape)
    data['gt']=gt
    data['sinogram']=sin
    for difficulty in range(8):
        starting_angle=0
        data['starting_angle']=starting_angle
        A_limited=create_ct_operator(CtData, 512, difficulty, starting_angle)
        rec, error = tv(x0, A_limited, subsample(noisy_sin, difficulty, starting_angle), alpha, L, Niter, mask)
        data['tv_rec_diff_'+str(difficulty)]=rec
        fbp=filtered_backprojection(subsample(noisy_sin, difficulty, starting_angle), A_limited)
        data['fbp_diff_'+str(difficulty)]=fbp
    
        plt.figure()
        plt.imshow(data['gt'])
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.imshow(data['tv_rec_diff_'+str(difficulty)])
        plt.colorbar()
        plt.show()
    
    np.save(folder+'circle_'+str(i+1)+'.npy',data)
    
n_ellipse=500
for i in range(n_ellipse):
    data={}
    mask, gt=random_ellipses(tv_rec,9)
    sin=np.reshape(A*mask,sinogram.shape)
    noisy_sin=sin+np.random.normal(0,sigma,sin.shape)
    data['gt']=gt
    data['sinogram']=sin
    for difficulty in range(8):
        starting_angle=random_starting_angle(difficulty)
        data['starting_angle']=starting_angle
        A_limited=create_ct_operator(CtData, 512, difficulty, starting_angle)
        rec, error = tv(x0, A_limited, subsample(noisy_sin, difficulty, starting_angle), alpha, L, Niter, mask)
        data['tv_rec_diff_'+str(difficulty)]=rec
        fbp=filtered_backprojection(subsample(noisy_sin, difficulty, starting_angle), A_limited)
        data['fbp_diff_'+str(difficulty)]=fbp
    
        plt.figure()
        plt.imshow(data['gt'])
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.imshow(data['tv_rec_diff_'+str(difficulty)])
        plt.colorbar()
        plt.show()
    
    np.save(folder+'ellipse_'+str(i+1)+'.npy',data)
    

    
n_mixed=500


for i in range(n_mixed):
    data={}
    mask, gt=random_rectangles(tv_rec,7)
    mask, gt=random_circles(mask,7)
    mask, gt=random_ellipses(mask,7)
    sin=np.reshape(A*mask,sinogram.shape)
    noisy_sin=sin+np.random.normal(0,sigma,sin.shape)
    data['gt']=gt
    data['sinogram']=sin
    for difficulty in range(8):
        starting_angle=random_starting_angle(difficulty)
        data['starting_angle']=starting_angle
        A_limited=create_ct_operator(CtData, 512, difficulty, starting_angle)
        rec, error = tv(x0, A_limited, subsample(noisy_sin, difficulty, starting_angle), alpha, L, Niter, mask)
        data['tv_rec_diff_'+str(difficulty)]=rec
        fbp=filtered_backprojection(subsample(noisy_sin, difficulty, starting_angle), A_limited)
        data['fbp_diff_'+str(difficulty)]=fbp
    
        plt.figure()
        plt.imshow(data['gt'])
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.imshow(data['tv_rec_diff_'+str(difficulty)])
        plt.colorbar()
        plt.show()
    np.save(folder+'mixed_'+str(i+1)+'.npy',data)
    

n_rectangle=500
for i in range(69):
    data={}
    mask, gt=random_rectangles(tv_rec,9)
    sin=np.reshape(A*mask,sinogram.shape)
    noisy_sin=sin+np.random.normal(0,sigma,sin.shape)
    data['gt']=gt
    data['sinogram']=sin
    for difficulty in range(8):
        starting_angle=0
        data['starting_angle']=starting_angle
        A_limited=create_ct_operator(CtData, 512, difficulty, starting_angle)
        rec, error = tv(x0, A_limited, subsample(noisy_sin, difficulty, starting_angle), alpha, L, Niter, mask)
        data['tv_rec_diff_'+str(difficulty)]=rec
        fbp=filtered_backprojection(subsample(noisy_sin, difficulty, starting_angle), A_limited)
        data['fbp_diff_'+str(difficulty)]=fbp
    
        plt.figure()
        plt.imshow(data['gt'])
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.imshow(data['tv_rec_diff_'+str(difficulty)])
        plt.colorbar()
        plt.show()
    np.save(folder+'rectangle_'+str(i+1)+'.npy',data)
    
n_simon=500
mask_folder="D:/Daten Holzkopf/Data-limited-angle/arrays/"

filelist=os.listdir(mask_folder)
for i in range(n_simon):
    data={}
    gt=np.load(mask_folder+filelist[i])
    gt[gt==255]=0
    mask=tv_rec.copy()
    mask[gt==0]=0
    sin=np.reshape(A*mask,sinogram.shape)
    noisy_sin=sin+np.random.normal(0,sigma,sin.shape)
    data['gt']=gt
    data['sinogram']=sin
    for difficulty in range(8):
        starting_angle=0
        data['starting_angle']=starting_angle
        A_limited=create_ct_operator(CtData, 512, difficulty, starting_angle)
        rec, error = tv(x0, A_limited, subsample(noisy_sin, difficulty, starting_angle), alpha, L, Niter, mask)
        data['tv_rec_diff_'+str(difficulty)]=rec
        fbp=filtered_backprojection(subsample(noisy_sin, difficulty, starting_angle), A_limited)
        data['fbp_diff_'+str(difficulty)]=fbp
    
        plt.figure()
        plt.imshow(data['gt'])
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.imshow(data['tv_rec_diff_'+str(difficulty)])
        plt.colorbar()
        plt.show()
    np.save(folder+'simon'+str(i+1)+'.npy',data)
    