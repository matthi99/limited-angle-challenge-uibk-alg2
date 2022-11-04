# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:35:51 2022

@author: Schwab Matthias
"""
import astra
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
import random
from scipy.fftpack import fftfreq
import torch

def load_data(data):
    mat = scipy.io.loadmat('Examples/htc2022_' + data + '_full.mat')
    fbp=scipy.io.loadmat('Examples/htc2022_' + data + '_full_recon_fbp.mat')['reconFullFbp']
    mask=scipy.io.loadmat('Examples/htc2022_' + data + '_full_recon_fbp_seg.mat')['reconFullFbpSeg']
    sinogram = mat['CtDataFull']['sinogram'][0][0]

    parameters = {}
    keys = ['projectName','scanner','measurers','date','dateFormat','geometryType','distanceSourceOrigin','distanceSourceDetector','distanceUnit','geometricMagnification','numberImages','angles','detector','detectorType','binning','pixelSize','exposureTime','exposureTimeUnit','tube','target','voltage','voltageUnit','current','currentUnit','xRayFilter','xRayFilterThickness','detectorRows','detectorCols','freeRayAtDetector','binningPost','pixelSizePost','effectivePixelSizePost','numDetectorsPost']
    vals = mat['CtDataFull']['parameters'][0][0][0][0]
    for i in range(len(vals)):
        if isinstance(vals[i][0], str):
            parameters[keys[i]] = vals[i][0]
        else:
            if vals[i][0].shape[0] > 1:
                parameters[keys[i]] = vals[i][0]
            else:
                parameters[keys[i]] = vals[i][0][0]


    CtData = dict(zip(['type', 'sinogram', 'parameters'], ['2d', sinogram, parameters]))
    return CtData, fbp, mask


def load(path):
    mat = scipy.io.loadmat(path)
    sinogram = mat['CtDataLimited']['sinogram'][0][0]

    parameters = {}
    keys = ['projectName','scanner','measurers','date','dateFormat','geometryType','distanceSourceOrigin','distanceSourceDetector','distanceUnit','geometricMagnification','numberImages','angles','detector','detectorType','binning','pixelSize','exposureTime','exposureTimeUnit','tube','target','voltage','voltageUnit','current','currentUnit','xRayFilter','xRayFilterThickness','detectorRows','detectorCols','freeRayAtDetector','binningPost','pixelSizePost','effectivePixelSizePost','numDetectorsPost']
    vals = mat['CtDataLimited']['parameters'][0][0][0][0]
    for i in range(len(vals)):
        if isinstance(vals[i][0], str):
            parameters[keys[i]] = vals[i][0]
        else:
            if vals[i][0].shape[0] > 1:
                parameters[keys[i]] = vals[i][0]
            else:
                parameters[keys[i]] = vals[i][0][0]


    CtData = dict(zip(['type', 'sinogram', 'parameters'], ['2d', sinogram, parameters]))
    return CtData


def create_ct_operator(CtData, n, difficulty, starting_angle):
    DSD             = CtData['parameters']['distanceSourceDetector']
    DSO             = CtData['parameters']['distanceSourceOrigin']
    M               = CtData['parameters']['geometricMagnification']
    angles          = CtData['parameters']['angles']
    numDetectors    = CtData['parameters']['numDetectorsPost']
    effPixel        = CtData['parameters']['effectivePixelSizePost']

    # Distance from origin to detector
    DOD             = DSD - DSO
    # Distance from source to origin specified in terms of effective pixel size
    DSO             = DSO / effPixel
    # Distance from origin to detector specified in terms of effective pixel size
    DOD             = DOD / effPixel
    
    if difficulty==0:
        angles=angles
    elif difficulty==1:
        angles=angles[2*starting_angle:2*starting_angle+181]
    elif difficulty==2:
        angles=angles[2*starting_angle:2*starting_angle+161]
    elif difficulty==3:
        angles=angles[2*starting_angle:2*starting_angle+141]
    elif difficulty==4:
        angles=angles[2*starting_angle:2*starting_angle+121]
    elif difficulty==5:
        angles=angles[2*starting_angle:2*starting_angle+101]
    elif difficulty==6:
        angles=angles[2*starting_angle:2*starting_angle+81]
    elif difficulty==7:
        angles=angles[2*starting_angle:2*starting_angle+61]
    else:
        print("wrong difficulty")
    
def create_operator(CtData, n, difficulty):
    DSD             = CtData['parameters']['distanceSourceDetector']
    DSO             = CtData['parameters']['distanceSourceOrigin']
    M               = CtData['parameters']['geometricMagnification']
    angles          = np.arange(0, 360, 0.5)
    numDetectors    = CtData['parameters']['numDetectorsPost']
    effPixel        = CtData['parameters']['effectivePixelSizePost']

    # Distance from origin to detector
    DOD             = DSD - DSO
    # Distance from source to origin specified in terms of effective pixel size
    DSO             = DSO / effPixel
    # Distance from origin to detector specified in terms of effective pixel size
    DOD             = DOD / effPixel
    
    if difficulty==0:
        angles=angles
    elif difficulty==1:
        angles=angles[0:+181]
    elif difficulty==2:
        angles=angles[0:161]
    elif difficulty==3:
        angles=angles[0:141]
    elif difficulty==4:
        angles=angles[0:121]
    elif difficulty==5:
        angles=angles[0:101]
    elif difficulty==6:
        angles=angles[0:81]
    elif difficulty==7:
        angles=angles[0:61]
    else:
        print("wrong difficulty")    

    # ASTRA uses angles in radians
    anglesRad = angles*np.pi/180

    volumeGeometry = astra.create_vol_geom(n, n)
    projectionGeometry = astra.create_proj_geom('fanflat', M, numDetectors, anglesRad, DSO, DOD)

    proj_id = astra.create_projector('cuda', projectionGeometry, volumeGeometry)
    
    A = astra.OpTomo(proj_id)
    return A

def random_starting_angle(difficulty):
    if difficulty==0:
        return 0
    elif difficulty==1:
        starting_angle=random.randint(0, 360-90)
        return starting_angle
    elif difficulty==2:
        starting_angle=random.randint(0, 360-80)
        return starting_angle
    elif difficulty==3:
        starting_angle=random.randint(0, 360-70)
        return starting_angle
    elif difficulty==4:
        starting_angle=random.randint(0, 360-60)
        return starting_angle
    elif difficulty==5:
        starting_angle=random.randint(0, 360-50)
        return starting_angle
    elif difficulty==6:
        starting_angle=random.randint(0, 360-40)
        return starting_angle
    elif difficulty==7:
        starting_angle=random.randint(0, 360-30)
        return starting_angle
    else:
        return("wrong difficulty")

def ismembertol(X, Y):
    # i = 0
    # idx = np.zeros_like(X, dtype=int)
    tol = 1e-6
    idx = np.where( abs(X-np.array(Y)[:,None]) <= tol)[0]
    # for x in X:
    #     for y in Y:
    #         if abs(x-y) <= tol:
    #             idx[i] = 1
    #             break
        # i += 1
    return idx

def subsample_sinogram(CtData, anglesReduced):
    idx = ismembertol(CtData['parameters']['angles'], anglesReduced)
    
    CtDataSubsampled                               = CtData.copy()
    CtDataSubsampled['sinogram']                   = CtData['sinogram'][idx, :]
    CtDataSubsampled['parameters']['numberImages'] = np.sum(idx)
    CtDataSubsampled['parameters']['angles']       = CtData['parameters']['angles'][idx]
    return CtDataSubsampled

def soft_thresh(x, alpha):
    return np.sign(x)*np.maximum(0, np.abs(x)-alpha)

def my_grad(X):
    fx = np.concatenate((X[1:,:],np.expand_dims(X[-1,:], axis=0)), axis=0) - X
    fy = np.concatenate((X[:,1:],np.expand_dims(X[:,-1], axis=1)), axis=1) - X
    return fx, fy

def my_div(Px, Py):
    fx = Px - np.concatenate((np.expand_dims(Px[0,:], axis=0), Px[0:-1,:]), axis=0)
    fx[0,:] = Px[0,:]
    fx[-1,:] = -Px[-2,:]
    
    fy = Py - np.concatenate((np.expand_dims(Py[:,0], axis=1), Py[:,0:-1]), axis=1)
    fy[:,0] = Py[:,0]
    fy[:,-1] = -Py[:,-2]
    return fx + fy

def tv(x0, A, g, alpha, L, Niter, f, multi=False):
    tau = 1/L
    sigma = 1/L
    theta = 1
    
    g=g.flatten()
    f = f.flatten()
    grad_scale = 1e+2
    n    = x0.shape[0]
    p    = np.zeros_like(g).flatten()
    qx   = x0
    qy   = x0
    u    = x0.flatten()
    ubar = x0.flatten()


    error = []
    recs=[]
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    for k in range(Niter):
        p  = (p + sigma*(A.dot(ubar) - g))/(1+sigma)
        [ubarx, ubary] = my_grad(np.reshape(ubar, [n,n]))
        qx = alpha*(qx + grad_scale*sigma*ubarx)/np.maximum(alpha, np.abs(qx + grad_scale*sigma*ubarx)) 
        qy = alpha*(qy + grad_scale*sigma*ubary)/np.maximum(alpha, np.abs(qy + grad_scale*sigma*ubary))
        
        uiter = np.maximum(0, u - tau*(A.transpose().dot(p) - grad_scale*my_div(qx, qy).flatten()))
        
        # uiter = soft_thresh(u - tau*(A.transpose().dot(p) - grad_scale*my_div(qx, qy).flatten()), 1e-3)
        
        ubar = uiter + theta*(uiter - u)
        u = uiter
        error.append(np.sum(abs(ubar - f)**2)/np.sum(abs(f)**2))
        if multi==True:
            if k in range(Niter-5,Niter,1):
                recs.append(np.reshape(ubar, [n, n]))
    rec=np.reshape(ubar, [n, n])
        
        # ax1.imshow(np.reshape(u, [n,n]))
        # ax2.plot(error)
        # plt.pause(0.000005)
        #print('Iteration: ' + str(k) + '/' + str(Niter) + ', Error: ' + str(error[k]))
    if multi==True:
        return recs, error
    else:
        return rec, error
    
    return recs, error

def subsample(sinogram, difficulty, starting_angle):
    if difficulty==0:
        return sinogram
    elif difficulty==1:
        return sinogram[2*starting_angle:2*starting_angle+180]
    elif difficulty==2:
        return sinogram[2*starting_angle:2*starting_angle+160]
    elif difficulty==3:
        return sinogram[2*starting_angle:2*starting_angle+140]
    elif difficulty==4:
        return sinogram[2*starting_angle:2*starting_angle+120]
    elif difficulty==5:
        return sinogram[2*starting_angle:2*starting_angle+100]
    elif difficulty==6:
        return sinogram[2*starting_angle:2*starting_angle+80]
    elif difficulty==7:
        return sinogram[2*starting_angle:2*starting_angle+60]
    else:
        return print("wrong difficulty")
    
    
def ram_lak_filtering(g):
    g = g.transpose()
    filt_len = g.shape[0]      
    if np.mod(filt_len, 2) != 0:
        filt_len += 1
    
    freq = fftfreq(filt_len).reshape(-1, 1)
    fou_filt = 2 * np.abs(freq)                       # ramp filter
    
    # if np.mod(filt_len, 3) != 0:
    #     fou_filt = fou_filt[1:]
    
    fou_filt = np.squeeze(fou_filt)
    gfilt = np.zeros_like(g)
    for i in range(g.shape[1]):
        aux = np.fft.fft(g[:,i])*fou_filt
        gfilt[:,i] = np.fft.ifft(aux)
# 
    gfilt = gfilt.transpose()
    return gfilt

def filtered_backprojection(sinogram, A):
    filtered = ram_lak_filtering(sinogram)
    rec = A.transpose()*(filtered.flatten())
    rec = np.reshape(rec,[512,512])
    return rec

def create_discs(batch_size):
    _, _, disc = load_data('solid_disc')
    discs=np.zeros((batch_size,1,512,512))
    for i in range(batch_size):
        discs[i,0,...]=disc
    discs=torch.from_numpy(discs.astype("float32"))
    return discs


class DiceLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = kwargs.get("smooth", 1.)

    def forward(self, predictions, targets):
        
        
        #flatten label and prediction tensors
        bs=predictions.size(0)
        p = predictions.reshape(bs, -1)
        t = targets.reshape(bs, -1)
        
        intersection = (p * t).sum(1)       
        total=(p+t).sum(1)                     
        dice = 1- ((2.*intersection + self.smooth)/(total + self.smooth))
        return dice.mean()

class BinaryCrossEntropyLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.num_classes = kwargs.get("num_classes", 1)
        self.weights = kwargs.get("weights", self.num_classes * [1])
        self.single_loss = torch.nn.BCELoss()

    def forward(self, prediction, target):
        loss = 0
        for c in range(self.num_classes):
            loss += self.single_loss(prediction[:, c, ...], target[:, c, ...]) * self.weights[c]
        return loss / sum(self.weights)
