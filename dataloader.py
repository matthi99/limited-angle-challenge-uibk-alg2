# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:58:04 2022

@author: Schwab Matthias
"""



import numpy as np
import torch
import os



class Dataset(torch.utils.data.Dataset):
    """
    Dataset for the depth/text data.
    This dataset loads an image and applies some transformations to it.
    """

    def __init__(self, folder= "D:/Daten Holzkopf/Data-limited-angle/", train=True, difficulty=7, rec="tv", **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train=train
        if self.train:
            self.folder = folder+"train/"
        else:
            self.folder = folder+"test/"
        self.examples = self.get_examples()
        self.difficulty=difficulty
        self.rec=rec
        

    def __getitem__(self,idx):
        data = np.load(os.path.join(self.folder, self.examples[idx]), allow_pickle=True).item()
        
        if self.rec=="tv":
            inp=data['tv_rec_diff_'+str(self.difficulty)]
        elif self.rec=="fbp":   
            inp=data['fbp_diff_'+str(self.difficulty)]
        else:
            print("Reconstruction method does not exist!")
        
        #Normalize
        inp=(inp-np.min(inp))/(np.max(inp)-np.min(inp))
        inp=np.expand_dims(inp,0)
        outp=data['gt']/np.max(data['gt'])
        outp=np.expand_dims(outp,0)
                            
        inp = torch.from_numpy(inp.astype("float32")).to(self.device)
        outp = torch.from_numpy(outp.astype("float32")).to(self.device)
        return inp, outp

    def __len__(self):
        return len(self.examples)
    

    def get_examples(self):
        examples = [f for f in os.listdir(self.folder) if f.endswith('.npy')]
        return examples

class Collator:
    """
    Data collator for different types of mask-combinations.
    """


    def __call__(self, batch, *args, **kwargs):
        inputs = []
        outputs = []
        
        for inp, outp in batch:
            inputs.append(inp)
            outputs.append(outp)
        return torch.stack(inputs), torch.stack(outputs)
