#!/usr/bin/env python3
# for lastest version, please refer to https://github.com/JiawenChenn/POLARIS

from __future__ import print_function
import re
import sys
from typing import List,Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from PIL import ImageEnhance
from skimage.segmentation import clear_border
from skimage import measure, color, io
from torchvision import transforms
import pandas as pd
import torch as t
from torch.utils.data import Dataset
from scipy.stats import rankdata
import squidpy as sq

import numpy as np
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import utils as utils
import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import models_mae
import squidpy as sq

import numpy as np
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import cv2

Image.MAX_IMAGE_PIXELS = 308000000

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def MAE_extract(temp_image_crop,model):
    with t.no_grad():
        img = temp_image_crop.resize((224, 224))
        img = np.array(img) / 255.
        assert img.shape == (224, 224, 3)
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std
        x = t.tensor(img)
        # make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)
        x = model.patch_embed(x.float())
        # add pos embed w/o cls token
        x = x + model.pos_embed[:, 1:, :]
        cls_token = model.cls_token + model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in model.blocks:
            x = blk(x)
    return x[:, 1:, :].mean(dim=1).detach()


class CountDataHelper(object):
    """
    Helper class for CountData class
    """
    @classmethod
    def update(self,func):
        def wrapper(self,*args,**kwargs):
            tmp = func(self,*args,**kwargs)
            self.G = int(self.cnt.shape[1])
            self.M = int(self.cnt.shape[0])
            self.Z = np.unique(self.lbl).shape[0]
            self.libsize = self.cnt.sum(dim = 1)
            return tmp
        return wrapper

class CountData(Dataset):
    """CountData Dataset class
    Class to hold count data from ST or
    Single Cell experiments
    Arguments:
    ---------
    cnt : pd.DataFrame
        Count data [n_observations x n_genes]
    lbl : pd.DataFrame
        Annotation/Label data [n_observations]
    """
    @CountDataHelper.update
    def __init__(self,
                 cnt : pd.DataFrame,
                 lbl : pd.DataFrame  = None,
                 st : bool = False,
                 st_dataset : str = "",
                 img_pths : List[str]=None,
                 sc_given_layer: bool=False,
                 sc_layer:pd.DataFrame  = None,
                 cell_num_given: bool = False,
                 transform =transforms.Compose([transforms.Resize((64, 64)),
                                       transforms.ToTensor()]),
                )-> None:
        self.cnt = cnt
        self.transform = transform
        self.lbl = np.ones(self.cnt.shape[0]) * np.nan
        self.zidx = np.ones(self.cnt.shape[0]) * np.nan
        self.sc_layer = np.ones(self.cnt.shape[0]) * np.nan
        self.sc_layer_idx = np.ones(self.cnt.shape[0]) * np.nan
        self.cell_num=np.ones(self.cnt.shape[0])

        # if labels are provided
        if lbl is not None:
            self.lbl = lbl
            self.index = self.cnt.index.intersection(self.lbl.index)
            self.cnt = self.cnt.loc[self.index,:]
            if cell_num_given:
                #print(self.lbl.iloc[:,0])
                self.cell_num = self.lbl.iloc[:,0].loc[self.index].values.reshape(-1,)
                self.lbl = self.lbl.iloc[:,1].loc[self.index].values.reshape(-1,)
            else:
                self.cell_num =np.ones(len(self.index))
                self.lbl = self.lbl.loc[self.index].values.reshape(-1,)
            # convert labels to numeric indices
            tonumeric = { v:k for k,v in enumerate(np.unique(self.lbl)) }
            self.zidx = np.array([tonumeric[l] for l in self.lbl])
            # Sort data according to label enumeration
            # to speed up element acession
            if st is False:
                srt = np.argsort(self.zidx)
                self.zidx = self.zidx[srt]
                self.lbl = self.lbl[srt]
                self.cell_num=self.cell_num[srt]
                self.cnt = self.cnt.iloc[srt,:]
            if sc_given_layer:
                self.sc_layer=sc_layer.loc[self.index].values.reshape(-1,)
                tonumeric = { v:k for k,v in enumerate(np.unique(self.sc_layer)) }
                self.sc_layer_idx = np.array([tonumeric[l] for l in self.sc_layer])

            self.zidx = t.LongTensor(self.zidx.flatten().astype(np.int32))

        self.genes = self.cnt.columns
        self.index = self.cnt.index
        self.xaxis = np.ones(len(self.index)) * np.nan
        self.yaxis = np.ones(len(self.index)) * np.nan
        self.slides = np.ones(len(self.index)) * np.nan
        self.image_tensor = [np.nan] *len(self.index)
        self.MAE_output= [np.nan] *len(self.index)
        chkpt_dir = './mae_visualize_vit_large_ganloss.pth'
        transforms1=transforms.Compose([transforms.Resize((64, 64))])

        if st == True:
            progressBar = utils.SimpleProgressBar(len(self.index),
                            silent_mode = True,
                            length = 20)
            if st_dataset =="mouse_brain":
                model = prepare_model(chkpt_dir, 'mae_vit_large_patch16').eval()
                for i in range(len(self.index)):
                    self.yaxis[i] = self.index[i].split("-")[1].split("_")[1]
                    self.xaxis[i] = self.index[i].split("-")[1].split("_")[0]
                    self.slides[i] = int(self.index[i].split("&-")[0])
                    if i==0:
                        temp_image = Image.open(img_pths[int(self.slides[i])])
                        temp_image=temp_image.convert('RGB')
                    elif self.slides[i-1] != self.slides[i]:
                        temp_image = Image.open(img_pths[int(self.slides[i])])
                        temp_image=temp_image.convert('RGB')
                    temp_image_crop = temp_image.crop((self.xaxis[i]-8, self.yaxis[i]-8,self.xaxis[i]+8, self.yaxis[i]+8))
                    temp_image_crop_neighboor = temp_image.crop((self.xaxis[i]-8*3, self.yaxis[i]-8*3, self.xaxis[i]+8*3, self.yaxis[i]+8*3))
                    self.MAE_output[i]=t.cat((MAE_extract(temp_image_crop.convert('RGB'),model).squeeze(0),
                                              MAE_extract(temp_image_crop_neighboor.convert('RGB'),model).squeeze(0)))
                    progressBar(i,0)
            elif st_dataset =="ob":
                for i in range(len(self.index)):
                    self.yaxis[i] = self.index[i].split("-")[1].split("_")[1]
                    self.xaxis[i] = self.index[i].split("-")[1].split("_")[0]
                    self.slides[i] = int(self.index[i].split("&-")[0])
                    if i==0:
                        temp_image = Image.open(img_pths[int(self.slides[i])])
                        temp_image=temp_image.convert('RGB')
                    elif self.slides[i-1] != self.slides[i]:
                        temp_image = Image.open(img_pths[int(self.slides[i])])
                        temp_image=temp_image.convert('RGB')
            elif st_dataset =="breast_cancer":
                model = prepare_model(chkpt_dir, 'mae_vit_large_patch16').eval()
                progressBar = utils.SimpleProgressBar(len(self.index),
                            silent_mode = True,
                            length = 20)
                for i in range(len(self.index)):
                    self.yaxis[i] = self.index[i].split("-")[1].split("_")[1]
                    self.xaxis[i] = self.index[i].split("-")[1].split("_")[0]
                    self.slides[i] = int(self.index[i].split("&-")[0])
                    if i==0:
                        temp_image = Image.open(img_pths[int(self.slides[i])])
                    elif self.slides[i-1] != self.slides[i]:
                        temp_image = Image.open(img_pths[int(self.slides[i])])
                    temp_image_crop = temp_image.crop((self.xaxis[i]-60, self.yaxis[i]-60,self.xaxis[i]+60, self.yaxis[i]+60))
                    temp_image_crop_neighboor = temp_image.crop((self.xaxis[i]-60*3, self.yaxis[i]-60*3, self.xaxis[i]+60*3, self.yaxis[i]+60*3))
                    self.MAE_output[i]=t.cat((MAE_extract(temp_image_crop.convert('RGB'),model).squeeze(0),
                                              MAE_extract(temp_image_crop_neighboor.convert('RGB'),model).squeeze(0)))
                    progressBar(i,0)
            elif st_dataset =="human_heart":
                model = prepare_model(chkpt_dir, 'mae_vit_large_patch16').eval()
                progressBar = utils.SimpleProgressBar(len(self.index),
                            silent_mode = True,
                            length = 20)
                for i in range(len(self.index)):
                    self.yaxis[i] = self.index[i].split("-")[1].split("_")[1]
                    self.xaxis[i] = self.index[i].split("-")[1].split("_")[0]
                    self.slides[i] = int(self.index[i].split("&-")[0])
                    if i==0:
                        temp_image = Image.open(img_pths[int(self.slides[i])])
                        temp_image = temp_image.transpose(Image.FLIP_LEFT_RIGHT)
                        temp_image=temp_image#.convert('RGB')
                        temp_image=ImageOps.invert(temp_image)
                        enh_con=ImageEnhance.Contrast(temp_image)
                        temp_image = enh_con.enhance(10)
                    elif self.slides[i-1] != self.slides[i]:
                        temp_image = Image.open(img_pths[int(self.slides[i])])
                        temp_image = temp_image.transpose(Image.FLIP_LEFT_RIGHT)
                        temp_image=temp_image#.convert('RGB')
                        temp_image=ImageOps.invert(temp_image)
                        enh_con=ImageEnhance.Contrast(temp_image)
                        temp_image = enh_con.enhance(10)
                    temp_image_crop = temp_image.crop((self.xaxis[i]-227, self.yaxis[i]-212, self.xaxis[i]+227, self.yaxis[i]+212))
                    temp_image_crop_neighboor = temp_image.crop((self.xaxis[i]-227*3, self.yaxis[i]-212*3, self.xaxis[i]+227*3, self.yaxis[i]+212*3))
                    self.MAE_output[i]=t.cat((MAE_extract(temp_image_crop.convert('RGB'),model).squeeze(0),
                                              MAE_extract(temp_image_crop_neighboor.convert('RGB'),model).squeeze(0)))
                    progressBar(i,0)
            elif st_dataset =="simulation":
                print("No image for simulation")
        # Convert to tensor
        self.x_order=rankdata(self.xaxis, method='dense')- 1
        self.y_order=rankdata(self.yaxis, method='dense')- 1
        self.cnt = t.tensor(self.cnt.values.astype(np.float32))
        self.libsize = self.cnt.sum(dim = 1)
        if self.cell_num.max()>0:
            ridx = np.where(self.cell_num>0)[0] #filter cells
            self.cell_num=self.cell_num[ridx]
            self.image_tensor=[self.image_tensor[i] for i in ridx]
            self.MAE_output=[self.MAE_output[i] for i in ridx]
            self.cnt = self.cnt[ridx,:]
            self.lbl = self.lbl[ridx]
            self.sc_layer=self.sc_layer[ridx]
            self.sc_layer_idx=self.sc_layer_idx[ridx]
            self.zidx = self.zidx[ridx].type(t.LongTensor)
            self.xaxis =self.xaxis[ridx]
            self.yaxis = self.yaxis[ridx]
            self.x_order=self.x_order[ridx]
            self.y_order=self.y_order[ridx]
            self.slides = self.slides[ridx]
            self.index=self.index[ridx]
    @CountDataHelper.update
    def filter_genes(self,
                     pattern : str = None,
                    )-> None:
        if pattern is None:
           pattern =  '^RP|MALAT1'
        keep = [ re.search(pattern,x.upper()) is \
                None for x in self.genes]
        self.cnt = self.cnt[:,keep]
        self.genes = self.genes[keep]
    @CountDataHelper.update
    def filter_bad(self,
                   min_counts : int = 0,
                   min_occurance : int = 0,
                  )-> None:
        row_thrs, col_thrs = min_counts,min_occurance
        ridx = np.where(self.cnt.sum(dim = 1) > row_thrs)[0] #filter cells
        cidx = np.where((self.cnt != 0).type(t.float32).sum(dim = 0) > col_thrs)[0]#fileter gene
        self.cnt = self.cnt[ridx,:][:,cidx]
        self.lbl = self.lbl[ridx]
        self.cell_num=self.cell_num[ridx]
        self.sc_layer=self.sc_layer[ridx]
        self.sc_layer_idx=self.sc_layer_idx[ridx]
        self.genes=self.genes[cidx]
        self.zidx = self.zidx[ridx].type(t.LongTensor)
        self.xaxis =self.xaxis[ridx]
        self.yaxis = self.yaxis[ridx]
        self.x_order=self.x_order[ridx]
        self.y_order=self.y_order[ridx]
        self.slides = self.slides[ridx]
        self.index=self.index[ridx]
        self.image_tensor=[self.image_tensor[i] for i in ridx]
        self.MAE_output=[self.MAE_output[i] for i in ridx]
        #self.image_tensor = self.image_tensor[ridx]
    @CountDataHelper.update
    def intersect(self,
                  exog_genes : pd.Index,
                 ) -> pd.Index:
        inter = exog_genes.intersection(self.genes)
        inter = np.unique(inter)
        keep = np.array([ self.genes.get_loc(x) for x in inter])
        self.genes = pd.Index(inter)
        self.cnt = self.cnt[:,keep]
        return self.genes
    def unique_labels(self,
                     )->np.ndarray:
        _,upos = np.unique(self.zidx, return_index = True)
        typenames = self.lbl[upos]
        return typenames
    def __getitem__(self,
                    idx: List[int],
                   )-> Dict:
        sample = {'x' : self.cnt[idx,:],
                  'meta' : self.zidx[idx],
                  'sf' : self.libsize[idx],
                  'gidx' : t.tensor(idx),
                  #'sc_layer' :self.sc_layer[idx],
                  'sc_layer_idx' :self.sc_layer_idx[idx],
                  'xaxis' : self.xaxis[idx],
                  'yaxis' : self.yaxis[idx],
                  'x_order': self.x_order[idx],
                  'y_order':self.y_order[idx],
                  'slides' : self.slides[idx],
                  'image_tensor' : self.image_tensor[idx],
                  'cell_num': self.cell_num[idx],
                  'MAE_output':self.MAE_output[idx]
                 }
        return sample
        
    def __len__(self,
               )-> int:
        """Length of CountData object"""
        return self.M


def make_sc_dataset(cnt_pth : str,
                    lbl_pth : str,
                    topn_genes : int = None,
                    gene_list_pth : str = None,
                    sc_given_layer : bool = False,
                    filter_genes : bool = False,
                    lbl_colname : str = None,
                    min_counts : int = 300,
                    min_cells : int = 0,
                    transpose : bool = False,
                    upper_bound : int = None,
                    lower_bound : int = None,
                    ):
    sc_ext = utils.get_extenstion(cnt_pth)
    cnt = utils.read_file(cnt_pth,sc_ext)
    if transpose:
        cnt = cnt.T
    lbl = utils.read_file(lbl_pth)
    # get labels
    if sc_given_layer:
        sc_layer=lbl.iloc[:,1]
    else:
        sc_layer=None
    if lbl_colname is None:
        lbl = lbl.iloc[:,0]
    else:
        lbl = lbl.loc[:,lbl_colname]
        
    # match count and label data
    inter = cnt.index.intersection(lbl.index)
    if inter.shape[0] < 1:
        print("[ERROR] : single cell count and annotation"\
              " data did not match. Exiting.",
              file = sys.stderr,
              )
    cnt = cnt.loc[inter,:]
    lbl = lbl.loc[inter]
    if sc_given_layer:
        sc_layer=sc_layer.loc[inter]
    if upper_bound is not None or\
       lower_bound is not None:
        cnt,lbl = utils.subsample_data(cnt,
                                       lbl,
                                       lower_bound,
                                       upper_bound,
                                       )
        if sc_given_layer:
            inter = lbl.index.intersection(sc_layer.index)
            sc_layer=sc_layer.loc[inter]


    # select top N expressed genes
    if topn_genes is not None:
        genesize = cnt.values.sum(axis = 0)
        topn_genes = np.min((topn_genes,genesize.shape[0]))
        sel = np.argsort(genesize)[::-1]
        sel = sel[0:topn_genes]
        cnt = cnt.iloc[:,sel]
    # only use genes in specific genes list
    # if specified
    if gene_list_pth is not None:
        with open(gene_list_pth,'r+') as fopen:
            gene_list = fopen.readlines()
        gene_list = pd.Index([ x.replace('\n','') for x in gene_list ])
        sel = cnt.columns.intersection(gene_list)
        cnt = cnt.loc[:,sel]
    # create sc data set
    dataset = CountData(cnt = cnt,
                        lbl = lbl,
                        sc_given_layer=sc_given_layer,
                        sc_layer=sc_layer)
    # filter genes based on names
    if filter_genes:
        dataset.filter_genes()
    # filter data based on quality
    if any([min_counts > 0,min_cells > 0]):
        dataset.filter_bad(min_counts = min_counts,
                           min_occurance = min_cells,
                          )
    return dataset

def make_st_dataset(cnt_pths : List[str],
                    image_pths : List[str],
                    lbl_pths =None,
                    topn_genes : bool = None,
                    cell_num_given: bool=False,
                    min_counts : int = 0,
                    min_spots : int = 0,
                    filter_genes : bool = False,
                    transpose : bool = False,
                    dataset_source : str = "",
                    )-> CountData :

    # create joint matrix for count data

    st_ext = utils.get_extenstion(cnt_pths[0])
    if st_ext == "h5ad":
        cnt = utils.read_h5ad_st(cnt_pths)
    else:
        cnt = utils.make_joint_matrix(cnt_pths,transpose)
        if lbl_pths==None:
            lbl=None
        else:
            lbl = utils.make_joint_matrix(lbl_pths,False)
            if cell_num_given:
                # match count and label data
                inter = cnt.index.intersection(lbl.index)
                if inter.shape[0] < 1:
                    print("[ERROR] : st spot and annotation"\
                        " data did not match. Exiting.",
                        file = sys.stderr,)

                cnt = cnt.loc[inter,:]
                lbl = lbl.loc[inter,:]
            else:
                lbl = lbl.iloc[:,0]
                # match count and label data
                inter = cnt.index.intersection(lbl.index)
                if inter.shape[0] < 1:
                    print("[ERROR] : st spot and annotation"\
                        " data did not match. Exiting.",
                        file = sys.stderr,)

                cnt = cnt.loc[inter,:]
                lbl = lbl.loc[inter]

    # select top N genes if specified
    if topn_genes is not None:
        genesize = cnt.values.sum(axis = 0)
        topn_genes = np.min((topn_genes,genesize.shape[0]))
        sel = np.argsort(genesize)[::-1]
        sel = sel[0:topn_genes]
        cnt = cnt.iloc[:,sel]

    dataset = CountData(cnt=cnt,
                        lbl=lbl,
                        st=True,
                        cell_num_given=cell_num_given,
                        st_dataset=dataset_source,
                        img_pths=image_pths
                        )

    # filter genes based on name
    if filter_genes:
        dataset.filter_genes()

    # filter data based on quality
    if any([min_counts > 0,min_spots > 0]):
        dataset.filter_bad(min_counts = min_counts,
                           min_occurance = min_spots,
                           )
    return dataset