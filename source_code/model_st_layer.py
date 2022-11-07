#!/usr/bin/env python3
# for lastest version, please refer to https://github.com/JiawenChenn/POLARIS

import sys
import torch as t
import torch.nn as nn
from torch.nn.parameter import Parameter
import copy

import numpy as np
import pandas as pd

from typing import NoReturn, List, Tuple, Union, Collection
import logging

import os.path as osp
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def NormalizeData(data):
    return (data - data.min()) / (data.max() - data.min())

class ScModel(nn.Module):
    """ Model for singel cell data """

    def __init__(self,
                 ncell:int,
                 n_genes : int,
                 n_celltypes : int,
                 device : t.device,
                 with_coord = False,
                 old_model = None,
                 layer_num=None,
                 Tsg_sd:float=1.0,
                 )->None:

        super().__init__()

        # Get dimensions from data
        self.K = n_celltypes
        self.G = n_genes
        self.n = ncell
        self.with_coord=with_coord
        self.device=device
        self.Tsg_sd=Tsg_sd
        # Define parameters to be estimated
        self.theta = Parameter(t.Tensor(self.G,self.K).to(device))
        self.R = t.Tensor(self.G,self.K).to(device)
        self.o = Parameter(t.Tensor(self.G,1).to(device))

        # Initialize parameters
        nn.init.normal_(self.o,
                        mean = 0.0,
                        std = 1.0)

        nn.init.normal_(self.theta,
                        mean = 0.0,
                        std = 1.0)

        # Functions to be used
        self.nb = t.distributions.NegativeBinomial
        self.softpl = nn.functional.softplus
        self.logsig = nn.functional.logsigmoid

        if self.with_coord is True:
            #print(x_embed.weight)
            self.layer_num=layer_num
            self.layer_embed=nn.Embedding(layer_num,self.G).to(device)
        
        if old_model is not None:
            self.load_state_dict(old_model.state_dict())

    def _llnb_with_coord(self,
              x : t.Tensor,
              Tsg:t.Tensor,
              meta : t.LongTensor,
              sf : t.Tensor,
             ) -> t.Tensor :

        log_unnormalized_prob = (sf*self.softpl(self.theta[:,meta]+Tsg) * self.logsig(-self.o) +
                                 x * self.logsig(self.o))

        log_normalization = -t.lgamma(sf*self.softpl(self.theta[:,meta]+Tsg) + x) + \
                             t.lgamma(1. + x) + \
                             t.lgamma(sf*self.softpl(self.theta[:,meta]+Tsg))
        ll = t.sum(log_unnormalized_prob - log_normalization)

        return ll

    def _llnb(self,
              x : t.Tensor,
              meta : t.LongTensor,
              sf : t.Tensor,
             ) -> t.Tensor :

        log_unnormalized_prob = (sf*self.R[:,meta] * self.logsig(-self.o) +
                                 x * self.logsig(self.o))

        log_normalization = -t.lgamma(sf*self.R[:,meta] + x) + \
                             t.lgamma(1. + x) + \
                             t.lgamma(sf*self.R[:,meta])
        ll = t.sum(log_unnormalized_prob - log_normalization)

        return ll

    def noise_loss(self,
                   Tsg,
                  )-> t.Tensor:
        """Regularizing term for noise"""
        return -0.5/(self.Tsg_sd**2)*t.sum(t.pow((Tsg),2))

    def forward(self,
                x : t.Tensor,
                meta : t.LongTensor,
                gidx : t.Tensor,
                sf : t.Tensor,
                sc_layer_idx : t.Tensor,
                **kwargs,
                ) -> t.Tensor :
        """Forward pass during optimization"""
        # rates for each cell type
        if self.with_coord is False:
            self.R = self.softpl(self.theta)
            self.loss = -self._llnb(x.transpose(1,0),
                        meta,sf)
        else:
            #self.R = self.softpl(self.theta)
            #self.R = self.theta
            #Tsg= t.clamp(self.softpl(self.layer_embed(sc_layer_idx)), max=5)
            Tsg= self.layer_embed(sc_layer_idx)
            #print(Tsg.shape)
            #print(Tsg.get_device())
            # get loss for current parameters
            self.loss = -self._llnb_with_coord(x.transpose(1,0),
                                    Tsg.transpose(1,0),
                                    meta,sf)-self.noise_loss(Tsg)

        return self.loss

    def __str__(self,):
        return f"sc_model"


class STModel(nn.Module):

    def __init__(self,
                 n_spots: int,
                 R : np.ndarray,
                 logits : np.ndarray,
                 device : t.device,
                 layer_num: int,
                 layer_embed=None,
                 old_model=None,
                 Tsg_sd:float=1.0,
                 use_image:bool=False,
                 **kwargs,
                 )->None:

        super().__init__()
        self.S = n_spots
        self.G, self.K = R.shape
        self.Tsg_sd=Tsg_sd
        self.use_image=use_image
        self.dropout = nn.Dropout(0.1)
        if use_image:
            self.e_fc_1 = nn.Linear(2048, 512).to(device)
            self.e_fc_2 = nn.Linear(512, self.K).to(device)
            #self.e_fc_3 = nn.Linear(512, self.K).to(device)
            #nn.init.kaiming_uniform_(self.e_fc_1.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.kaiming_uniform_(self.e_fc_2.weight, mode='fan_in', nonlinearity='relu')
        else:
            self.theta = Parameter(t.tensor(np.zeros((self.K,self.S)).astype(np.float32)).to(device))
            nn.init.normal_(self.theta, mean = 0.0,std = 1.0)
        #self.bn1 = nn.BatchNorm1d(self.K).to(device)
        # Data from single cell estimates; Rates (R) and logits (o)
        self.R = t.tensor(R.astype(np.float32)).to(device)
        self.o = t.tensor(logits.astype(np.float32).reshape(-1,1)).to(device)

        # model specific parameters
        self.softpl = nn.functional.softplus
        self.lsig = nn.functional.logsigmoid
        self.sig = t.sigmoid

        self.beta = Parameter(t.tensor(np.zeros((self.G,1)).astype(np.float32)).to(device))
        self.beta_trans = self.softpl
        nn.init.normal_(self.beta, mean = 0.0, std = 0.1)
        self.v = t.tensor(np.zeros((self.K,self.S)).astype(np.float32)).to(device)

        self.loss = t.tensor(0.0)
        self.model_ll = 0.0

        if layer_embed is not None:
            self.layer_embed=copy.deepcopy(layer_embed)
            self.layer_embed.requires_grad=False
        else:
            self.layer_num=layer_num
            self.layer_embed=nn.Embedding(layer_num,self.G).to(device)


        self.loss = t.tensor(0.0)
        self.model_ll = 0.0

        if old_model is not None:
            self.load_state_dict(old_model.state_dict())


    def noise_loss(self,
                   Tsg,
                  )-> t.Tensor:
        """Regularizing term for noise"""
        return -0.5/(self.Tsg_sd**2)*t.sum(t.pow((Tsg),2))

    def _llnb(self,
              x : t.Tensor,
              )->t.Tensor:
        """Log Likelihood function for standard model"""

        log_unnormalized_prob = (self.r) * self.lsig(-self.o) + \
                                x * self.lsig(self.o)
        log_normalization = -t.lgamma((self.r) + x) + \
                             t.lgamma(1. + x) + \
                             t.lgamma((self.r))

        ll = t.sum(log_unnormalized_prob - log_normalization)

        self.ll = ll.item()

        return ll

    def __str__(self,
               )-> str:
        return f"st_model"

    def forward(self,
                x : t.tensor,
                gidx : t.tensor,
                meta: t.tensor,
                #image_tensor: t.tensor,
                MAE_output: t.tensor,
                cell_num:t.tensor,
                **kwargs,
                ) -> t.tensor:

        """Forward pass"""

        self.gidx = gidx
        Tsg= self.layer_embed(meta)
        self.Rhat = t.mul(self.beta_trans(self.beta).unsqueeze(1), 
                          self.softpl(self.R.repeat(gidx.shape[0],1,1).transpose(0,1)+
                          Tsg.repeat(self.R.shape[1],1,1).transpose(0,2)))
        if self.use_image:
            temp = self.e_fc_1(MAE_output)
            #temp = self.dropout(temp)
            temp = self.e_fc_2(temp)
            #self.temp = self.bn1(self.temp)
            #self.temp=t.clamp(self.temp, min=1e-04, max=20)
            temp = self.softpl(temp)
            #temp = t.sigmoid(temp)
            #temp = (cell_num.unsqueeze(-1)*temp).float()
            temp = temp.float()
            self.v[:,self.gidx] = temp.transpose(1,0).clone().detach()
            self.r = t.einsum('gsz,sz->gs',[self.Rhat,temp])
        else:
            self.v = self.softpl(self.theta)
            self.r = t.einsum('gsz,zs->gs',[self.Rhat,self.v[:,self.gidx]])

        # get loss for current parameters
        self.loss = -self._llnb(x.transpose(1,0))-self.noise_loss(Tsg)

        return self.loss
