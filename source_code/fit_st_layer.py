#!/usr/bin/env python3
# for lastest version, please refer to https://github.com/JiawenChenn/POLARIS
import sys
from os import mkdir
import os.path as osp
import torch.nn as nn
from typing import NoReturn, Union, Dict

import torch as t
from torch.utils.data import DataLoader

import copy
import numpy as np
import pandas as pd

import model_st_layer as M
import datasets as D
import utils as utils
import matplotlib.pyplot as plt

def fit_st_data_layer(sc_data : D.CountData,
                device : t.device,                                                                                                                                                                                                                                                                                                                                    
                st_data : D.CountData,
                learning_rate: float=0.05,
                sc_batch_size:int =512,
                st_batch_size : int=512,
                sc1_epochs:int  =10000,
                #sc2_epochs:int =10000,
                st1_epochs:int =10000, 
                silent_mode : bool = False,
                sc_from_model : str = None,
                #sc2_from_model : str = None,
                st_from_model : str = None,
                st_from_model2 : str = None,
                out_dir:str="",
                use_image: bool=True,
                Tsg_sd: float=1.0,
                **kwargs)->Dict[str,Union[pd.DataFrame,M.STModel]]:


    if not osp.exists(out_dir+"/temp"):
        mkdir(out_dir+"/temp")
    if not osp.exists(out_dir+"/loss"):
        mkdir(out_dir+"/loss")    
    t.manual_seed(2022)
    # first train sc model without coordinates or using provided coordinates
    # define single cell model
    ####     ####     ####     ####   Tsg learn from st     ####     ####     ####     ####     #### 
    # load sc-model if provided
    sc_model = M.ScModel(n_genes = sc_data.G,
                        device=device,
                        ncell=sc_data.lbl.shape[0],
                        n_celltypes = sc_data.Z,
                        with_coord=False,
                        Tsg_sd=Tsg_sd,
                        layer_num=st_data.zidx.unique().shape[0])

    sc_epoch_loss_all=np.zeros(sc1_epochs) * np.nan

    sc_model.theta.requires_grad=True
    sc_model.to(device)

    # instatiate progressbar
    progressBar = utils.SimpleProgressBar(sc1_epochs,
                                    silent_mode = silent_mode,
                                    length = 20)
    if sc_batch_size is None:
        sc_batch_size = sc_data.M
    else:
        sc_batch_size = int(np.min((sc_batch_size,sc_data.M)))

    sc_dataloader = DataLoader(sc_data,
                            batch_size = sc_batch_size,
                            shuffle = False,
                            )

    if sc_from_model is not None and osp.exists(sc_from_model):
        try:
            if device == t.device('cpu'):
                sc_model.load_state_dict(t.load(sc_from_model,map_location ='cpu'))
            else:
                sc_model.load_state_dict(t.load(sc_from_model))
        except RuntimeError:
            if device == t.device('cpu'):
                temp_model=t.load(sc_from_model,map_location ='cpu')
                del temp_model['layer_embed.weight']
                sc_model.load_state_dict(temp_model)
            else:
                temp_model=t.load(sc_from_model)
                del temp_model['layer_embed.weight']
                sc_model.load_state_dict(temp_model)

    optim_sc = t.optim.Adam(sc_model.parameters(),lr = 0.05)


        #print(st_model.x_embed.weight)
        #sc_model2.x_embed.weight.requires_grad=False
        #sc_model2.y_embed.weight.requires_grad=False
    for epoch in range(sc1_epochs):
        epoch_loss_sc = 0.0    
        for batch in sc_dataloader:
            # move batch items to device
            for k,v in batch.items():
                batch[k] = v.to(device)
            batch['x'].requires_grad = True
            # reset gradients
            optim_sc.zero_grad()
            # compute loss
            loss = sc_model.forward(**batch)
            #print(loss)
            epoch_loss_sc += loss.item()
            # compute gradients
            loss.backward()
            #if clip == True:
            #t.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            # update parameters based on gradients
            optim_sc.step()

        progressBar(epoch, epoch_loss_sc)
        sc_epoch_loss_all[epoch]=epoch_loss_sc

    t.save(sc_model.state_dict(),out_dir+"/temp/sc1_epoch"+str(sc1_epochs)+"_loss"+str(epoch_loss_sc)+".pt")
    
    logits = sc_model.o.data.cpu().numpy()
    R = sc_model.theta.data.cpu().numpy()
    # get cell type names
    typenames = sc_data.unique_labels()
    R = pd.DataFrame(R,
                    index = sc_data.genes,
                    columns = typenames,
                    )

    logits = pd.DataFrame(logits,
                        index = sc_data.genes,
                        columns = pd.Index(['logits']))


    if use_image:
        st_model = M.STModel(n_spots=st_data.M,
                        R = R.values,
                        logits = logits.values,
                        device = device,
                        Tsg_sd=Tsg_sd,
                        cell_type = R.columns.values,
                        freeze_beta = False,
                        use_image=True,
                        layer_embed=None,
                        layer_num=st_data.zidx.unique().shape[0]
                        )
        #st_model.layer_embed.requires_grad = True
    else:
        st_model = M.STModel(n_spots=st_data.M,
                    R = R.values,
                    Tsg_sd=Tsg_sd,
                    logits = logits.values,
                    device = device,
                    cell_type = R.columns.values,
                    freeze_beta = False,
                    use_image=False,
                    layer_embed=None,
                    layer_num=st_data.zidx.unique().shape[0]
                    )
    st_model.to(device)
    if st_from_model is not None and osp.exists(st_from_model):
        st_model.load_state_dict(t.load(st_from_model))

    st_dataloader = DataLoader(st_data,
                            batch_size = st_batch_size,
                            shuffle = False,)

    progressBar = utils.SimpleProgressBar(st1_epochs,
                                silent_mode = silent_mode,
                                length = 20)

    optim_st = t.optim.Adam(st_model.parameters(),
                        lr = 0.05)

    st_epoch_loss_all=np.zeros(st1_epochs) * np.nan
    print("Starting ST final training")
    for epoch in range(st1_epochs):
        epoch_loss_st = 0.0
        for batch in st_dataloader:
            # move batch items to device
            for k,v in batch.items():
                batch[k] = v.to(device)
            batch['x'].requires_grad = True
            # reset gradients
            optim_st.zero_grad()
            # compute loss
            loss = st_model.forward(**batch)
            #print(loss)
            epoch_loss_st += loss.item()
            # compute gradients
            loss.backward()
            #if clip == True:
            #t.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            # update parameters based on gradients
            optim_st.step()
        progressBar(epoch, epoch_loss_st)
        # record loss progression
        #loss_tracker(epoch_loss,epoch)
        st_epoch_loss_all[epoch]=epoch_loss_st
    t.save(st_model.state_dict(),out_dir+"/temp/st_epoch"+str(st1_epochs)+"_loss"+str(epoch_loss_st)+".pt")

    # newline after complettion
    print('\n')
    # write final loss
    #plt.clf()
    #fig = plt.figure()
    #ax = plt.axes()
    #x = np.linspace(0, sc1_epochs-1, sc1_epochs)
    #ax.plot(x, sc_epoch_loss_all)
    #plt.savefig(out_dir+"/loss/scloss.epoch"+str(sc1_epochs)+".all.png")
    #plt.clf()
    #fig = plt.figure()
    #ax = plt.axes()
    #x = np.linspace(0, st1_epochs-1, st1_epochs)
    #ax.plot(x, st_epoch_loss_all)
    #plt.savefig(out_dir+"/loss/stloss.epoch"+str(st1_epochs)+".all.png")

    W  = st_model.v.data.cpu().numpy().T

    W = W[:,0:st_model.K]
    w_columns = R.columns

    W = pd.DataFrame(W,
                     index = st_data.index,
                     columns = w_columns)


    return {'proportions':W,
            'st-model':st_model,
            'sc-model':sc_model,
            #'AE':AE_trans
           }



def fit_scRNA_data_layer(sc_data : D.CountData,
                device : t.device,                                                                                                                                                                                                                                                                                                                                    
                st_data : D.CountData,
                learning_rate: float=0.05,
                sc_batch_size:int =512,
                st_batch_size : int=512,
                sc1_epochs:int  =10000,
                sc2_epochs:int =10000,
                st1_epochs:int =10000, 
                silent_mode : bool = False,
                sc_from_model : str = None,
                sc2_from_model : str = None,
                st_from_model : str = None,
                out_dir:str="",
                use_image: bool=True,
                Tsg_sd: float=1.0,
                **kwargs)->Dict[str,Union[pd.DataFrame,M.STModel]]:


    if not osp.exists(out_dir+"/temp"):
        mkdir(out_dir+"/temp")
    if not osp.exists(out_dir+"/loss"):
        mkdir(out_dir+"/loss")    
    t.manual_seed(2022)
    # first train sc model without coordinates or using provided coordinates
    # define single cell model

    # load sc-model if provided
    st_epoch_loss_all=np.zeros(st1_epochs) * np.nan
    sc_model = M.ScModel(n_genes = sc_data.G,
                        device=device,
                        ncell=sc_data.lbl.shape[0],
                        n_celltypes = sc_data.Z,
                        with_coord=False,
                        Tsg_sd=Tsg_sd,
                        #old_model=sc_model2,
                        layer_num=st_data.zidx.unique().shape[0])

    nn.init.constant_(sc_model.layer_embed.weight, 0)
    sc_model.layer_embed.weight.requires_grad=False
    sc_epoch_loss_all=np.zeros(sc1_epochs+sc2_epochs) * np.nan

    sc_model.theta.requires_grad=True
    sc_model.to(device)

    # instatiate progressbar
    progressBar = utils.SimpleProgressBar(sc1_epochs,
                                    silent_mode = silent_mode,
                                    length = 20)
    if sc_batch_size is None:
        sc_batch_size = sc_data.M
    else:
        sc_batch_size = int(np.min((sc_batch_size,sc_data.M)))

    sc_dataloader = DataLoader(sc_data,
                            batch_size = sc_batch_size,
                            shuffle = False,
                            )

    if sc_from_model is not None and osp.exists(sc_from_model):
        sc_model.load_state_dict(t.load(sc_from_model))
    optim_sc = t.optim.Adam(sc_model.parameters(),lr = 0.05)


    for epoch in range(sc1_epochs):
        epoch_loss_sc = 0.0    
        for batch in sc_dataloader:
            # move batch items to device
            for k,v in batch.items():
                batch[k] = v.to(device)
            batch['x'].requires_grad = True
            # reset gradients
            optim_sc.zero_grad()
            # compute loss
            loss = sc_model.forward(**batch)
            #print(loss)
            epoch_loss_sc += loss.item()
            # compute gradients
            loss.backward()
            #if clip == True:
            #t.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            # update parameters based on gradients
            optim_sc.step()

        progressBar(epoch, epoch_loss_sc)
        # record loss progression
        #loss_tracker(epoch_loss,epoch)
        sc_epoch_loss_all[epoch]=epoch_loss_sc
    t.save(sc_model.state_dict(),out_dir+"/temp/sc1_epoch"+str(sc1_epochs)+"_loss"+str(epoch_loss_sc)+".pt")
    

    #sc_model_temp = copy.deepcopy(sc_model)
    sc_model = M.ScModel(n_genes = sc_data.G,
                    device=device,
                    Tsg_sd=Tsg_sd,
                    ncell=sc_data.lbl.shape[0],
                    n_celltypes = sc_data.Z,
                    with_coord=True,
                    old_model=sc_model,
                    layer_num=st_data.zidx.unique().shape[0])

    optim_sc = t.optim.Adam(sc_model.parameters(),lr = 0.05)
    progressBar = utils.SimpleProgressBar(sc2_epochs,
                                silent_mode = silent_mode,
                                length = 20)
    
    nn.init.normal_(sc_model.layer_embed.weight, mean=1.0,std=1.0)
    
    if sc2_from_model is not None and osp.exists(sc2_from_model):
        sc_model.load_state_dict(t.load(sc2_from_model))

    sc_model.layer_embed.weight.requires_grad=True
    sc_model.theta.requires_grad=False
    sc_model.o.requires_grad=False
    for epoch in range(sc2_epochs):
        epoch_loss_sc = 0.0    
        for batch in sc_dataloader:
            # move batch items to device
            for k,v in batch.items():
                batch[k] = v.to(device)
            batch['x'].requires_grad = True
            # reset gradients
            optim_sc.zero_grad()
            # compute loss
            loss = sc_model.forward(**batch)
            #print(loss)
            epoch_loss_sc += loss.item()
            # compute gradients
            loss.backward()
            #if clip == True:
            #t.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            # update parameters based on gradients
            optim_sc.step()
        progressBar(epoch, epoch_loss_sc)
        # record loss progression
        #loss_tracker(epoch_loss,epoch)
        sc_epoch_loss_all[sc1_epochs+epoch]=epoch_loss_sc
    t.save(sc_model.state_dict(),out_dir+"/temp/sc2_epoch"+str(sc2_epochs)+"_loss"+str(epoch_loss_sc)+".pt")

    logits = sc_model.o.data.cpu().numpy()
    R = sc_model.theta.data.cpu().numpy()
    # get cell type names
    typenames = sc_data.unique_labels()
    R = pd.DataFrame(R,
                    index = sc_data.genes,
                    columns = typenames,
                    )

    logits = pd.DataFrame(logits,
                        index = sc_data.genes,
                        columns = pd.Index(['logits']))

    if use_image:
        st_model = M.STModel(n_spots=st_data.M,
                        R = R.values,
                        logits = logits.values,
                        device = device,
                        Tsg_sd=Tsg_sd,
                        cell_type = R.columns.values,
                        freeze_beta = False,
                        use_image=True,
                        layer_embed=sc_model.layer_embed.eval(),
                        layer_num=st_data.zidx.unique().shape[0]
                        )
        #st_model.layer_embed.requires_grad = True
    else:
        st_model = M.STModel(n_spots=st_data.M,
                    R = R.values,
                    Tsg_sd=Tsg_sd,
                    logits = logits.values,
                    device = device,
                    cell_type = R.columns.values,
                    freeze_beta = False,
                    use_image=False,
                    layer_embed=sc_model.layer_embed.eval(),
                    layer_num=st_data.zidx.unique().shape[0]
                    )
    st_model.to(device)
    if st_from_model is not None and osp.exists(st_from_model):
        st_model.load_state_dict(t.load(st_from_model))

    st_dataloader = DataLoader(st_data,
                            batch_size = st_batch_size,
                            shuffle = False,)

    progressBar = utils.SimpleProgressBar(st1_epochs,
                                silent_mode = silent_mode,
                                length = 20)

    optim_st = t.optim.Adam(st_model.parameters(),
                        lr = 0.05)


    st_epoch_loss_all=np.zeros(st1_epochs) * np.nan
    print("Starting ST final training")
    for epoch in range(st1_epochs):
        epoch_loss_st = 0.0
        for batch in st_dataloader:
            # move batch items to device
            for k,v in batch.items():
                batch[k] = v.to(device)
            batch['x'].requires_grad = True
            # reset gradients
            optim_st.zero_grad()
            # compute loss
            loss = st_model.forward(**batch)
            #print(loss)
            epoch_loss_st += loss.item()
            # compute gradients
            loss.backward()
            #if clip == True:
            #t.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            # update parameters based on gradients
            optim_st.step()
        progressBar(epoch, epoch_loss_st)
        # record loss progression
        #loss_tracker(epoch_loss,epoch)
        st_epoch_loss_all[epoch]=epoch_loss_st
    t.save(st_model.state_dict(),out_dir+"/temp/st_epoch"+str(st1_epochs)+"_loss"+str(epoch_loss_st)+".pt")

    # newline after complettion
    print('\n')
    # write final loss
    #plt.clf()
    #fig = plt.figure()
    #ax = plt.axes()
    #x = np.linspace(0, sc1_epochs-1, sc1_epochs)
    #ax.plot(x, sc_epoch_loss_all)
    #plt.savefig(out_dir+"/loss/scloss.epoch"+str(sc1_epochs+sc2_epochs)+".all.png")
    #plt.clf()
    #fig = plt.figure()
    #ax = plt.axes()
    #x = np.linspace(0, st1_epochs-1, st1_epochs)
    #ax.plot(x, st_epoch_loss_all)
    #plt.savefig(out_dir+"/loss/stloss.epoch"+str(st1_epochs)+".all.png")

    W  = st_model.v.data.cpu().numpy().T

    W = W[:,0:st_model.K]
    w_columns = R.columns

    W = pd.DataFrame(W,
                     index = st_data.index,
                     columns = w_columns)


    return {'proportions':W,
            'st-model':st_model,
            'sc-model':sc_model,
            #'AE':AE_trans
           }


def fit_st_superres_data_layer(sc_data : D.CountData,
                device : t.device,                                                                                                                                                                                                                                                                                                                                    
                st_data : D.CountData,
                learning_rate: float=0.001,
                sc_batch_size:int =512,
                st_batch_size : int=512,
                sc1_epochs:int  =10000,
                #sc2_epochs:int =10000,
                st1_epochs:int =10000, 
                st2_epochs:int =10000, 
                silent_mode : bool = False,
                sc_from_model : str = None,
                #sc2_from_model : str = None,
                st_from_model : str = None,
                st_from_model2: str = None,
                out_dir:str="",
                use_image: bool=True,
                Tsg_sd: float=1.0,
                **kwargs)->Dict[str,Union[pd.DataFrame,M.STModel]]:


    if not osp.exists(out_dir+"/temp"):
        mkdir(out_dir+"/temp")
    if not osp.exists(out_dir+"/loss"):
        mkdir(out_dir+"/loss")    
    t.manual_seed(2022)
    # first train sc model without coordinates or using provided coordinates
    # define single cell model
    ####     ####     ####     ####   Tsg learn from st     ####     ####     ####     ####     #### 
    # load sc-model if provided
    sc_model = M.ScModel(n_genes = sc_data.G,
                        device=device,
                        ncell=sc_data.lbl.shape[0],
                        n_celltypes = sc_data.Z,
                        with_coord=False,
                        Tsg_sd=Tsg_sd,
                        layer_num=st_data.zidx.unique().shape[0])

    sc_epoch_loss_all=np.zeros(sc1_epochs) * np.nan

    sc_model.theta.requires_grad=True
    sc_model.to(device)

    # instatiate progressbar
    progressBar = utils.SimpleProgressBar(sc1_epochs,
                                    silent_mode = silent_mode,
                                    length = 20)
    if sc_batch_size is None:
        sc_batch_size = sc_data.M
    else:
        sc_batch_size = int(np.min((sc_batch_size,sc_data.M)))

    sc_dataloader = DataLoader(sc_data,
                            batch_size = sc_batch_size,
                            shuffle = False,
                            )

    if sc_from_model is not None and osp.exists(sc_from_model):
        try:
            if device == t.device('cpu'):
                sc_model.load_state_dict(t.load(sc_from_model,map_location ='cpu'))
            else:
                sc_model.load_state_dict(t.load(sc_from_model))
        except RuntimeError:
            temp_model=t.load(sc_from_model)
            del temp_model['layer_embed.weight']
            sc_model.load_state_dict(temp_model)

            
    optim_sc = t.optim.Adam(sc_model.parameters(),lr = 0.05)


        #print(st_model.x_embed.weight)
        #sc_model2.x_embed.weight.requires_grad=False
        #sc_model2.y_embed.weight.requires_grad=False
    for epoch in range(sc1_epochs):
        epoch_loss_sc = 0.0    
        for batch in sc_dataloader:
            # move batch items to device
            for k,v in batch.items():
                batch[k] = v.to(device)
            batch['x'].requires_grad = True
            # reset gradients
            optim_sc.zero_grad()
            # compute loss
            loss = sc_model.forward(**batch)
            #print(loss)
            epoch_loss_sc += loss.item()
            # compute gradients
            loss.backward()
            #if clip == True:
            #t.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            # update parameters based on gradients
            optim_sc.step()

        progressBar(epoch, epoch_loss_sc)
        # record loss progression
        #loss_tracker(epoch_loss,epoch)
        sc_epoch_loss_all[epoch]=epoch_loss_sc
    t.save(sc_model.state_dict(),out_dir+"/temp/sc1_epoch"+str(sc1_epochs)+"_loss"+str(epoch_loss_sc)+".pt")
    
    logits = sc_model.o.data.cpu().numpy()
    R = sc_model.theta.data.cpu().numpy()
    # get cell type names
    typenames = sc_data.unique_labels()
    R = pd.DataFrame(R,
                    index = sc_data.genes,
                    columns = typenames,
                    )

    logits = pd.DataFrame(logits,
                        index = sc_data.genes,
                        columns = pd.Index(['logits']))

    # st first round
    st_model = M.STModel(n_spots=st_data.M,
                    R = R.values,
                    Tsg_sd=Tsg_sd,
                    logits = logits.values,
                    device = device,
                    cell_type = R.columns.values,
                    freeze_beta = False,
                    use_image=False,
                    layer_embed=None,
                    layer_num=st_data.zidx.unique().shape[0]
                    )
    st_model.to(device)
    if st_from_model is not None and osp.exists(st_from_model):
        st_model.load_state_dict(t.load(st_from_model))

    st_dataloader = DataLoader(st_data,
                            batch_size = st_batch_size,
                            shuffle = False,)

    progressBar = utils.SimpleProgressBar(st1_epochs,
                                silent_mode = silent_mode,
                                length = 20)

    optim_st = t.optim.Adam(st_model.parameters(),
                        lr = 0.05)

    st_epoch_loss_all=np.zeros(st1_epochs+st2_epochs) * np.nan
    print("Starting ST final training")
    for epoch in range(st1_epochs):
        epoch_loss_st = 0.0
        for batch in st_dataloader:
            # move batch items to device
            for k,v in batch.items():
                batch[k] = v.to(device)
            batch['x'].requires_grad = True
            # reset gradients
            optim_st.zero_grad()
            # compute loss
            loss = st_model.forward(**batch)
            #print(loss)
            epoch_loss_st += loss.item()
            # compute gradients
            loss.backward()
            #if clip == True:
            #t.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            # update parameters based on gradients
            optim_st.step()
        progressBar(epoch, epoch_loss_st)
        # record loss progression
        #loss_tracker(epoch_loss,epoch)
        st_epoch_loss_all[epoch]=epoch_loss_st
    t.save(st_model.state_dict(),out_dir+"/temp/st1_epoch"+str(st1_epochs)+"_loss"+str(epoch_loss_st)+".pt")

    st_model2 = M.STModel(n_spots=st_data.M,
                    R = R.values,
                    logits = logits.values,
                    device = device,
                    Tsg_sd=Tsg_sd,
                    cell_type = R.columns.values,
                    freeze_beta = False,
                    use_image=True,
                    layer_embed=st_model.layer_embed.eval(),
                    layer_num=st_data.zidx.unique().shape[0]
                    )
    st_model2.layer_embed.weight.requires_grad = False

    st_model2.to(device)
    if st_from_model2 is not None and osp.exists(st_from_model2):
        st_model2.load_state_dict(t.load(st_from_model2))

    st_dataloader = DataLoader(st_data,
                            batch_size = st_batch_size,
                            shuffle = False,)

    progressBar = utils.SimpleProgressBar(st2_epochs,
                                silent_mode = silent_mode,
                                length = 20)

    optim_st = t.optim.Adam(st_model2.parameters(),
                        lr = learning_rate)

    print("Starting ST final training")
    for epoch in range(st2_epochs):
        epoch_loss_st = 0.0
        for batch in st_dataloader:
            # move batch items to device
            for k,v in batch.items():
                batch[k] = v.to(device)
            batch['x'].requires_grad = True
            # reset gradients
            optim_st.zero_grad()
            # compute loss
            loss = st_model2.forward(**batch)
            #print(loss)
            epoch_loss_st += loss.item()
            # compute gradients
            loss.backward()
            #if clip == True:
            #t.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            # update parameters based on gradients
            optim_st.step()
        progressBar(epoch, epoch_loss_st)
        # record loss progression
        #loss_tracker(epoch_loss,epoch)
        st_epoch_loss_all[epoch+st1_epochs]=epoch_loss_st
    t.save(st_model2.state_dict(),out_dir+"/temp/st2_epoch"+str(st2_epochs)+"_loss"+str(epoch_loss_st)+".pt")

    # newline after complettion
    print('\n')
    # write final loss
    #plt.clf()
    #fig = plt.figure()
    #ax = plt.axes()
    #x = np.linspace(0, sc1_epochs-1, sc1_epochs)
    #ax.plot(x, sc_epoch_loss_all)
    #plt.savefig(out_dir+"/loss/scloss.epoch"+str(sc1_epochs)+".all.png")
    #plt.clf()
    #fig = plt.figure()
    #ax = plt.axes()
    #x = np.linspace(0, st1_epochs+st2_epochs-1, st1_epochs+st2_epochs)
    #ax.plot(x, st_epoch_loss_all)
    #plt.savefig(out_dir+"/loss/stloss.epoch"+str(st1_epochs+st2_epochs)+".all.png")

    W  = st_model2.v.data.cpu().numpy().T

    W = W[:,0:st_model2.K]
    w_columns = R.columns

    W = pd.DataFrame(W,
                     index = st_data.index,
                     columns = w_columns)


    return {'proportions':W,
            'st-model':st_model2,
            'sc-model':sc_model,
            #'AE':AE_trans
           }
