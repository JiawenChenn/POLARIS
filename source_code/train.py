#!/usr/bin/env python3
# for lastest version, please refer to https://github.com/JiawenChenn/POLARIS

import sys
from os import mkdir, makedirs,getcwd
import os.path as osp
import argparse as arp
import torch.nn as nn


import torch as t
from torch.cuda import is_available
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import fit_st_layer as fit_st_layer
import datasets as D
import model_st_layer as M
import utils as utils

prs = arp.ArgumentParser()
parser = arp.ArgumentParser()

parser.add_argument('-scc','--sc_cnt',
                    required = False,
                    type = str,
                    help = ''.join(["path to single cell",
                                    " count file. Should be",
                                    " on format n_cells x n_genes",
                                    " use flag sct to transpose if",
                                    " if necessary"]))

parser.add_argument('-scl','--sc_labels',
                    required = False,
                    type = str,
                    help = ''.join(["path to single cell",
                                    " labels file. Should be on",]))

parser.add_argument('--prefix',
                    required = False,
                    type = str,
                    help = 'output prefix')

parser.add_argument('--sc_given_layers',
                    required = False,
                    action = 'store_true',
                    default = False,
                    help = 'sc given layer or not')

parser.add_argument('--sc_transpose',
                    action = 'store_true',
                    default = False,
                    help = 'whether transpose scRNA data')

parser.add_argument('--st_transpose',
                    action = 'store_true',
                    default = False,
                    help = 'whether transpose st data')

parser.add_argument('-scm1','--sc_from_model',
                    required = False,
                    type = str,
                    default=None,
                    help ='path to already fitted sc model1')

parser.add_argument('-scm2','--sc2_from_model',
                    required = False,
                    type = str,
                    default=None,
                    help ='path to already fitted sc model2')

parser.add_argument('-stc','--st_cnt',
                    required = False,
                    nargs = '+',
                    help = ''.join(["path to spatial",
                            " transcriptomics count file.",
                            " Shoul be on form",
                            " n_spots x n_genes"]))

parser.add_argument('--st_saved_data',
                    required = False,
                    type = str,
                    default=None,
                    help = 'path to saved st data obj')

parser.add_argument('-image','--st_image_file',
                    required = False,
                    nargs = '+',
                    help = ''.join(["path to spatial",
                            " transcriptomics image file."]))

parser.add_argument('-sts','--st_source',
                    required = True,
                    type = str,
                    help = 'source of st file. Now supported mouse_brain, xiehe')


parser.add_argument('-stm','--st_model',
                    required = False,
                    default=None,
                    type = str,
                    help = 'path to already fitted st model')

parser.add_argument('-stm2','--st_model2',
                    required = False,
                    default=None,
                    type = str,
                    help = 'path to already fitted st model2')

parser.add_argument('-stb','--st_batch_size',
                    required = False,
                    default = 512,
                    type = int,
                    help = ''.join(["batch size for",
                            " st data set",
                            ]))

parser.add_argument('-scb','--sc_batch_size',
                    required = False,
                    default = 512,
                    type = int,
                    help = ''.join(["batch size for",
                            " sc data set",
                            ]))

parser.add_argument('-ste','--st_epochs',
                    default = 10000,
                    type = int,
                    help = ''.join(["number of epochs",
                            " to be used in fitting",
                            " of spatial transcriptomics",
                            " data.",
                            ]))    

parser.add_argument('-ste2','--st_epochs2',
                    default = 10000,
                    type = int,
                    help = ''.join(["number of epochs",
                            " to be used in fitting",
                            " of spatial transcriptomics",
                            " data.",
                            ]))    

parser.add_argument('-sce1','--sc_epochs1',
                    default = 10000,
                    type = int,
                    help = ''.join(["number of epochs",
                            " to be used in fitting1",
                            " of scRNA",
                            " data.",
                            ]))    

parser.add_argument('-sce2','--sc_epochs2',
                    default = 10000,
                    type = int,
                    help = ''.join(["number of epochs",
                            " to be used in fitting2",
                            " of scRNA",
                            " data.",
                            ]))  

parser.add_argument('-o','--out_dir',
                    required = False,
                    default = None,
                    type = str,
                    help = ''.join([" full path to output",
                                    " directory. Files will",
                                    " be saved with standard ",
                                    " name and timestamp",
                                    ]))

parser.add_argument("-lr","--learning_rate",
                    required = False,
                    default = 0.01,
                    type = float,
                    help = ''.join([f"learning rate to be",
                                    f" used."
                                    ]))
                                    
parser.add_argument('-gp','--gpu',
                    required = False,
                    action = 'store_true',
                    default = False,
                    help = ''.join(["use gpu",
                                    ]))

parser.add_argument('--Tsg_learn_from_scRNA',
                    required = False,
                    action = 'store_true',
                    default = False,
                    help = 'Tsg_learn_from_scRNA')

parser.add_argument('--use_image',
                    required = False,
                    action = 'store_true',
                    default = False,
                    help = 'use_image')

parser.add_argument('-stl','--st_label',
                    required = False,
                    default = None,
                    nargs = '+',
                    help = 'path to ST labels file.')



args = parser.parse_args()

if args.out_dir is None:
    args.out_dir = getcwd()

out_dir=args.out_dir
makedirs(out_dir,exist_ok=True)
makedirs(out_dir+"/Tsg/",exist_ok=True)
    

timestamp = utils.generate_identifier()
print(timestamp)

if args.gpu:
    device = t.device('cuda')
else:
    device = t.device('cpu')

device = (device if is_available() else t.device('cpu'))

cnt_pths = (args.st_cnt if \
                isinstance(args.st_cnt,list) else \
                [args.st_cnt])

img_pths = (args.st_image_file if \
isinstance(args.st_image_file,list) else \
[args.st_image_file])

lbl_pths = (args.st_label if \
isinstance(args.st_label,list) else \
[args.st_label])


if cnt_pths[0] is not None:
    # generate identifiying tag for each section
    sectiontag = list(map(lambda x: '.'.join(osp.basename(x).split('.')[0:-1]),cnt_pths))


sc_data = D.make_sc_dataset(cnt_pth=args.sc_cnt,
                            lbl_pth=args.sc_labels,
                            min_counts=1,min_cells=3,
                            sc_given_layer=args.sc_given_layers,
                            transpose=args.sc_transpose)

if args.st_saved_data is not None:
    fileObj = open(args.st_saved_data, 'rb')
    st_data = pickle.load(fileObj)
    fileObj.close()
else:
    st_data =  D.make_st_dataset(cnt_pths=cnt_pths,
                                 dataset_source=args.st_source,
                                 lbl_pths=lbl_pths,
                                 image_pths=img_pths,
                                 transpose=args.st_transpose)
    fileObj = open(out_dir+"/st_data.obj", 'wb')
    pickle.dump(st_data,fileObj)
    fileObj.close()

sc_old=args.sc_from_model
sc_old_with_coord=args.sc2_from_model
st_old=args.st_model
st_old2=args.st_model2


sc_data.intersect(st_data.genes)
st_data.intersect(sc_data.genes)

if args.Tsg_learn_from_scRNA:
    st_res=fit_st_layer.fit_scRNA_data_layer(sc_data=sc_data,
                    sc1_epochs=args.sc_epochs1,
                    sc2_epochs=args.sc_epochs2,
                    sc_batch_size=args.sc_batch_size,
                    st1_epochs=args.st_epochs,
                    learning_rate=args.learning_rate,
                    silent_mode =False,
                    sc_from_model=sc_old,   
                    sc_from_model_with_coord=sc_old_with_coord,
                    st_data = st_data,
                    device=device,
                    st_batch_size=args.st_batch_size,
                    st_from_model = st_old,
                    use_image=args.use_image,
                    out_dir=out_dir,)
else:
    if args.use_image:
        st_res=fit_st_layer.fit_st_superres_data_layer(sc_data=sc_data,
                        sc1_epochs=args.sc_epochs1,
                        sc_batch_size=args.sc_batch_size,
                        st1_epochs=args.st_epochs,
                        st2_epochs=args.st_epochs2,
                        learning_rate=args.learning_rate,
                        sc_from_model=sc_old,   
                        st_data = st_data,
                        device=device,
                        st_batch_size=args.st_batch_size,
                        st_from_model = st_old,
                        st_from_model2 = st_old2,
                        use_image=True,
                        out_dir=out_dir,)

    else:
        st_res=fit_st_layer.fit_st_data_layer(sc_data=sc_data,
                        sc1_epochs=args.sc_epochs1,
                        sc_batch_size=args.sc_batch_size,
                        st1_epochs=args.st_epochs,
                        learning_rate=args.learning_rate,
                        sc_from_model=sc_old,   
                        st_data = st_data,
                        device=device,
                        st_batch_size=args.st_batch_size,
                        st_from_model = st_old,
                        use_image=False,
                        out_dir=out_dir,)


W,st_model,sc_model = st_res['proportions'],st_res['st-model'],st_res['sc-model']
wlist = utils.split_joint_matrix(W)

# save st model
#oname_st_model = osp.join(out_dir,'.'.join(['st_model',timestamp,'pt']))


if args.prefix is not None:
    prefix = args.prefix
else:
    prefix = '.'.join(["W",timestamp,'tsv'])

t.save(st_model.state_dict(),osp.join(out_dir,prefix+".pt"))
# save st data proportion estimates results
for s in range(len(wlist)):
    section_dir = osp.join(out_dir,sectiontag[s])
    if not osp.exists(section_dir):
        mkdir(section_dir)
    oname_W = osp.join(section_dir,'.'.join([prefix,'tsv']))
    #log.info("saving proportions for section {} to {}".format(sectiontag[s],oname_W))
    utils.write_file(wlist[s],oname_W)

import torch.nn as nn
layer_embed=st_model.layer_embed.eval().cpu()
Tsg=layer_embed(st_data.zidx.unique())
st_model.R

for i in range(st_model.R.shape[1]):
    pd.DataFrame((nn.functional.softplus(
    st_model.R.repeat(st_data.zidx.unique().shape[0],1,1).transpose(0,1).cpu()+
    Tsg.repeat(st_model.R.shape[1],1,1).transpose(0,2).cpu().detach()).numpy())[:,:,i].transpose(1,0),
    columns=sc_data.genes,index=st_data.zidx.unique().numpy()).to_csv(out_dir+"/Tsg/Tsg."+prefix+".celltype"+str(i)+".txt")

pd.DataFrame(t.sigmoid(st_model.o).cpu().detach().numpy(),index=sc_data.genes).to_csv(out_dir+"/Tsg/logit."+prefix+".txt")
#pd.DataFrame(nn.functional.logsigmoid(st_model.o).cpu().detach().numpy(),index=sc_data.genes).to_csv(out_dir+"/Tsg/logit."+timestamp+".txt")
pd.DataFrame(st_model.R.cpu().detach().numpy(),index=sc_data.genes,columns=sc_data.unique_labels()).to_csv(out_dir+"/Tsg/R"+prefix+".txt")
pd.DataFrame(Tsg.cpu().detach().numpy(),columns=sc_data.genes,index=st_data.zidx.unique().numpy()).to_csv(out_dir+"/Tsg/Tsg."+prefix+"..txt")
pd.DataFrame(nn.functional.softplus(st_model.beta).cpu().detach().numpy(),index=sc_data.genes).to_csv(out_dir+"/Tsg/Beta."+prefix+".txt")

###################################################################################################################################################