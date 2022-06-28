# Cell composition inference and identification of layer-specific transcriptional profiles with POLARIS
This repository contains the source code of POLARIS and all the code, preprocessed data used in the analysis. This repository is currently under construction.

## POLARIS main functionalities
POLARIS is a versatile and generally applicable method for the analysis of spatial transcriptomics (ST) data. In particular, POLARIS has the following three main functionalities:
1. estimate cell type composition of each spatial spot;
2. detect layer-specific differentially expressed (LDE) genes, where layers refer to different anatomical or functional regions;
3. infer layer sub-structures solely from histological images.

## POLARIS usage
Please begin by cloning this repository. After cloning the repository, the following commands can be used to perform deconvolution. POLARIS will also output the layer-specific parameters. Here is an example of performing deconvolution on the developing human heart data with and without image.

### no image
```console
username:~$ git clone https://github.com/JiawenChenn/POLARIS
username:~$ cd ./POLARIS/source_code
username:~$ # replace the following path by your own path
username:~$ python ./train.py \
--sc_cnt ../data/heart/ISS/ref/development_heart.scRNA.processed.cnt.genexrow.tsv  \
--sc_labels ../data/heart/ISS/ref/development_heart.scRNA.processed.mta.tsv  \
--sc_transpose  \
--st_cnt ../data/development_heart/PCW6.5_1/PCW6.5_1_st_cnt.tsv  \
--st_source simluation  \
--st_label ../data/development_heart/PCW6.5_1/heart_4layer.tsv  \
--st_batch_size 1024  \
--sc_batch_size 1024  \
--st_epochs 20000 \
--sc_epochs1 20000 \
--gpu \
--out_dir ./heart_noimage/ \
--prefix heart_noimage
```
### with image
```console
username:~$ python ./train.py \
--sc_cnt ../data/heart/ISS/ref/development_heart.scRNA.processed.cnt.genexrow.tsv  \
--sc_labels ../data/heart/ISS/ref/development_heart.scRNA.processed.mta.tsv  \
--sc_transpose  \
--st_cnt ../data/development_heart/PCW6.5_1/PCW6.5_1_st_cnt.tsv  \
--st_source human_heart  \
--st_label ../data/development_heart/PCW6.5_1/heart_4layer.tsv  \
--st_batch_size 1024  \
--sc_batch_size 1024  \
--st_epochs 20000 \
--st_epochs2 20000 \
--sc_epochs1 20000 \
--use_image \
-lr 0.001 \
--gpu \
--out_dir ./heart_image/ \
--prefix heart_image
```

# Detection of layer-specific differentially expressed (LDE) genes 
The codes used for calculate log2 fold change for the simulation results are in the down_stream folder.
