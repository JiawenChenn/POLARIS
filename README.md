# Cell composition inference and identification of layer-specific transcriptional profiles with POLARIS
This repository contains the source code of POLARIS and all the code, preprocessed data used in the analysis.

## POLARIS usage
We begin by cloning this repository. After cloning the repository, we use the following command to perform deconvolution. POLARIS will also output the layer-specific paramters. Here is an example of performing deconvolution on the developing human heart data with and without image.

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
python ./train.py \
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

# simulation

# permutation test