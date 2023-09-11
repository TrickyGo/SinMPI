### Welcome to the Pytorch implementation of paper "SinMPI: Novel View Synthesis from a Single Image with Expanded Multiplane Images" (SIGGRAPH Asia 2023).

## Quick demo

### 1. Prepare
(1) Create a new conda environment specified in requirements.txt.
(2) Download [ecweights](https://drive.google.com/drive/folders/1FZZ6laPuqEMSfrGvEWYaDZWEPaHvGm6r) and put them in to warpback/ecweights/xxx.pth.
### 2. Run demo
```
sh scripts/train_all.sh
```
This demo converts 'test_images/Syndney.jpg' to an expanded MPI and renders novel views as in 'ckpts/Exp-Syndney-new/MPI_rendered_views.mp4'.

## What happens when running the demo?
### 1. Outpaint the input image
In the above demo, we specify 'test_images/Syndney.jpg'

<img src='test_images/Syndney.jpg' width="30%" >

as the input image, then we continuously outpaint the input image:
```
CUDA_VISIBLE_DEVICES=$cuda python outpaint_rgbd.py \
    --width $width \
    --height $height \
    --ckpt_path $ckpt_path \
    --img_path $img_path \
    --extrapolate_times $extrapolate_times
```
Then we get the outpainted image and its depth estimated by a monocular depth estimator (DPT):

<img src='ckpts/Exp-syndney/canvas.png' width="30%" > <img src='ckpts/Exp-syndney/canvas_depth.png' width="30%" >

### 2. Finetune Depth-aware Inpainter and create Pseudo-multi-view images
```
CUDA_VISIBLE_DEVICES=$cuda python train_inpainting.py \
     --width $width \
     --height $height \
     --ckpt_path $ckpt_path \
     --img_path $img_path \
     --num_epochs 10  \
     --extrapolate_times $extrapolate_times \
     --batch_size 1  #--load_warp_pairs --debugging 
```

### 3. Optimizing the expanded MPI
```
CUDA_VISIBLE_DEVICES=$cuda python train_mpi.py \
    --width $width \
    --height $height \
    --ckpt_path $ckpt_path \
    --img_path $img_path \
    --num_epochs 10 \
    --extrapolate_times $extrapolate_times \
    --batch_size 1 #--debugging #--resume
```
After optimization, we render novel views:
<img src="ckpts/Exp-syndney/MPI_rendered_views.gif" width="30%" >


