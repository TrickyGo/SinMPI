#conda activate SinMPI
cuda=0
width=512
height=512
data_name='Syndney'
extrapolate_times=2

ckpt_path='Exp-Syndney'
img_path=$data_name'.jpg'

CUDA_VISIBLE_DEVICES=$cuda python outpaint_rgbd.py \
    --width $width \
    --height $height \
    --ckpt_path $ckpt_path \
    --img_path $img_path \
    --extrapolate_times $extrapolate_times

CUDA_VISIBLE_DEVICES=$cuda python train_inpainting.py \
     --width $width \
     --height $height \
     --ckpt_path $ckpt_path \
     --img_path $img_path \
     --num_epochs 10  \
     --extrapolate_times $extrapolate_times \
     --batch_size 1  #--load_warp_pairs --debugging 

CUDA_VISIBLE_DEVICES=$cuda python train_mpi.py \
    --width $width \
    --height $height \
    --ckpt_path $ckpt_path \
    --img_path $img_path \
    --num_epochs 10 \
    --extrapolate_times $extrapolate_times \
    --batch_size 1 #--debugging #--resume