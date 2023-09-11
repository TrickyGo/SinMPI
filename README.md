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
