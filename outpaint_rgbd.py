import argparse
import torch
import random
import os
import torchvision
import torchvision.transforms as T

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer

from dataloaders.single_img_data import SinImgDataset
from diffusers import StableDiffusionInpaintPipeline
from utils_for_train import tensor_to_depth
from pytorch_lightning import seed_everything

class SDOutpainter(LightningModule):
    def __init__(self, opt):
        super(SDOutpainter, self).__init__()

        self.opt = opt
        self.loss = []

        W, H = self.opt.width, self.opt.height
        self.save_base_dir = f'ckpts/{opt.ckpt_path}'
        if not os.path.exists(self.save_base_dir):
            os.makedirs(self.save_base_dir)

        self.dataset_type = 'SinImgDataset'
        self.train_dataset = SinImgDataset(img_path=self.opt.img_path, width=W, height=H, repeat_times=1)
        self.extrapolate_times = self.opt.extrapolate_times

        if self.extrapolate_times == 3: # extend w = 3 * w
            self.center_top_left = (self.opt.height, self.opt.width)
        elif self.extrapolate_times == 2:  # extend w = 2 * w
            self.center_top_left = (self.opt.height//2, self.opt.width//2)
        elif self.extrapolate_times == 1:
            self.center_top_left = (0, 0)

        with torch.no_grad():
            self.sd = StableDiffusionInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-inpainting",
                    torch_dtype=torch.float16,
                    local_files_only=True,
                    use_auth_token=""
                ).to("cuda:0")
            self.extrapolate_RGBDs = self.gen_extrapolate_RGBDs()
            torch.save(self.extrapolate_RGBDs, self.save_base_dir + "/" + "extrapolate_RGBDs.pkl")

    def gen_extrapolate_RGBDs(self):

        self.prompt = ["continuous sky, without animal, without text, without copy", #for up
                       "continuous scene, without text, without copy", #for mid
                       "continuous sea, without text" #for down
                       ]

        ref_img = self.train_dataset.ref_img.cpu()
        depth = tensor_to_depth(ref_img.cuda()).cpu()

        ref_depth = depth

        rgbd = (ref_img.cuda(), ref_depth.cuda())

        _,_,h,w = ref_img.shape

        canvas = torch.zeros(1,3,h*self.extrapolate_times,w*self.extrapolate_times)
        mask = torch.zeros(1,1,h*self.extrapolate_times,w*self.extrapolate_times)

        if self.extrapolate_times == 3: # extend w = 3 * w
            top_left_points = [
                            (h//2,w), (0,w), #top
                            (h + h//2,w), (h + h,w), #down
                            (h,w//2),(h,0), #left
                            (h, w + w//2), (h, w + w), #right

                            (h//2,w//2), (0,w//2), (h//2,0) ,(0,0), #top left
                            (h + h//2,w//2), (h + h//2,0), (h + h,w//2), (h + h,0), #down left
                            (h//2,w + w//2), (0,w + w//2), (h//2,w + w), (0 ,w + w), #top right
                            (h + h//2, w + w//2), (h + h//2, w + w), (h + h, w + w//2), (h + h, w + w), #down right 
                            ]
            up = [0,1,8,9,10,11,16,17,18,19]
            mid = [4,5,6,7]
            down = [2,3,12,13,14,15,20,21,22,23]
        elif self.extrapolate_times == 2:  # extend w = 2 * w
            top_left_points = [
                            (0,w//2), #top
                            (h,w//2), #down
                            (h//2,0), #left
                            (h//2,w), #right

                            (0,0), #top left
                            (h,0), #down left
                            (0,w), #top right
                            (h,w), #down right 
                            ]
            up = [0,4,6]
            mid = [2,3]
            down = [1,3,5,7]
        elif self.extrapolate_times == 1:
            top_left_points = []
            # return rgbd

        canvas[:,:,self.center_top_left[0]:self.center_top_left[0] + h, self.center_top_left[1]:self.center_top_left[1] + w] = ref_img
        mask[:,:,self.center_top_left[0]:self.center_top_left[0] + h, self.center_top_left[1]:self.center_top_left[1] + w] = torch.ones(1,1,h,w)

        for i, point in enumerate(top_left_points):
            canvas_part = canvas[:,:,point[0]:point[0]+ h, point[1]:point[1]+ w]
            mask_part = mask[:,:,point[0]:point[0]+ h, point[1]:point[1]+ w]
            if i in up:
                prompt = self.prompt[0]
            elif i in mid:
                prompt = self.prompt[1]
            else:
                prompt = self.prompt[2]
            canvas[:,:,point[0]:point[0]+ h, point[1]:point[1]+ w] = self.run_sd(canvas_part, mask_part, prompt, h, w)
            mask[:,:,point[0]:point[0]+ h, point[1]:point[1]+ w] = torch.ones(1,1,h,w)
        
        depth = tensor_to_depth(canvas.cuda())

        align_depth = True
        if align_depth:
            extrapolate_depth = depth
            extrapolate_center_depth = extrapolate_depth[:,:,self.center_top_left[0]:self.center_top_left[0] + h, self.center_top_left[1]:self.center_top_left[1] + w]
            # align depth with ref_depth
            depth[:,:,:,:] = (depth - extrapolate_center_depth.min())/(extrapolate_center_depth.max() - extrapolate_center_depth.min()) * (ref_depth.max() - ref_depth.min()) + ref_depth.min()

        extrapolate_RGBDs = (canvas.cuda(), depth.cuda())
        torchvision.utils.save_image(canvas[0], self.save_base_dir + "/" + "canvas.png")
        torchvision.utils.save_image(depth[0], self.save_base_dir + "/" + "canvas_depth.png")
        return extrapolate_RGBDs

    def run_sd(self, canvas, mask, prompt, w, h):
        # Run sd
        # prompt = "room"
        transform = T.ToPILImage()
        warp_rgb_PIL = transform(canvas[0,...]).convert("RGB").resize((512, 512))
        warp_mask_PIL = transform(255 * (1 - mask[0,...].to(torch.int32))).convert("RGB").resize((512, 512))
        inpainted_warp_image = self.sd(prompt=prompt, image=warp_rgb_PIL, mask_image=warp_mask_PIL).images[0]
        inpainted_warp_image = inpainted_warp_image.resize((h,w))
        inpainted_warp_image = T.ToTensor()(inpainted_warp_image).unsqueeze(0)  
        inpainted_warp_image = canvas * mask + inpainted_warp_image * (1 - mask)
        return inpainted_warp_image  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_path', type=str, default="images/0810.png")
    parser.add_argument('--disp_path', type=str, default="images/depth/0810.png")
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--ckpt_path', type=str, default="ExpX")
    parser.add_argument('--debugging', default=False, action="store_true") 
    parser.add_argument('--extrapolate_times', type=int, default=1)
    
    opt, _ = parser.parse_known_args()

    seed = 50
    seed_everything(seed)

    sd_outpainter = SDOutpainter(opt)