import argparse
import torch
import torch.nn.functional as F
import math
import random

from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from dataloaders.single_img_data import SinImgDataset
from model.Inpainter import InpaintingModule
from torchvision.utils import save_image
from utils_for_train import VGGPerceptualLoss
from model.VitExtractor import VitExtractor
from utils_for_train import tensor_to_depth
from warpback.utils import (
    RGBDRenderer, 
    image_to_tensor, 
    transformation_from_parameters,
)
from pytorch_lightning import seed_everything

class TrainInpaintingModule(LightningModule):
    def __init__(self, opt):
        super(TrainInpaintingModule, self).__init__()

        self.opt = opt
        self.loss = []

        W, H = self.opt.width, self.opt.height
        self.save_base_dir = f'ckpts/{opt.ckpt_path}'

        if self.opt.resume:
            self.inpaint_module = InpaintingModule()
            self.inpaint_module.load_state_dict(torch.load(f'ckpts/{self.opt.ckpt_path}/inpaint_latest.pt'), strict=True)
        else:
            self.inpaint_module = InpaintingModule()

        self.models = [self.inpaint_module.cuda()]

        # for training
        self.extrapolate_times = self.opt.extrapolate_times
        self.train_dataset = SinImgDataset(img_path=self.opt.img_path, width=W, height=H, repeat_times=1)

        if self.extrapolate_times == 3: # extend w = 3 * w
            self.center_top_left = (self.opt.height, self.opt.width)
        elif self.extrapolate_times == 2:  # extend w = 2 * w
            self.center_top_left = (self.opt.height//2, self.opt.width//2)
        elif self.extrapolate_times == 1:
            self.center_top_left = (0, 0)

        self.K = torch.tensor([
            [0.58, 0, 0.5],
            [0, 0.58, 0.5],
            [0, 0, 1]
            ])

        with torch.no_grad():
            if self.extrapolate_times == 1:
                ref_img = image_to_tensor(self.save_base_dir + "/" + "canvas.png", unsqueeze=False)  # [3,h,w]
                ref_img = ref_img.unsqueeze(0).cuda()
                if ref_img.shape[1] == 4:
                    ref_img = ref_img[:,:3,:,:]

                ref_depth = tensor_to_depth(ref_img)
                save_image(ref_depth[0,0,...], self.save_base_dir + "/" + "canvas_depth.png")
            else:
                ref_img, ref_depth = torch.load(self.save_base_dir + "/" + "extrapolate_RGBDs.pkl")

            ref_depth = (ref_depth - ref_depth.min())/(ref_depth.max() - ref_depth.min())

            self.extrapolate_RGBDs = (ref_img.cpu(), ref_depth.cpu())

            if self.opt.load_warp_pairs:
                self.inpaint_pairs = torch.load(self.save_base_dir + "/" + "inpaint_pairs.pkl")
            else:
                self.renderer = RGBDRenderer('cuda:0')
                self.inpaint_pairs = self.get_pairs()
                torch.save(self.inpaint_pairs,self.save_base_dir + "/" + "inpaint_pairs.pkl")

            self.perceptual_loss = VGGPerceptualLoss()
            self.VitExtractor = VitExtractor(
                    model_name='dino_vits16', device='cuda:0')
            
            self.renderer_pair_saved = False


    def configure_optimizers(self):
        from torch.optim import SGD, Adam

        parameters = []
        for model in self.models:
            parameters += list(model.parameters())
        self.optimizer = Adam(parameters, lr=5e-4, eps=1e-8, weight_decay=0)

        return [self.optimizer], []


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=self.opt.batch_size,
                          pin_memory=True)


    def get_rand_ext(self, bs=1):
        def rand_tensor(r, l):
            if r < 0:  
                return torch.zeros((l, 1, 1))
            rand = torch.rand((l, 1, 1))        
            sign = 2 * (torch.randn_like(rand) > 0).float() - 1
            return sign * (r / 2 + r / 2 * rand)

        trans_range={"x":0.2, "y":-0.2, "z":-0.2, "a":-0.2, "b":-0.2, "c":-0.2}
        x, y, z = trans_range['x'], trans_range['y'], trans_range['z']
        a, b, c = trans_range['a'], trans_range['b'], trans_range['c']
        cix = rand_tensor(x, bs)
        ciy = rand_tensor(y, bs)
        ciz = rand_tensor(z, bs)
        aix = rand_tensor(math.pi / a, bs)
        aiy = rand_tensor(math.pi / b, bs)
        aiz = rand_tensor(math.pi / c, bs)
        
        axisangle = torch.cat([aix, aiy, aiz], dim=-1)  # [b,1,3]
        translation = torch.cat([cix, ciy, ciz], dim=-1)
        
        cam_ext = transformation_from_parameters(axisangle, translation)  # [b,4,4]
        cam_ext_inv = torch.inverse(cam_ext)  # [b,4,4]
        return cam_ext, cam_ext_inv

    def get_pairs(self):
        all_poses = self.train_dataset.all_poses

        aug_pose_factor = 0 # set pose augmentation for better results
        cnt = len(all_poses)
        if aug_pose_factor > 0:
            for i in range(cnt):
                cur_pose = torch.FloatTensor(all_poses[i])
                for _ in range(aug_pose_factor):
                    cam_ext, cam_ext_inv = self.get_rand_ext()  # [b,4,4]
                    cur_aug_pose = torch.matmul(cam_ext, cur_pose)
                    all_poses += [cur_aug_pose]

        ref_depth = self.extrapolate_RGBDs[1]
            
        ref_img = self.extrapolate_RGBDs[0]
        W, H = self.opt.width * self.extrapolate_times, self.opt.height * self.extrapolate_times

        inpaint_pairs = []  #(warp_back_image, warp_back_disp, warp_back_mask, ref_img, ref_depth)
        val_pairs = [] #(cam_ext, ref_img, warp_image, warp_disp, warp_mask, gt_img)

        print("all_poses len:",len(all_poses))
        for i, cur_pose in enumerate(all_poses[:]):
            cur_pose = all_poses[i]
            c2w = cur_pose
            c2w = torch.FloatTensor(c2w)

            cam_int = self.K.repeat(1, 1, 1)  # [b,3,3]

            #load cam_ext
            cam_ext = c2w
            cam_ext_inv = torch.inverse(cam_ext)
            cam_ext = cam_ext.repeat(1, 1, 1)[:,:-1,:]
            cam_ext_inv = cam_ext_inv.repeat(1, 1, 1)[:,:-1,:]

            rgbd = torch.cat([ref_img, ref_depth], dim=1).cuda()
            cam_int = cam_int.cuda()
            cam_ext = cam_ext.cuda()
            cam_ext_inv = cam_ext_inv.cuda()

            # warp to a random novel view
            mesh = self.renderer.construct_mesh(rgbd, cam_int)
            warp_image, warp_disp, warp_mask = self.renderer.render_mesh(mesh, cam_int, cam_ext)
            
            # warp back to the original view
            warp_rgbd = torch.cat([warp_image, warp_disp], dim=1)  # [b,4,h,w]
            warp_mesh = self.renderer.construct_mesh(warp_rgbd, cam_int)
            warp_back_image, warp_back_disp, warp_back_mask = self.renderer.render_mesh(warp_mesh, cam_int, cam_ext_inv)

            ref_depth_2 = ref_depth
            # all depth should be in [0~1]
            inpaint_pairs.append((ref_img, ref_depth_2, cur_pose,
                                    warp_image, warp_disp, warp_mask,
                                    warp_back_image, warp_back_disp, warp_back_mask))
            print("collecting inpaint_pairs:", len(inpaint_pairs))

        return inpaint_pairs


    def forward(self, renderer_pair):
        (ref_img, ref_depth, cur_pose,
         warp_rgb, warp_disp, warp_mask, 
         warp_back_image, warp_back_disp, warp_back_mask)  = renderer_pair

        inpainted_warp_image, inpainted_warp_disp = self.inpaint_module(warp_rgb.cuda(), warp_disp.cuda(), warp_mask.cuda())
        inpainted_warp_back_image, inpainted_warp_back_disp = self.inpaint_module(warp_back_image.cuda(), warp_back_disp.cuda(), warp_back_mask.cuda())

        return {
            "ref_img": ref_img,
            "ref_depth": ref_depth,
            "warp_image": warp_rgb,
            "warp_disp": warp_disp,
            "inpainted_warp_image":inpainted_warp_image,
            "inpainted_warp_disp":inpainted_warp_disp,
            "warp_back_image": warp_back_image,
            "warp_back_disp": warp_back_disp,
            "inpainted_warp_back_image":inpainted_warp_back_image,
            "inpainted_warp_back_disp":inpainted_warp_back_disp,
        }


    def training_step(self, batch, batch_idx, optimizer_idx=0):
        renderer_pair = random.choice(self.inpaint_pairs)
        batch = self(renderer_pair)

        ref_img = batch["ref_img"].cuda()
        ref_depth = batch["ref_depth"].cuda()

        warp_image = batch["warp_image"]
        warp_disp = batch["warp_disp"]

        inpainted_warp_image = batch["inpainted_warp_image"]
        inpainted_warp_disp = batch["inpainted_warp_disp"]

        warp_back_image = batch["warp_back_image"]
        warp_back_disp = batch["warp_back_disp"]

        inpainted_warp_back_image = batch["inpainted_warp_back_image"]
        inpainted_warp_back_disp = batch["inpainted_warp_back_disp"]
        
        # Losses
        loss_total = 0
        loss_L1 = 0
        lambda_loss_L1 = 10

        loss_perc = 0
        lambda_loss_perc = 5

        loss_L1 += F.l1_loss(ref_img, inpainted_warp_back_image) 
        loss_total += loss_L1 * lambda_loss_L1

        loss_perc += self.perceptual_loss(inpainted_warp_image, ref_img) + self.perceptual_loss(inpainted_warp_back_image, ref_img)
        loss_total += loss_perc * lambda_loss_perc

        loss_inpainted_vit = 1e-1
        lambda_loss_vit = 0
        ref_vit_feature = self.get_vit_feature(ref_img)
        inpainted_vit_feature = self.get_vit_feature(inpainted_warp_image)
        inpainted_warp_back_vit_feature = self.get_vit_feature(inpainted_warp_back_image)
        loss_inpainted_vit += F.mse_loss(inpainted_vit_feature, ref_vit_feature) + F.mse_loss(inpainted_warp_back_vit_feature, ref_vit_feature)
        loss_total += loss_inpainted_vit * lambda_loss_vit

        loss_depth = 0
        lambda_loss_depth = 1
        loss_depth += F.l1_loss(ref_depth, inpainted_warp_back_disp) 
        loss_total += lambda_loss_depth * loss_depth

        if self.opt.debugging:
            self.training_epoch_end(None)
            assert 0

        return {'loss': loss_total}


    def training_epoch_end(self, outputs):
        with torch.no_grad():

            # pred_frames = []

            self.renderer_pairs = []                
            for i, inpaint_pair in enumerate(self.inpaint_pairs):
                (ref_img, ref_depth, cur_pose,
                 warp_image, warp_disp, warp_mask,
                 warp_back_image, warp_back_disp, warp_back_mask) = inpaint_pair
                inpainted_warp_image, inpainted_warp_disp = self.inpaint_module(warp_image.cuda(), warp_disp.cuda(), warp_mask.cuda())
               
                self.renderer_pairs += [(cur_pose, 
                                         ref_img, 
                                         inpainted_warp_image)]

            torch.save(self.inpaint_module.state_dict(), self.save_base_dir + "/" +"inpaint_latest.pt")
            if not self.renderer_pair_saved:
                torch.save(self.renderer_pairs, self.save_base_dir + "/" + "renderer_pairs.pkl")
            if self.opt.debugging:
                assert 0, "no bug"
        return

    def get_vit_feature(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=x.device).reshape(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224))
        x = (x - mean) / std
        return self.VitExtractor.get_feature_from_input(x)[-1][0, 0, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_path', type=str, default="test_images/Syndney.jpg")
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--ckpt_path', type=str, default="Exp-X")
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--resume', default=False, action="store_true")
    parser.add_argument('--batch_size', type=int, default=1)   
    parser.add_argument('--debugging', default=False, action="store_true") 
    parser.add_argument('--extrapolate_times', type=int, default=1)
    parser.add_argument('--load_warp_pairs', default=False, action="store_true")

    opt, _ = parser.parse_known_args()

    seed = 50
    seed_everything(seed)

    system = TrainInpaintingModule(opt)

    trainer = Trainer(max_epochs=opt.num_epochs,
                      progress_bar_refresh_rate=1,
                      gpus=1,
                      num_sanity_val_steps=1)

    trainer.fit(system)