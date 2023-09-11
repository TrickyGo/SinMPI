import argparse
import torch
import torch.nn.functional as F
import random
from moviepy.editor import ImageSequenceClip
from utils_for_train import VGGPerceptualLoss
from dataloaders.single_img_data import SinImgDataset
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from model.MPF import MultiPlaneField
from model.VitExtractor import VitExtractor
from mpi.homography_sampler import HomographySample
from mpi import mpi_rendering
from model.TrainableFilter import TrainableFilter
from pytorch_lightning import seed_everything

class System(LightningModule):
    def __init__(self, opt):
        super(System, self).__init__()
        self.save_base_dir = f'ckpts/{opt.ckpt_path}'
        self.opt = opt
        self.loss = []
        self.models = []
        W, H = self.opt.width, self.opt.height

        self.extrapolate_times = self.opt.extrapolate_times
        self.train_dataset = SinImgDataset(img_path=self.opt.img_path, width=W, height=H, repeat_times=10)

        W, H = self.opt.width * self.extrapolate_times, self.opt.height * self.extrapolate_times
        
        self.K = torch.tensor([
            [0.58, 0, 0.5],
            [0, 0.58, 0.5],
            [0, 0, 1]
            ])
        self.K[0, :] *= W
        self.K[1, :] *= H
        self.K = self.K.unsqueeze(0)

        if self.extrapolate_times == 3: # extend w = 3 * w
            self.center_top_left = (self.opt.height, self.opt.width)
        elif self.extrapolate_times == 2:  # extend w = 2 * w
            self.center_top_left = (self.opt.height//2, self.opt.width//2)
        elif self.extrapolate_times == 1:
            self.center_top_left = (0, 0)

        with torch.no_grad():
            self.extrapolate_RGBDs = torch.load(self.save_base_dir + "/" + "extrapolate_RGBDs.pkl")
            img, depth = self.extrapolate_RGBDs
            depth = (depth - depth.min())/(depth.max() - depth.min())
            self.extrapolate_RGBDs = (img.cuda(), depth.cuda())

            # create MPI
            self.num_planes = 64
            self.MPF = MultiPlaneField(num_planes=self.num_planes,
                                    image_size=(H, W),
                                    assign_origin_planes=self.extrapolate_RGBDs,
                                    depth_range=[self.extrapolate_RGBDs[1].min()+1e-6, self.extrapolate_RGBDs[1].max()])
            self.trainable_filter = TrainableFilter(ksize=3).cuda()

            if self.opt.resume:
                self.MPF.load_state_dict(torch.load(f'ckpts/{self.opt.ckpt_path}/MPF_latest.pt'), strict=True)

            self.renderer_pairs = torch.load(self.save_base_dir + "/" + "renderer_pairs.pkl")

            self.models += [self.MPF]
            self.models += [self.trainable_filter]

        self.perceptual_loss = VGGPerceptualLoss()
        self.VitExtractor = VitExtractor(
                model_name='dino_vits16', device='cuda:0')  

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.opt.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        from torch.optim import SGD, Adam
        from torch.optim.lr_scheduler import MultiStepLR

        parameters = []
        for model in self.models:
            parameters += list(model.parameters())
    
        self.optimizer = Adam(parameters, lr=5e-4, eps=1e-8, 
                         weight_decay=0)

        scheduler = MultiStepLR(self.optimizer, milestones=[20], 
                            gamma=0.1)

        return [self.optimizer], [scheduler]  


    def training_step(self, batch, batch_idx, optimizer_idx=0):

        renderer_pair = random.choice(self.renderer_pairs)
        (cam_ext, ref_img, inpainted_warp_image) = renderer_pair
        cam_ext, ref_img, inpainted_warp_image= cam_ext.cuda(), ref_img.cuda(), inpainted_warp_image.cuda()
        # Losses
        loss = 0
        ref_vit_feature = self.get_vit_feature(ref_img)

        # run MPI forward
        frames_tensor = self(cam_ext)

        # mpi losses
        loss_L1 = F.l1_loss(frames_tensor, inpainted_warp_image) 

        lambda_loss_L1 = 10
        loss += loss_L1 * lambda_loss_L1

        lambda_loss_vit = 1e-1
        frames_vit_feature = self.get_vit_feature(frames_tensor)
        loss_vit = F.mse_loss(frames_vit_feature, ref_vit_feature)
        loss += loss_vit * lambda_loss_vit


        if self.opt.debugging:
            # loss *= 0
            self.training_epoch_end(None)
            assert 0

        return {'loss': loss}


    def forward(self, cam_ext, only_render_in_fov=False):
        # mpi_planes[b,s,4,h,w], mpi_disp[b,s]
        mpi = self.MPF()
        mpi_all_rgb_src = mpi[:, :, 0:3, :, :]
        mpi_all_sigma_src = mpi[:, :, 3:4, :, :]
        disparity_all_src = self.MPF.planes_disp
        
        k_tgt = k_src = self.K
        k_src_inv = torch.inverse(k_src).cuda()
        k_tgt = k_tgt.cuda()
        k_src = k_src.cuda()
        h, w = mpi.shape[-2:]
        homography_sampler = HomographySample(h, w, "cuda:0")
        G_tgt_src = cam_ext.cuda()

        xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
            homography_sampler.meshgrid,
            disparity_all_src,
            k_src_inv
        )

        xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
            xyz_src_BS3HW,
            G_tgt_src
        )

        tgt_imgs_syn = mpi_rendering.render_tgt_rgb_depth(
            homography_sampler,
            mpi_all_rgb_src,
            mpi_all_sigma_src,
            disparity_all_src,
            xyz_tgt_BS3HW,
            G_tgt_src,
            k_src_inv,
            k_tgt,
            only_render_in_fov=only_render_in_fov,
            center_top_left=self.center_top_left
        )

        tgt_imgs_syn = self.trainable_filter(tgt_imgs_syn)

        return tgt_imgs_syn

    def training_epoch_end(self, outputs):
        with torch.no_grad():

            pred_frames = []
            for i, renderer_pair in enumerate(self.renderer_pairs):
                (cam_ext, ref_img, inpainted_warp_image) = renderer_pair
                # run MPI forward
                frames_tensor = self(cam_ext, only_render_in_fov=True) 
                cam_ext, ref_img, inpainted_warp_image = cam_ext.cuda(), ref_img.cuda(), inpainted_warp_image.cuda()
      
                pred_frame_np = frames_tensor.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()  # [b,h,w,3]
                pred_frame_np = np.clip(np.round(pred_frame_np * 255), a_min=0, a_max=255).astype(np.uint8)
                pred_frames += [pred_frame_np]

            rgb_clip = ImageSequenceClip(pred_frames, fps=10)
            save_path = f'ckpts/{self.opt.ckpt_path}/MPI_rendered_views.mp4'
            rgb_clip.write_videofile(save_path, verbose=False, codec='mpeg4', logger=None, bitrate='2000k')
            
            save_base_dir = f'ckpts/{self.opt.ckpt_path}'

            torch.save(self.MPF.state_dict(), save_base_dir + "/" +"MPF_latest.pt")

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
    parser.add_argument('--batch_size', type=int, default=1)   
    parser.add_argument('--debugging', default=False, action="store_true") 
    parser.add_argument('--resume', default=False, action="store_true")
    parser.add_argument('--extrapolate_times', type=int, default=1) 
    opt, _ = parser.parse_known_args()

    seed = 50
    seed_everything(seed)

    system = System(opt)

    trainer = Trainer(max_epochs=opt.num_epochs,
                      resume_from_checkpoint=opt.resume_path,
                      progress_bar_refresh_rate=1,
                      gpus=1,
                      num_sanity_val_steps=1)

    trainer.fit(system)