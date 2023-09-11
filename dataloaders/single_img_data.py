import sys
sys.path.append(".")
sys.path.append("..")
import math
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

def gen_swing_path(init_pose=torch.eye(4), num_frames=10, r_x=0.2, r_y=0, r_z=-0.2):
    "Return a list of matrix [4, 4]"
    t = torch.arange(num_frames) / (num_frames - 1)
    poses = init_pose.repeat(num_frames, 1, 1)

    swing = torch.eye(4).repeat(num_frames, 1, 1)
    swing[:, 0, 3] = r_x * torch.sin(2. * math.pi * t)
    swing[:, 1, 3] = r_y * torch.cos(2. * math.pi * t)
    swing[:, 2, 3] = r_z * (torch.cos(2. * math.pi * t) - 1.)

    for i in range(num_frames):
        poses[i, :, :] = poses[i, :, :] @ swing[i, :, :]
    return list(poses.unbind())

def create_spheric_poses_along_y(n_poses=10):

    def spheric_pose_y(phi, radius=10):
        trans_t = lambda t : np.array([
            [1,0,0, math.sin(2. * math.pi * t) * radius],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
        ])

        # rotation along y
        rot_phi = lambda phi : np.array([
            [np.cos(phi),0, np.sin(phi),0],
            [0,1,0,0],
            [-np.sin(phi),0,np.cos(phi),0],
            [0,0,0,1],
        ])

        c2w =  rot_phi(phi) @ trans_t(phi)
        c2w = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]]) @ c2w
        c2w = torch.tensor(c2w).float()
        return c2w

    def spheric_pose_x(phi, radius=10):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,math.sin(2. * math.pi * t) * radius * -1],
            [0,0,1,0],
            [0,0,0,1],
        ])

        # rotation along x
        rot_theta = lambda th : np.array([
            [1,0,0,0],
            [0,np.cos(th),-np.sin(th),0],
            [0,np.sin(th), np.cos(th),0],
            [0,0,0,1],
        ])

        c2w =  rot_theta(phi) @ trans_t(phi)
        c2w = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]]) @ c2w
        c2w = torch.tensor(c2w).float()
        return c2w

    spheric_poses = []
    poses = gen_swing_path()
    spheric_poses += poses

    factor = 1
    y_angle = (1/16) * np.pi * factor
    x_angle = (1/16) * np.pi * factor
    x_radius = 0.1 * factor
    y_radius = 0.1 * factor

    # rotate left and right
    for th in np.linspace(0, -1 * y_angle, n_poses//2):
        spheric_poses += [spheric_pose_y(th, y_radius)]

    poses = gen_swing_path(spheric_poses[-1])
    spheric_poses += poses

    for th in np.linspace(-1 * y_angle, y_angle, n_poses)[:-1]:
        spheric_poses += [spheric_pose_y(th, y_radius)] 
    
    poses = gen_swing_path(spheric_poses[-1])
    spheric_poses += poses

    for th in np.linspace(y_angle, 0, n_poses//2)[:-1]:
        spheric_poses += [spheric_pose_y(th, y_radius)] 

    # rotate up and down
    for th in np.linspace(0, -1 * x_angle, n_poses//2):
        spheric_poses += [spheric_pose_x(th, x_radius)]

    poses = gen_swing_path(spheric_poses[-1])
    spheric_poses += poses

    for th in np.linspace(-1 * x_angle, x_angle, n_poses)[:-1]:
        spheric_poses += [spheric_pose_x(th, x_radius)] 
    
    poses = gen_swing_path(spheric_poses[-1])
    spheric_poses += poses

    for th in np.linspace(x_angle, 0, n_poses//2)[:-1]:
        spheric_poses += [spheric_pose_x(th, x_radius)] 

    return spheric_poses


def convert(c2w, phi=0):
    # rot_along_y
    c2w = np.concatenate((c2w, np.array([[0, 0, 0, 1]])), axis=0)
    rot = np.array([
            [np.cos(phi),0, np.sin(phi),0],
            [0,1,0,0],
            [-np.sin(phi),0,np.cos(phi),0],
            [0,0,0,1],
           ])    
    return rot @ c2w

class SinImgDataset(Dataset):
    def __init__(
        self,
        img_path,
        width=512,
        height=512,
        repeat_times = 1
    ):
        self.repeat_times = repeat_times
        self.img_wh = [width, height]
        self.transform = transforms.ToTensor()
        self.img_path = img_path
        self.read_meta()

    def read_meta(self):

        self.all_rgbs = []
        self.all_poses = []
        self.all_poses += create_spheric_poses_along_y()

        img_path = "test_images/" + self.img_path
        self.ref_img = self.load_img(img_path)


    def load_img(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img) # (3, h, w)
        img = img.unsqueeze(0).cuda()
        return img # [img:1*3*h*w]


    def __len__(self):
        return len(self.all_poses) * self.repeat_times

    def __getitem__(self, idx):
        sample = {
                  'cur_pose':self.all_poses[idx % len(self.all_poses)]
                  }
        return sample
