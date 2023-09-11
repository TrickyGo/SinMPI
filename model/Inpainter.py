import sys
sys.path.append(".")
sys.path.append("..")
import os
import glob
import numpy as np
from skimage.feature import canny
import torch
import torch.nn.functional as F
from torchvision import transforms

from warpback.networks import get_edge_connect

class InpaintingModule(torch.nn.Module):
    def __init__(
        self,
        data_root='',
        width=512,
        height=512,
        depth_dir_name="dpt_depth",
        device="cuda:0", 
        trans_range={"x":0.2, "y":-1, "z":-1, "a":-1, "b":-1, "c":-1},  # xyz for translation, abc for euler angle
        ec_weight_dir="warpback/ecweight",
    ):
        super(InpaintingModule, self).__init__()
        self.data_root = data_root
        self.depth_dir_name = depth_dir_name
        self.width = width
        self.height = height
        self.device = device
        self.trans_range = trans_range
        self.image_path_list = glob.glob(os.path.join(self.data_root, "*.jpg"))
        self.image_path_list += glob.glob(os.path.join(self.data_root, "*.png"))

        self.edge_model, self.inpaint_model, self.disp_model = get_edge_connect(ec_weight_dir)
        self.edge_model = self.edge_model.to(self.device)
        self.inpaint_model = self.inpaint_model.to(self.device)
        self.disp_model = self.disp_model.to(self.device)

    def preprocess_rgbd(self, image, disp):
        image = F.interpolate(image.unsqueeze(0), (self.height, self.width), mode="bilinear").squeeze(0)
        disp = F.interpolate(disp.unsqueeze(0), (self.height, self.width), mode="bilinear").squeeze(0)
        return image, disp

    def forward(self, image, disp, mask):

        image_gray = transforms.Grayscale()(image)
        edge = self.get_edge(image_gray, mask)
        
        mask_hole = 1 - mask

        # inpaint edge
        edge_model_input = torch.cat([image_gray, edge, mask_hole], dim=1)  # [b,4,h,w]
        edge_inpaint = self.edge_model(edge_model_input)  # [b,1,h,w]

        # inpaint RGB
        inpaint_model_input = torch.cat([image + mask_hole, edge_inpaint], dim=1)
        image_inpaint = self.inpaint_model(inpaint_model_input)
        image_merged = image * (1 - mask_hole) + image_inpaint * mask_hole

        # inpaint Disparity
        disp_model_input = torch.cat([disp + mask_hole, edge_inpaint], dim=1)
        disp_inpaint = self.disp_model(disp_model_input)
        disp_merged = disp * (1 - mask_hole) + disp_inpaint * mask_hole

        return image_merged, disp_merged

    def get_edge(self, image_gray, mask):
        image_gray_np = image_gray.squeeze(1).cpu().numpy()  # [b,h,w]
        mask_bool_np = np.array(mask.squeeze(1).cpu(), dtype=np.bool_)  # [b,h,w]
        edges = []
        for i in range(mask.shape[0]):
            cur_edge = canny(image_gray_np[i], sigma=2, mask=mask_bool_np[i])
            edges.append(torch.from_numpy(cur_edge).unsqueeze(0))  # [1,h,w]
        edge = torch.cat(edges, dim=0).unsqueeze(1).float()  # [b,1,h,w]
        return edge.to(self.device)

