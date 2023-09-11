import torch

class MultiPlaneField(torch.nn.Module):
    def __init__(
        self,
        image_size=(256, 384),
        num_planes=12,
        assign_origin_planes=None,
        depth_range=None
    ):
        super(MultiPlaneField, self).__init__()
        # x:horizontal axis, y:vertical axis, z:inward axis

        self.num_planes = num_planes
        self.near, self.far = depth_range

        (self.plane_h, self.plane_w) = image_size

        self.planes_disp = torch.linspace(self.near, self.far, num_planes, requires_grad=False).unsqueeze(0).cuda() # [b,s]
        #[S:num_planes, H:plane_h , W:plane_w, 4:rgb+transparency]

        self.extrapolate_RGBDs = assign_origin_planes

        init_planes  = self.assign_image_to_planes(self.extrapolate_RGBDs[0], self.extrapolate_RGBDs[1])
        self.planes_mid2 = init_planes

        init_val = torch.ones(1, num_planes, 4, self.plane_h, self.plane_w) * 0.1
        init_val[:,:,:4,:,:] *= 0.01

        self.planes_residual = torch.nn.Parameter(init_val)


    def assign_image_to_planes(self, ref_img, ref_disp): 
        planes = torch.zeros(1, self.num_planes, 4, self.plane_h, self.plane_w, requires_grad=False).cuda()
        # set ref_img alpha channels all ones
        ref_img = torch.cat([ref_img, torch.ones_like(ref_disp) * self.far],dim=1) #[1,3+1,h,w]
        depth_levels = torch.linspace(self.near, self.far, self.num_planes).cuda()     

        planes_masks = []
        for i in range(len(depth_levels)):
            cur_depth_mask = torch.where(ref_disp < depth_levels[i],
                                         torch.ones_like(ref_disp).cuda(),
                                         torch.zeros_like(ref_disp).cuda())
            planes_masks.append(cur_depth_mask)
            cur_depth_pixels = ref_img * cur_depth_mask.repeat(1,4,1,1)
            cur_depth_pixels = cur_depth_pixels.unsqueeze(0)
            planes[:,i:i+1,:,:,:] = cur_depth_pixels#[1,1,4,h,w]

            ref_disp = ref_disp + cur_depth_mask * (self.far + 1)# the cur_masked area are discarded

        return planes

    def forward(self):
        planes = self.planes_mid2
        pred_planes = self.planes_residual
        
        residual_mask = torch.where(planes > 0,
                                    torch.zeros_like(planes).cuda(), 
                                    torch.ones_like(planes).cuda())

        x = pred_planes * residual_mask + planes 
        return x
