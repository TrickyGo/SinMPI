import torch

from mpi.homography_sampler import HomographySample
from mpi.rendering_utils import transform_G_xyz


def render(rgb_BS3HW, sigma_BS1HW, xyz_BS3HW):
    imgs_syn, weights = plane_volume_rendering(sigma_BS1HW, rgb_BS3HW)
    return imgs_syn

def plane_volume_rendering(alpha_BK1HW, value_BKCHW):
    B, K, _, H, W = alpha_BK1HW.size()
    alpha_comp_cumprod = torch.cumprod(1 - alpha_BK1HW, dim=1)  # BxKx1xHxW
    preserve_ratio = torch.cat((torch.ones((B, 1, 1, H, W), dtype=alpha_BK1HW.dtype, device=alpha_BK1HW.device),
                                alpha_comp_cumprod[:, 0:K-1, :, :, :]), dim=1)  # BxKx1xHxW
    weights = alpha_BK1HW * preserve_ratio  # BxKx1xHxW
    value_composed = torch.sum(value_BKCHW * weights, dim=1, keepdim=False)  # Bx3xHxW
    return value_composed, weights

def get_src_xyz_from_plane_disparity(meshgrid_src_homo,
                                     mpi_disparity_src,
                                     K_src_inv):
    """

    :param meshgrid_src_homo: 3xHxW
    :param mpi_disparity_src: BxS
    :param K_src_inv: Bx3x3
    :return:
    """
    B, S = mpi_disparity_src.size()
    H, W = meshgrid_src_homo.size(1), meshgrid_src_homo.size(2)
    mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS

    K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, 3, 3)

    # 3xHxW -> BxSx3xHxW
    meshgrid_src_homo = meshgrid_src_homo.unsqueeze(0).unsqueeze(1).repeat(B, S, 1, 1, 1)
    meshgrid_src_homo_Bs3N = meshgrid_src_homo.reshape(B * S, 3, -1)
    xyz_src = torch.matmul(K_src_inv_Bs33, meshgrid_src_homo_Bs3N)  # BSx3xHW
    xyz_src = xyz_src.reshape(B, S, 3, H * W) * mpi_depth_src.unsqueeze(2).unsqueeze(3)  # BxSx3xHW
    xyz_src_BS3HW = xyz_src.reshape(B, S, 3, H, W)

    return xyz_src_BS3HW


def get_tgt_xyz_from_plane_disparity(xyz_src_BS3HW,
                                     G_tgt_src):
    """

    :param xyz_src_BS3HW: BxSx3xHxW
    :param G_tgt_src: Bx4x4
    :return:
    """
    B, S, _, H, W = xyz_src_BS3HW.size()
    G_tgt_src_Bs33 = G_tgt_src.unsqueeze(1).repeat(1, S, 1, 1).reshape(B*S, 4, 4)
    xyz_tgt = transform_G_xyz(G_tgt_src_Bs33, xyz_src_BS3HW.reshape(B*S, 3, H*W))  # Bsx3xHW
    xyz_tgt_BS3HW = xyz_tgt.reshape(B, S, 3, H, W)  # BxSx3xHxW
    return xyz_tgt_BS3HW


def render_tgt_rgb_depth(H_sampler: HomographySample,
                         mpi_rgb_src,
                         mpi_sigma_src,
                         mpi_disparity_src,
                         xyz_tgt_BS3HW,
                         G_tgt_src,
                         K_src_inv, K_tgt,
                         only_render_in_fov=False,
                         center_top_left=None,
                         infov_size=512):
    """
    :param H_sampler:
    :param mpi_rgb_src: BxSx3xHxW
    :param mpi_sigma_src: BxSx1xHxW
    :param mpi_disparity_src: BxS
    :param xyz_tgt_BS3HW: BxSx3xHxW
    :param G_tgt_src: Bx4x4
    :param K_src_inv: Bx3x3
    :param K_tgt: Bx3x3
    :return:
    """
    B, S, _, H, W = mpi_rgb_src.size()
    mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS

    # note that here we concat the mpi_src with xyz_tgt, because H_sampler will sample them for tgt frame
    # mpi_src is the same in whatever frame, but xyz has to be in tgt frame
    mpi_xyz_src = torch.cat((mpi_rgb_src, mpi_sigma_src, xyz_tgt_BS3HW), dim=2)  # BxSx(3+1+3)xHxW

    # homography warping of mpi_src into tgt frame
    G_tgt_src_Bs44 = G_tgt_src.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 4, 4)  # Bsx4x4
    K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 3, 3)  # Bsx3x3
    K_tgt_Bs33 = K_tgt.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 3, 3)  # Bsx3x3

    # BsxCxHxW, BsxHxW
    tgt_mpi_xyz_BsCHW, tgt_mask_BsHW = H_sampler.sample(mpi_xyz_src.view(B*S, 7, H, W),
                                                        mpi_depth_src.view(B*S),
                                                        G_tgt_src_Bs44,
                                                        K_src_inv_Bs33,
                                                        K_tgt_Bs33)

    # mpi composition
    if only_render_in_fov:
        tgt_mpi_xyz = tgt_mpi_xyz_BsCHW.view(B, S, 7, H, W)
        tgt_rgb_BS3HW = tgt_mpi_xyz[:, :, 0:3, center_top_left[0]:center_top_left[0] + infov_size, center_top_left[1]:center_top_left[1] + infov_size]
        tgt_sigma_BS1HW = tgt_mpi_xyz[:, :, 3:4, center_top_left[0]:center_top_left[0] + infov_size, center_top_left[1]:center_top_left[1] + infov_size]
        tgt_xyz_BS3HW = tgt_mpi_xyz[:, :, 4:, center_top_left[0]:center_top_left[0] + infov_size, center_top_left[1]:center_top_left[1] + infov_size]
        H = W = infov_size
    else:
        tgt_mpi_xyz = tgt_mpi_xyz_BsCHW.view(B, S, 7, H, W)
        tgt_rgb_BS3HW = tgt_mpi_xyz[:, :, 0:3, :, :]
        tgt_sigma_BS1HW = tgt_mpi_xyz[:, :, 3:4, :, :]
        tgt_xyz_BS3HW = tgt_mpi_xyz[:, :, 4:, :, :]

    # Bx3xHxW, Bx1xHxW, Bx1xHxW
    tgt_z_BS1HW = tgt_xyz_BS3HW[:, :, -1:]
    tgt_sigma_BS1HW = torch.where(tgt_z_BS1HW >= 0,
                                  tgt_sigma_BS1HW,
                                  torch.zeros_like(tgt_sigma_BS1HW, device=tgt_sigma_BS1HW.device))
    tgt_rgb_syn = render(tgt_rgb_BS3HW, tgt_sigma_BS1HW, tgt_xyz_BS3HW)

    return tgt_rgb_syn
