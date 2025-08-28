# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional, Tuple

import numpy as np
import torch
from torch.fft import irfftn, rfftn
from torch.nn.functional import grid_sample, log_softmax, pad,interpolate

from .metrics import angle_error
from .utils import make_grid, rotmat2d

def mask_norm(value,valid):
    """calculate mask normalization
    value: [B,C,H,W]
    valid: [B,1,H,W]
    """
    valid_sum = valid.sum(dim = (-2,-1),keepdims = True)
    mean = (value * valid).sum(dim = (-2,-1),keepdims = True) / valid_sum
    std = ((value - mean).abs() * valid).sum(dim = (-2,-1),keepdims = True) / valid_sum
    return (value - mean) / std

def crop_map(map,uv,yaw,bev_size):
    map_size = map.new_tensor(map.shape[-2:][::-1])
    grid_uv = make_grid(bev_size[0],bev_size[1],step_x = 1,step_y = 1,orig_x = -(bev_size[0] // 2),orig_y = -bev_size[1],y_up = False,device = bev_size.device)
    rotmats = rotmat2d(yaw / 180 * torch.pi)
    grid_uv_rot = torch.einsum("...nij, ...hwj -> ...nhwi", rotmats, grid_uv)
    grid_uv_map = grid_uv_rot + uv.view(-1,1,1,2)
    grid_uv_map_norm = grid_uv_map / (map_size.view(1,-1) - 1)
    grid_uv_map_norm  = grid_uv_map_norm * 2 - 1
    feats_m_crop = grid_sample(map,grid_uv_map_norm,align_corners=True, mode = 'bilinear')
    mask_map = torch.ones_like(map[:,:1,:,:])
    mask_map = grid_sample(mask_map,grid_uv_map_norm,align_corners=True, mode = 'bilinear') > (1 - 1e-4)
    
    return feats_m_crop,mask_map

def depth_loss(depth,disparity,valid):
    """calculate the depth/disparity loss
    depth: [B,1,H,W], relative depth estimation (pred)
    disparity: [B,1,H,W], disparity estimation (gt)
    valid: [B,1,H,W], valid map
    return: loss, [B,]
    """
    depth = interpolate(depth,size = valid.shape[2:],mode = 'bilinear')
    disparity = interpolate(disparity,size = valid.shape[2:],mode = 'bilinear')
    pred_disparity = 1.0 / depth
    # pred_disparity_norm = (pred_disparity - pred_disparity[valid].min()) / (pred_disparity[valid].max() - pred_disparity[valid].min())
    # disparity_norm = (disparity - disparity[valid].min()) / (disparity[valid].max() - disparity[valid].min())
    pred_disparity_norm = mask_norm(pred_disparity,valid)
    disparity_norm = mask_norm(disparity,valid)
    return ((pred_disparity_norm - disparity_norm) * valid).abs().sum(dim = (-2,-1)) / valid.sum(dim = (-2,-1))

def normal_sampling(xy_gt,yaw_gt,grid_xy,grid_yaw,sigma_xy,sigma_yaw,lim = 1e-3):
    # grid_xy: [H,W,2]
    # grid_yaw: [B,N,1]

    grid_xy_dis = ((grid_xy[None,None,...] - xy_gt) ** 2).sum(dim = -1)
    grid_yaw_dis = ((grid_yaw[None,:,None,None] % 360 - yaw_gt % 360).abs() % 180) ** 2 # absolute distance

    grid_xy_logit = torch.exp(-(grid_xy_dis) / (2 * sigma_xy ** 2)) # [B,H,W]
    grid_yaw_logit = torch.exp(-(grid_yaw_dis) / (2 * sigma_yaw ** 2) ) # [B,N]

    grid_xy_yaw_logit = grid_xy_logit * grid_yaw_logit # [B,N,H,W]
    grid_xy_yaw_logit = log_softmax(grid_xy_yaw_logit.flatten(-3),-1).reshape(grid_xy_yaw_logit.shape)

    return grid_xy_yaw_logit


class TemplateSampler(torch.nn.Module):
    def __init__(self, grid_xz_bev, ppm, num_rotations, optimize=True):
        super().__init__()

        Δ = 1 / ppm
        h, w = grid_xz_bev.shape[:2] # z_max, x_max * 2
        ksize = max(w, h * 2 + 1)
        radius = ksize * Δ
        grid_xy = make_grid(
            radius,
            radius,
            step_x=Δ,
            step_y=Δ,
            orig_y=(Δ - radius) / 2,
            orig_x=(Δ - radius) / 2,
            y_up=True,
        )

        if optimize:
            assert (num_rotations % 4) == 0
            angles = torch.arange(
                0, 90, 90 / (num_rotations // 4), device=grid_xz_bev.device
            )
        else:
            angles = torch.arange(
                0, 360, 360 / num_rotations, device=grid_xz_bev.device
            )
        rotmats = rotmat2d(angles / 180 * np.pi)
        grid_xy_rot = torch.einsum("...nij,...hwj->...nhwi", rotmats, grid_xy)

        grid_ij_rot = (grid_xy_rot - grid_xz_bev[..., :1, :1, :]) * grid_xy.new_tensor(
            [1, -1] # y axis is down
        )
        grid_ij_rot = grid_ij_rot / Δ
        grid_norm = (grid_ij_rot + 0.5) / grid_ij_rot.new_tensor([w, h]) * 2 - 1

        self.optimize = optimize
        self.num_rots = num_rotations
        self.register_buffer("angles", angles, persistent=False)
        self.register_buffer("grid_norm", grid_norm, persistent=False)

    def forward(self, image_bev):
        grid = self.grid_norm
        b, c = image_bev.shape[:2]
        n, h, w = grid.shape[:3]
        b,c,n,h,w = int(b), int(c), int(n), int(h), int(w)
        grid = grid[None].repeat_interleave(b, 0).reshape(b * n, h, w, 2)
        image = (
            image_bev[:, None]
            .repeat_interleave(n, 1)
            .reshape(b * n, *image_bev.shape[1:])
        )
        kernels = grid_sample(image, grid.to(image.dtype), align_corners=False).reshape(
            b, n, c, h, w
        )

        if self.optimize:  # we have computed only the first quadrant
            kernels_quad234 = [torch.rot90(kernels, -i, (-2, -1)) for i in (1, 2, 3)]
            kernels = torch.cat([kernels] + kernels_quad234, 1)

        return kernels


def conv2d_fft_batchwise(signal, kernel, padding="same", padding_mode="constant"):
    if padding == "same":
        padding = [i // 2 for i in kernel.shape[-2:]]
    padding_signal = [p for p in padding[::-1] for _ in range(2)]
    signal = pad(signal, padding_signal, mode=padding_mode) # pad to keep the original size
    assert signal.size(-1) % 2 == 0

    padding_kernel = [
        pad for i in [1, 2] for pad in [0, signal.size(-i) - kernel.size(-i)]
    ]
    kernel_padded = pad(kernel, padding_kernel) # signal_size+kernel_size

    signal_fr = rfftn(signal, dim=(-1, -2))
    kernel_fr = rfftn(kernel_padded, dim=(-1, -2))

    kernel_fr.imag *= -1  # flip the kernel (convolution)
    output_fr = torch.einsum("bc...,bdc...->bd...", signal_fr, kernel_fr)
    output = irfftn(output_fr, dim=(-1, -2))

    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
        slice(0, (signal.size(i) - kernel.size(i) + 1)) for i in [-2, -1]
    ]
    output = output[crop_slices].contiguous()

    return output


class SparseMapSampler(torch.nn.Module):
    def __init__(self, num_rotations):
        super().__init__()
        angles = torch.arange(0, 360, 360 / self.conf.num_rotations)
        rotmats = rotmat2d(angles / 180 * np.pi)
        self.num_rotations = num_rotations
        self.register_buffer("rotmats", rotmats, persistent=False)

    def forward(self, image_map, p2d_bev):
        h, w = image_map.shape[-2:]
        locations = make_grid(w, h, device=p2d_bev.device)
        p2d_candidates = torch.einsum(
            "kji,...i,->...kj", self.rotmats.to(p2d_bev), p2d_bev
        )
        p2d_candidates = p2d_candidates[..., None, None, :, :] + locations.unsqueeze(-1)
        # ... x N x W x H x K x 2

        p2d_norm = (p2d_candidates / (image_map.new_tensor([w, h]) - 1)) * 2 - 1
        valid = torch.all((p2d_norm >= -1) & (p2d_norm <= 1), -1)
        value = grid_sample(
            image_map, p2d_norm.flatten(-4, -2), align_corners=True, mode="bilinear"
        )
        value = value.reshape(image_map.shape[:2] + valid.shape[-4])
        return valid, value
    
def sample_normal_xyr(volume, xy_grid, angle_grid, nearest_for_inf=False):
    # (B, C, H, W, N) to (B, C, H, W, N+1)
    volume_padded = pad(volume, [0, 1, 0, 0, 0, 0], mode="circular") # from last to the first !

    _,_,H,W,N = volume_padded.shape
    grid_xy = make_grid(W,H,device = volume_padded.device) # [H,W,2]
    grid_yaw = torch.arange(0,N,1,device = volume_padded.device)
    grid_yaw = grid_yaw * 360 / (N - 1)
    
    grid_xy_log_prob = normal_sampling(xy_grid,angle_grid,grid_xy,grid_yaw,sigma_xy=2.0,sigma_yaw=2.0)

    
    # size = xy_grid.new_tensor(volume.shape[-3:-1][::-1])
    # xy_norm = xy_grid / (size - 1)  # align_corners=True 
    # angle_norm = (angle_grid / 360) % 1 # angle_grid = angle in fact
    # grid = torch.concat([angle_norm.unsqueeze(-1), xy_norm], -1)
    # grid_norm = grid * 2 - 1 # [0,1] -> [-1,1]

    # valid = torch.all((grid_norm >= -1)  & (grid_norm <= 1), -1)
    # value = grid_sample(volume_padded, grid_norm, align_corners=True, mode="bilinear")
    value = volume * grid_xy_log_prob + (1 - volume) * (1 - grid_xy_log_prob)
    value = value.mean(dim = (-3,-2,-1))

    return value, value

def sample_xyr(volume, xy_grid, angle_grid, nearest_for_inf=False):
    # (B, C, H, W, N) to (B, C, H, W, N+1)
    volume_padded = pad(volume, [0, 1, 0, 0, 0, 0], mode="circular") # from last to the first !

    size = xy_grid.new_tensor(volume.shape[-3:-1][::-1])
    xy_norm = xy_grid / (size - 1)  # align_corners=True 
    angle_norm = (angle_grid / 360) % 1 # angle_grid = angle in fact
    grid = torch.concat([angle_norm.unsqueeze(-1), xy_norm], -1)
    grid_norm = grid * 2 - 1 # [0,1] -> [-1,1]

    valid = torch.all((grid_norm >= -1)  & (grid_norm <= 1), -1)
    value = grid_sample(volume_padded, grid_norm, align_corners=True, mode="bilinear")

    # if one of the values used for linear interpolation is infinite,
    # we fallback to nearest to avoid propagating inf
    if nearest_for_inf:
        value_nearest = grid_sample(
            volume_padded, grid_norm, align_corners=True, mode="nearest"
        )
        value = torch.where(~torch.isfinite(value) & valid, value_nearest, value)

    return value, valid

def nll_normal_loss_xyr(log_probs, xy, angle):
    log_prob, _ = sample_normal_xyr(
        log_probs.unsqueeze(1), xy[:, None, None, None], angle[:, None, None, None]
    )
    nll = -log_prob.reshape(-1)  # remove C,H,W,N
    return nll

def nll_loss_xyr(log_probs, xy, angle):
    log_prob, _ = sample_xyr(
        log_probs.unsqueeze(1), xy[:, None, None, None], angle[:, None, None, None]
    )
    nll = -log_prob.reshape(-1)  # remove C,H,W,N
    return nll

def res_sem_xyr(f_bev,f_map,xy,angle,valid_bev):
    # (B, C, H, W, N) to (B, C, H, W, N+1)
    B,N,C,H,W = f_bev.shape
    f_bev = f_bev.permute(0,2,1,3,4)
    valid_bev = valid_bev.float().permute(2,0,1,3,4)
    f_bev_padded = pad(f_bev, [0, 0, 0, 0, 0, 1], mode="circular") # from last to the first !
    valid_bev_padded = pad(valid_bev, [0, 0, 0, 0, 0, 1], mode="circular") # from last to the first !

    bev_size = xy.new_tensor(f_bev.shape[-2:][::-1]) # [129,129], width/x --> first
    map_size = xy.new_tensor(f_map.shape[-2:][::-1]) # [256,256], width/x --> first
    radius = max(bev_size) // 2

    
    grid_xy = make_grid(bev_size[0],bev_size[1],step_x = 1,step_y = 1,orig_x = -radius,orig_y = -radius,y_up = False,device = f_bev.device)
    # rotmats = rotmat2d(angle / 180 * np.pi)
    grid_xy_norm = grid_xy / (radius)
    angle = (angle / 360) % 1 
    angle_norm = angle * 2 - 1
    grid_xy_norm = torch.cat([grid_xy_norm.view(1,1,*grid_xy_norm.shape).repeat(B,1,1,1,1),angle_norm.view(-1,1,1,1,1).repeat(1,1,H,W,1)],-1) # [B,N,3]
    grid_xy_map = grid_xy + xy.view(-1,1,1,2)
    grid_xy_map_norm = grid_xy_map / (map_size.view(1,-1) - 1)
    grid_xy_map_norm  = grid_xy_map_norm * 2 - 1
    
    valid_rot_gt = torch.all((grid_xy_norm >= -1) & (grid_xy_norm <= 1), -1)
    valid_map_gt = torch.all((grid_xy_map_norm >= -1) & (grid_xy_map_norm <= 1), -1)
    
    f_bev_gt = grid_sample(f_bev_padded,grid_xy_norm,align_corners=True,mode = 'bilinear').squeeze(2)
    f_map_gt = grid_sample(f_map,grid_xy_map_norm,align_corners=True, mode = 'bilinear')
    # valid_bev_gt = grid_sample(valid_bev_padded,grid_xy_norm,align_corners=True,mode = 'bilinear').squeeze(2)
    # valid_map_gt = grid_sample(valid_map, grid_xy_map_norm,align_corners=True,mode = 'bilinear')
    valid_gt = valid_bev * valid_rot_gt * valid_map_gt
    # f_bev_gt = grid_sample(f_bev,grid_norm,align_corners=True,mode = "bilinear")
    # valid_bev_gt = grid_sample(valid_bev,grid_norm,align_corners=True,mode = "bilinear")
    num_valid = valid_gt.float().sum((-3,-2, -1))
    res_l1 = ((f_map_gt - f_bev_gt.detach()) * valid_gt).abs().sum((-3,-2,-1)) / num_valid
    return res_l1 

"""
def res_l1loss_xyr(f_bev, f_map, xy, angle, valid_bev):
    # f_bev: [B,C,H,W] --> [4,8,129,129]
    # f_map: [B,C,Y,X] --> [4,8,256,256]
    # f_bev_template = f_bev.transpose(1,2) # batch-channel-rotation-x-y
    # valid_template = valid_template.permute(2,1,0,3,4) # batch-channel-rotation-x-y

    # f_bev_padded = pad(f_bev_template, [0, 0, 0, 0, 0, 1], mode="circular") # pad for the latest bin
    # w,h = f_bev.shape[-2:]
    bev_size = xy.new_tensor(f_bev.shape[-2:][::-1]) # [129,64], width/x --> first
    map_size = xy.new_tensor(f_map.shape[-2:][::-1]) # [256,256], width/x --> first
    # f_bev_template: [B,N,C,2*r+1,2*r+1] --> [4,64,8,129,129]
    # f_map: [B,C,H,W] --> [4,8,256,256]

    radius = max(bev_size) // 2
    grid_uv = make_grid(bev_size[0],bev_size[1],step_x = 1,step_y = 1,orig_x = -radius,orig_y = -radius,y_up = False,device = f_bev.device)
    rotmats = rotmat2d(angle / 180 * np.pi)
    grid_uv_rot = torch.einsum("nij, hwj -> nhwi", rotmats, grid_uv)
    # grid_uv_rot = grid_uv_rot # + xy.view(-1,1,1,2) # translation bias
    grid_uv_norm = grid_uv_rot / radius # align_corners = True
    # grid_uv_norm = grid_uv_norm * 2 - 1
    grid_uv_map = grid_uv + xy.view(-1,1,1,2) # biased map sampling coors
    grid_uv_map_norm = grid_uv_map / (map_size.view(1,-1) - 1) # normalize
    grid_uv_map_norm = grid_uv_map_norm * 2 - 1 # [0,1] --> [-1,1]
    valid_map =  torch.all((grid_uv_map_norm >= -1)  & (grid_uv_map_norm <= 1), -1)
    valid_bev_rot = torch.all((grid_uv_norm >= -1) & (grid_uv_norm <= 1), -1)
    # angle_norm = (angle / 360) % 1
    # xy_norm = xy / (size - 1)
    # grid_norm = torch.cat([angle_norm.unsqueeze(-1),xy_norm],-1)
    
    f_bev_gt = grid_sample(f_bev,grid_uv_map_norm,align_corners=True,mode = "bilinear")
    f_map_gt = grid_sample(f_map,grid_uv_norm,align_corners=True,mode = "bilinear")
    # valid_bev_gt = grid_sample(valid_bev,grid_norm,align_corners=True,mode = "bilinear")
    valid_gt = valid_bev * valid_map * valid_bev_rot
    num_valid = valid_gt.float().sum((-2, -1))
    res_l1 = ((f_map_gt.detach() - f_bev_gt) * valid_gt.unsqueeze(1)).abs().sum((-3,-2,-1)) / num_valid
    return res_l1 
"""

def res_l1loss_xyr(f_bev, f_map, xy, angle, valid_bev):
    # f_bev: [B,C,H,W] --> [4,8,129,129]
    # f_map: [B,C,Y,X] --> [4,8,256,256]
    # f_bev_template = f_bev.transpose(1,2) # batch-channel-rotation-x-y
    # valid_template = valid_template.permute(2,1,0,3,4) # batch-channel-rotation-x-y

    # f_bev_padded = pad(f_bev_template, [0, 0, 0, 0, 0, 1], mode="circular") # pad for the latest bin
    # w,h = f_bev.shape[-2:]
    bev_size = xy.new_tensor(f_bev.shape[-2:][::-1]) # [129,64], width/x --> first
    # map_size = xy.new_tensor(f_map.shape[-2:][::-1]) # [256,256], width/x --> first
    # f_bev_template: [B,N,C,2*r+1,2*r+1] --> [4,64,8,129,129]
    # f_map: [B,C,H,W] --> [4,8,256,256]

    f_map_gt,valid_map_gt = crop_map(f_map,xy,angle,bev_size)
    
    valid_gt = valid_bev * valid_map_gt.squeeze(1)
    num_valid = valid_gt.float().sum((-2, -1))

    # f_bev_gt = F.normalize(f_bev_gt,dim = 1)
    # f_map_gt = F.normalize(f_map_gt,dim = 1)

    res_l1 = ((f_map_gt.detach() - f_bev).abs().sum(dim = 1) * valid_gt).sum((-2,-1)) / num_valid / f_map_gt.shape[1]
    return res_l1 

def nll_loss_xyr_smoothed(log_probs, xy, angle, sigma_xy, sigma_r, mask=None):
    *_, nx, ny, nr = log_probs.shape
    grid_x = torch.arange(nx, device=log_probs.device, dtype=torch.float)
    dx = (grid_x - xy[..., None, 0]) / sigma_xy
    grid_y = torch.arange(ny, device=log_probs.device, dtype=torch.float)
    dy = (grid_y - xy[..., None, 1]) / sigma_xy
    dr = (
        torch.arange(0, 360, 360 / nr, device=log_probs.device, dtype=torch.float)
        - angle[..., None]
    ) % 360
    dr = torch.minimum(dr, 360 - dr) / sigma_r
    diff = (
        dx[..., None, :, None] ** 2
        + dy[..., :, None, None] ** 2
        + dr[..., None, None, :] ** 2
    )
    pdf = torch.exp(-diff / 2)
    if mask is not None:
        pdf.masked_fill_(~mask[..., None], 0)
        log_probs = log_probs.masked_fill(~mask[..., None], 0)
    pdf /= pdf.sum((-1, -2, -3), keepdim=True)
    return -torch.sum(pdf * log_probs.to(torch.float), dim=(-1, -2, -3))


def log_softmax_spatial(x, dims=3):
    return log_softmax(x.flatten(-dims), dim=-1).reshape(x.shape)


@torch.jit.script
def argmax_xy(scores: torch.Tensor) -> torch.Tensor:
    indices = scores.flatten(-2).max(-1).indices
    width = scores.shape[-1]
    x = indices % width
    y = torch.div(indices, width, rounding_mode="floor")
    return torch.stack((x, y), -1)


@torch.jit.script
def expectation_xy(prob: torch.Tensor) -> torch.Tensor:
    h, w = prob.shape[-2:]
    grid = make_grid(float(w), float(h), device=prob.device).to(prob)
    return torch.einsum("...hw,hwd->...d", prob, grid)


@torch.jit.script
def expectation_xyr(
    prob: torch.Tensor, covariance: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    h, w, num_rotations = prob.shape[-3:]
    x, y = torch.meshgrid(
        [
            torch.arange(w, device=prob.device, dtype=prob.dtype),
            torch.arange(h, device=prob.device, dtype=prob.dtype),
        ],
        indexing="xy",
    )
    grid_xy = torch.stack((x, y), -1)
    xy_mean = torch.einsum("...hwn,hwd->...d", prob, grid_xy)

    angles = torch.arange(0, 1, 1 / num_rotations, device=prob.device, dtype=prob.dtype)
    angles = angles * 2 * np.pi
    grid_cs = torch.stack([torch.cos(angles), torch.sin(angles)], -1)
    cs_mean = torch.einsum("...hwn,nd->...d", prob, grid_cs)
    angle = torch.atan2(cs_mean[..., 1], cs_mean[..., 0])
    angle = (angle * 180 / np.pi) % 360

    if covariance:
        xy_cov = torch.einsum("...hwn,...hwd,...hwk->...dk", prob, grid_xy, grid_xy)
        xy_cov = xy_cov - torch.einsum("...d,...k->...dk", xy_mean, xy_mean)
    else:
        xy_cov = None

    xyr_mean = torch.cat((xy_mean, angle.unsqueeze(-1)), -1)
    return xyr_mean, xy_cov


@torch.jit.script
def argmax_xyr(scores: torch.Tensor) -> torch.Tensor:
    indices = scores.flatten(-3).max(-1).indices
    width, num_rotations = scores.shape[-2:]
    wr = width * num_rotations
    y = torch.div(indices, wr, rounding_mode="floor")
    x = torch.div(indices % wr, num_rotations, rounding_mode="floor")
    angle_index = indices % num_rotations
    angle = angle_index * 360 / num_rotations
    xyr = torch.stack((x, y, angle), -1)
    return xyr

# @torch.jit.script
def topk_xyr(scores: torch.Tensor, k: int = 5) -> torch.Tensor:
    indices = scores.flatten(-3).topk(k,-1).indices
    width, num_rotations = scores.shape[-2:]
    wr = width * num_rotations
    y = torch.div(indices, wr, rounding_mode="floor")
    x = torch.div(indices % wr, num_rotations, rounding_mode="floor")
    angle_index = indices % num_rotations
    angle = angle_index * 360 / num_rotations
    xyr = torch.stack((x, y, angle), -1)
    return xyr



@torch.jit.script
def mask_yaw_prior(
    scores: torch.Tensor, yaw_prior: torch.Tensor, num_rotations: int
) -> torch.Tensor:
    step = 360 / num_rotations
    step_2 = step / 2
    angles = torch.arange(step_2, 360 + step_2, step, device=scores.device)
    yaw_init, yaw_range = yaw_prior.chunk(2, dim=-1)
    rot_mask = angle_error(angles, yaw_init) < yaw_range
    return scores.masked_fill_(~rot_mask[:, None, None], -np.inf)


def fuse_gps(log_prob, uv_gps, ppm, sigma=10, gaussian=False):
    grid = make_grid(*log_prob.shape[-3:-1][::-1]).to(log_prob)
    dist = torch.sum((grid - uv_gps) ** 2, -1)
    sigma_pixel = sigma * ppm
    if gaussian:
        gps_log_prob = -1 / 2 * dist / sigma_pixel**2
    else:
        gps_log_prob = torch.where(dist < sigma_pixel**2, 1, -np.inf)
    log_prob_fused = log_softmax_spatial(log_prob + gps_log_prob.unsqueeze(-1))
    return log_prob_fused
