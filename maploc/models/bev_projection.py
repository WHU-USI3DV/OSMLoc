# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch.nn.functional import grid_sample,interpolate
import torch.nn as nn
from ..utils.geometry import from_homogeneous, to_homogeneous, in_matrix, in_matrix_inv
from .utils import make_grid



class PolarProjectionDepth(torch.nn.Module):
    def __init__(self, z_max, ppm, scale_range, z_min=None):
        super().__init__()
        self.z_max = z_max
        self.Δ = Δ = 1 / ppm
        self.z_min = z_min = Δ if z_min is None else z_min
        self.scale_range = scale_range
        z_steps = torch.arange(z_min, z_max + Δ, Δ)
        self.register_buffer("depth_steps", z_steps, persistent=False)
        scale_steps = torch.arange(z_min, z_max + Δ, Δ)
        self.register_buffer("scale_steps", scale_steps, persistent=False)

    def sample_depth_scores(self, pixel_scales, camera):
        scale_steps = camera.f[..., None, 1] / self.depth_steps.flip(-1) # y_up = False
        log_scale_steps = torch.log2(scale_steps)
        scale_min, scale_max = self.scale_range
        log_scale_norm = (log_scale_steps - scale_min) / (scale_max - scale_min)
        log_scale_norm = log_scale_norm * 2 - 1  # in [-1, 1]

        values = pixel_scales.flatten(1, 2).unsqueeze(-1) # [B,H*W,C,1]
        indices = log_scale_norm.unsqueeze(-1)
        indices = torch.stack([torch.zeros_like(indices), indices], -1)
        depth_scores = grid_sample(values, indices, align_corners=True) # resample in N log scales position.
        depth_scores = depth_scores.reshape(
            pixel_scales.shape[:-1] + (len(self.depth_steps),)
        )
        return depth_scores

    def forward(
        self,
        image,
        pixel_scales,
        camera,
        return_total_score=False,
    ):
        depth_scores = self.sample_depth_scores(pixel_scales, camera)
        depth_prob = torch.softmax(depth_scores, dim=1)
        image_polar = torch.einsum("...dhw,...hwz->...dzw", image, depth_prob)
        if return_total_score:
            cell_score = torch.logsumexp(depth_scores, dim=-1, keepdim=True)
            return image_polar, cell_score.squeeze(1)
        return image_polar, depth_prob
    
    def sample_depth_scores_wrapper(self, pixel_scales,f):
        scale_steps = f[..., None, 1] / self.scale_steps.flip(-1)
        log_scale_steps = torch.log2(scale_steps)
        scale_min, scale_max = self.scale_range
        log_scale_norm = (log_scale_steps - scale_min) / (scale_max - scale_min)
        log_scale_norm = log_scale_norm * 2 - 1  # in [-1, 1]

        values = pixel_scales.flatten(1, 2).unsqueeze(-1) # [B,H*W,C,1]
        indices = log_scale_norm.unsqueeze(-1)
        indices = torch.stack([torch.zeros_like(indices), indices], -1)
        depth_scores = grid_sample(values, indices, align_corners=True) # resample in N log scales position.
        depth_scores = depth_scores.reshape(
            pixel_scales.shape[:-1] + (len(self.scale_steps),)
        )
        return depth_scores
    
    def forward_wrapper(
        self,
        image,
        valid_image,
        pixel_scales,
        # depth_logit,
        f
    ):
        """
        image: [B,C,H,W], 
        depth_logit: [B,H,W,D], D means depth planes
        """
        depth_scores = self.sample_depth_scores_wrapper(pixel_scales, f)
        depth_prob = depth_scores.softmax(dim = -1) # [B,H,W,D]
        image_polar = torch.einsum("...dhw,...hwz->...dzw", image, depth_prob)
        depth_abs = (depth_prob * self.scale_steps.reshape(1,1,1,-1).flip(-1).to(depth_prob)).sum(dim = -1).unsqueeze(1) # [B,1,H,W]
        # image_depth = (scale_probs * self.scale_steps.reshape(1,-1,1,1).to(rel_depth) * rel_depth).sum(dim = 1,keepdim = True)

        return image_polar, depth_abs

class ProjectionScalePolar(torch.nn.Module):
    def __init__(self, z_max, ppm, scale_range, z_min=None):
        super().__init__()
        self.z_max = z_max
        self.Δ = Δ = 1 / ppm
        self.scale_range = scale_range
        self.z_min = z_min = Δ if z_min is None else z_min
        scale_steps = torch.arange(z_min, z_max + Δ, Δ)
        self.register_buffer("scale_steps", scale_steps, persistent=False)

    def sample_depth_scores(self, pixel_scales,camera):
        scale_steps = camera.f[..., None, 1] / self.scale_steps.flip(-1)
        log_scale_steps = torch.log2(scale_steps)
        scale_min, scale_max = self.scale_range
        log_scale_norm = (log_scale_steps - scale_min) / (scale_max - scale_min)
        log_scale_norm = log_scale_norm * 2 - 1  # in [-1, 1]

        values = pixel_scales.flatten(1, 2).unsqueeze(-1) # [B,H*W,C,1]
        indices = log_scale_norm.unsqueeze(-1)
        indices = torch.stack([torch.zeros_like(indices), indices], -1)
        depth_scores = grid_sample(values, indices, align_corners=True) # resample in N log scales position.
        depth_scores = depth_scores.reshape(
            pixel_scales.shape[:-1] + (len(self.scale_steps),)
        )
        return depth_scores

    def sample_depth_scores_wrapper(self, pixel_scales,f):
        scale_steps = f[..., None, 1] / self.scale_steps.flip(-1)
        log_scale_steps = torch.log2(scale_steps)
        scale_min, scale_max = self.scale_range
        log_scale_norm = (log_scale_steps - scale_min) / (scale_max - scale_min)
        log_scale_norm = log_scale_norm * 2 - 1  # in [-1, 1]

        values = pixel_scales.flatten(1, 2).unsqueeze(-1) # [B,H*W,C,1]
        indices = log_scale_norm.unsqueeze(-1)
        indices = torch.stack([torch.zeros_like(indices), indices], -1)
        depth_scores = grid_sample(values, indices, align_corners=True) # resample in N log scales position.
        depth_scores = depth_scores.reshape(
            pixel_scales.shape[:-1] + (len(self.scale_steps),)
        )
        return depth_scores

    def forward(
        self,
        image,
        valid_image,
        pixel_scales,
        # depth_logit,
        camera,
    ):
        """
        image: [B,C,H,W], 
        depth_logit: [B,H,W,D], D means depth planes
        """
        depth_scores = self.sample_depth_scores(pixel_scales, camera)
        depth_prob = depth_scores.softmax(dim = -1) # [B,H,W,D]
        image_polar = torch.einsum("...dhw,...hwz->...dzw", image, depth_prob)
        depth_abs = (depth_prob * self.scale_steps.reshape(1,1,1,-1).flip(-1).to(depth_prob)).sum(dim = -1).unsqueeze(1) # [B,1,H,W]
        # image_depth = (scale_probs * self.scale_steps.reshape(1,-1,1,1).to(rel_depth) * rel_depth).sum(dim = 1,keepdim = True)

        return image_polar, depth_abs

    def forward_wrapper(
        self,
        image,
        valid_image,
        pixel_scales,
        # depth_logit,
        f
    ):
        """
        image: [B,C,H,W], 
        depth_logit: [B,H,W,D], D means depth planes
        """
        depth_scores = self.sample_depth_scores_wrapper(pixel_scales, f)
        depth_prob = depth_scores.softmax(dim = -1) # [B,H,W,D]
        image_polar = torch.einsum("...dhw,...hwz->...dzw", image, depth_prob)
        depth_abs = (depth_prob * self.scale_steps.reshape(1,1,1,-1).flip(-1).to(depth_prob)).sum(dim = -1).unsqueeze(1) # [B,1,H,W]
        # image_depth = (scale_probs * self.scale_steps.reshape(1,-1,1,1).to(rel_depth) * rel_depth).sum(dim = 1,keepdim = True)

        return image_polar, depth_abs

class ProjectionScale(torch.nn.Module):
    def __init__(self, z_max, y_max, x_max, ppm, scale_range, image_size, z_min=None,):
        super().__init__()
        self.z_max = z_max
        self.x_max = x_max
        self.y_max = y_max
        self.Δ = Δ = 1 / ppm
        self.scale_range = scale_range
        self.z_min = z_min = Δ if z_min is None else z_min
        self.bev_size = [int((x_max * 2 + Δ) // Δ), int(z_max // Δ)]
        scale_steps = torch.arange(z_min, z_max + Δ, Δ)
        grid_uv = make_grid(
            *image_size, step_y=1, step_x=1, orig_y=0, orig_x=0, y_up=False
        ).reshape(1,*image_size,2)
        grid_xz = make_grid(
             x_max * 2 + Δ, z_max, step_y=Δ, step_x=Δ, orig_y=Δ, orig_x=-x_max, y_up=True
        )
        dx = torch.tensor([Δ,20,Δ]).reshape(3,)
        self.register_buffer("dx",dx, persistent=False)
        self.register_buffer("grid_xz", grid_xz, persistent=False)
        self.register_buffer("grid_uv", grid_uv, persistent=False)
        self.register_buffer("scale_steps", scale_steps, persistent=False)

    def sample_scale_scores(self, pixel_scales):
        log_scale_steps = torch.log2(self.scale_steps)
        scale_min, scale_max = self.scale_range
        log_scale_norm = (log_scale_steps - scale_min) / (scale_max - scale_min)
        log_scale_norm = log_scale_norm * 2 - 1  # in [-1, 1]

        values = pixel_scales.flatten(1, 2).unsqueeze(-1) # [B,H*W,C,1]
        indices = log_scale_norm[None,:,None].repeat(pixel_scales.shape[0],1,1) # [B,N]
        indices = torch.stack([torch.zeros_like(indices), indices], -1) # [B,N,2]
        scale_scores = grid_sample(values, indices, align_corners=True) # resample in N log scales position.
        scale_scores = scale_scores.reshape(
            pixel_scales.shape[:-1] + (len(self.scale_steps),)
        )
        return scale_scores
    
    def geo_projection(
            self,
            depth,
            intrinsic_inv,
            valid_image
        ):
        """ project image points to frustum
        depth: estimated absolute depth, [B,D,H,W]
        intrinsic_inv: inverse intrinsic matrix of camera, [B,3,3]
        valid_image: valid map of image, [B,1,H,W]
        return: points in bev view, [B,D,H,W,3]
        """
        B,D,H,W = depth.shape
        grid_uvz = to_homogeneous(self.grid_uv).repeat(B,1,1,1) # [B,H,W,3]
        grid_uvz = grid_uvz[:,None,...] * depth[...,None] # [B,1,H,W,3] * [B,D,H,W,1] --> [B,D,H,W,3]
        grid_xyz = torch.einsum("...ij, ...dhwj -> ...dhwi", intrinsic_inv, grid_uvz) # [B,3,3] @ [B,D,H,W,3] --> [B,D,H,W,3]
        valid_xyz = valid_image.repeat(1,D,1,1) # [B,D,H,W]
        return grid_xyz,valid_xyz # y_up = False --> y_up = True


    def forward(
        self,
        image,
        valid_image,
        # pixel_scales,
        depth_logit,
        camera,
    ):
        """
        depth_logit: [B,D,H,W], D means depth planes
        """
        B,C,H,W = image.shape
        valid_image = interpolate(valid_image.float(),size = image.shape[2:],mode = 'bilinear')
        depth_prob = depth_logit.softmax(dim = 1) # [B,D,H,W]
        # depth = rel_depth * self.scale_steps.reshape(1,-1,1,1).to(rel_depth)  # [B,1,H,W] * [1,D,1,1] --> [B,D,H,W]
        abs_depth = (depth_prob * self.scale_steps.reshape(1,-1,1,1).to(depth_prob)).unsqueeze(1) # [B,1,D,H,W]
        image = depth_prob.unsqueeze(1) * image.unsqueeze(2) # [B,1,D,H,W] * [B,C,1,H,W] --> [B,C,D,H,W]
        K_inv = torch.inverse(in_matrix(camera)).to(image)
        grid_xyz,valid_xyz = self.geo_projection(self.scale_steps.reshape(1,-1,1,1).repeat(B,1,H,W).to(image),K_inv,valid_image) # [B,D,H,W,3]       
        image_bev,valid_bev = self.bev_pool(grid_xyz,image,valid_xyz) # [B,C,D,L]
        depth_bev,valid_bev = self.bev_pool(grid_xyz,abs_depth,valid_xyz)
        abs_depth = abs_depth.sum(dim = (1,2), keepdims = True).squeeze(1)
        # image_depth = (scale_probs * self.scale_steps.reshape(1,-1,1,1).to(rel_depth) * rel_depth).sum(dim = 1,keepdim = True)

        return image_bev, valid_bev, abs_depth, depth_bev
    
    

        
        

class CartesianProjection(torch.nn.Module):
    def __init__(self, z_max, x_max, ppm, z_min=None):
        super().__init__()
        self.z_max = z_max
        self.x_max = x_max
        self.Δ = Δ = 1 / ppm
        self.z_min = z_min = Δ if z_min is None else z_min

        grid_xz = make_grid(
            x_max * 2 + Δ, z_max, step_y=Δ, step_x=Δ, orig_y=Δ, orig_x=-x_max, y_up=True
        )
        self.register_buffer("grid_xz", grid_xz, persistent=False)

    def grid_to_polar(self, cam):
        f, c = cam.f[..., 0][..., None, None], cam.c[..., 0][..., None, None]
        u = from_homogeneous(self.grid_xz).squeeze(-1) * f + c # f_x * (X/Z) + c_x, map coordinates in world system to image coordinate system
        z_idx = (self.grid_xz[..., 1] - self.z_min) / self.Δ  # convert z value to index (image coordinate), z axis is up
        z_idx = z_idx[None].expand_as(u)
        grid_polar = torch.stack([u, z_idx], -1)
        return grid_polar
    
    def grid_to_polar_wrapper(self, c,f):
        f, c = f[..., 0][..., None, None], c[..., 0][..., None, None]
        u = from_homogeneous(self.grid_xz).squeeze(-1) * f + c # f_x * (X/Z) + c_x, map coordinates in world system to image coordinate system
        z_idx = (self.grid_xz[..., 1] - self.z_min) / self.Δ  # convert z value to index (image coordinate), z axis is up
        z_idx = z_idx[None].expand_as(u)
        grid_polar = torch.stack([u, z_idx], -1)
        return grid_polar

    def sample_from_polar(self, image_polar, valid_polar, grid_uz):
        size = grid_uz.new_tensor(image_polar.shape[-2:][::-1]) # bins of depths and rays ?
        grid_uz_norm = (grid_uz + 0.5) / size * 2 - 1 # +0.5: align_corners = False; * 2 - 1 : map [0,1] --> [-1,1]
        grid_uz_norm = grid_uz_norm * grid_uz.new_tensor([1, -1])  # y axis is up (change to down)
        image_bev = grid_sample(image_polar, grid_uz_norm, align_corners=False)

        if valid_polar is None:
            valid = torch.ones_like(image_polar[..., :1, :, :])
        else:
            valid = valid_polar.to(image_polar)[:, None]
        valid = grid_sample(valid, grid_uz_norm, align_corners=False)
        valid = valid.squeeze(1) > (1 - 1e-4)

        return image_bev, valid

    def forward(self, image_polar, valid_polar, cam):
        grid_uz = self.grid_to_polar(cam)
        image, valid = self.sample_from_polar(image_polar, valid_polar, grid_uz)
        return image, valid, grid_uz
    
    def forward_wrapper(self, image_polar, valid_polar, c,f):
        grid_uz = self.grid_to_polar_wrapper(c,f)
        image, valid = self.sample_from_polar(image_polar, valid_polar, grid_uz)
        return image, valid, grid_uz
