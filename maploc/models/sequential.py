# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
import torch.nn.functional as F
from random import random

from .voting import argmax_xyr, log_softmax_spatial, sample_xyr
from .utils import deg2rad,rad2deg, make_grid, rotmat2d


def log_gaussian(points, mean, sigma):
    return -1 / 2 * torch.sum((points - mean) ** 2, -1) / sigma**2


def log_laplace(points, mean, sigma):
    return -torch.sum(torch.abs(points - mean), -1) / sigma


def propagate_belief(
    Δ_xy, Δ_yaw, canvas_target, canvas_source, belief, num_rotations=None
):
    # We allow a different sampling resolution in the target frame
    if num_rotations is None:
        num_rotations = belief.shape[-1]

    angles = torch.arange(
        0, 360, 360 / num_rotations, device=Δ_xy.device, dtype=Δ_xy.dtype
    )
    uv_grid = make_grid(canvas_target.w, canvas_target.h, device=Δ_xy.device)
    xy_grid = canvas_target.to_xy(uv_grid.to(Δ_xy))

    Δ_xy_world = torch.einsum("nij,j->ni", rotmat2d(deg2rad(-angles)), Δ_xy) # trans_xy in world(canvas_target) system?
    xy_grid_prev = xy_grid[..., None, :] + Δ_xy_world[..., None, None, :, :]
    uv_grid_prev = canvas_source.to_uv(xy_grid_prev).to(Δ_xy)

    angles_prev = angles + Δ_yaw
    angles_grid_prev = angles_prev.tile((canvas_target.h, canvas_target.w, 1))

    prior, valid = sample_xyr(
        belief[None, None], # [1,1,H,W]
        uv_grid_prev.to(belief)[None], # [1,H,W,2]
        angles_grid_prev.to(belief)[None], # [1,H,W,1]
        nearest_for_inf=True,
    ) # update observation model(prob map) from frame_(i-1) to frame_i
    return prior, valid


def markov_filtering(observations, canvas, xys, yaws, idxs=None):
    assert len(observations) == len(canvas) == len(xys) == len(yaws)
    if idxs is None:
        idxs = range(len(observations))
    belief = None
    beliefs = []
    for i in idxs:
        obs = observations[i]
        if belief is None:
            belief = obs
        else:
            Δ_xy = rotmat2d(deg2rad(yaws[i])) @ (xys[i - 1] - xys[i])
            Δ_yaw = yaws[i - 1] - yaws[i]
            prior, valid = propagate_belief(
                Δ_xy, Δ_yaw, canvas[i], canvas[i - 1], belief
            ) # prediction step?  from time step i-1 to time step i
            prior = prior[0, 0].masked_fill_(~valid[0], -np.inf)
            belief = prior + obs
            belief = log_softmax_spatial(belief)
        beliefs.append(belief)
    uvt_seq = torch.stack([argmax_xyr(p) for p in beliefs])
    return beliefs, uvt_seq

def grid_sample_jac(image, optical, jac=None):
    # values in optical within range of [0, H], and [0, W]
    # image: [B,C,H,W,N]
    # optical: [B,H,W,3]
    B, C, IH, IW,IN = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0].view(B, 1, H, W)
    iy = optical[..., 1].view(B, 1, H, W)
    ir = optical[..., 2].view(B, 1, H, W) 

    with torch.no_grad():
        ix_nw = torch.floor(ix)  # north-west  upper-left-x [ix]
        iy_nw = torch.floor(iy)  # north-west  upper-left-y [iy]
        ix_ne = ix_nw + 1        # north-east  upper-right-x [ix] + 1, move --> in x direction
        iy_ne = iy_nw            # north-east  upper-right-y [iy] + 1
        ix_sw = ix_nw            # south-west  lower-left-x ix
        iy_sw = iy_nw + 1        # south-west  lower-left-y  [iy] + 1, move --> in y direction
        ix_se = ix_nw + 1        # south-east  lower-right-x
        iy_se = iy_nw + 1        # south-east  lower-right-y
        ir_be = torch.floor(ir)
        ir_fr = ir_be + 1

        torch.clamp(ix_nw, 0, IW -1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH -1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW -1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH -1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW -1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH -1, out=iy_sw)

        torch.clamp(ix_se, 0, IW -1, out=ix_se)
        torch.clamp(iy_se, 0, IH -1, out=iy_se)

        torch.clamp(ir_be,0, IN -1, out = ir_be)
        torch.clamp(ir_fr,0, IN -1, out = ir_fr)

    mask_x = (ix >= 0) & (ix <= IW - 1)
    mask_y = (iy >= 0) & (iy <= IH - 1)
    mask_r = (ir >= 0) & (ir <= IN - 1)
    mask = mask_x * mask_y * mask_r # valid mask

    assert torch.sum(mask) > 0

    nw_be = (ix_se - ix) * (iy_se - iy) * (ir_fr - ir) * mask # partition for bilinear grid sampling
    nw_fr = (ix_se - ix) * (iy_se - iy) * (ir - ir_be) * mask
    ne_be = (ix - ix_sw) * (iy_sw - iy) * (ir_fr - ir) * mask
    ne_fr = (ix - ix_sw) * (iy_sw - iy) * (ir - ir_be) *mask
    sw_be = (ix_ne - ix) * (iy - iy_ne) * (ir_fr - ir) *mask
    sw_fr = (ix_ne - ix) * (iy - iy_ne) * (ir - ir_be) *mask
    se_be = (ix - ix_nw) * (iy - iy_nw) * (ir_fr - ir) *mask
    se_fr = (ix - ix_nw) * (iy - iy_nw) * (ir - ir_be) *mask

    image = image.view(B, C, IH * IW * IN)

    nw_be_val = torch.gather(image, 2, (iy_nw * IW * IN + ix_nw * IN + ir_be).long().view(B, 1, H * W).repeat(1, C, 1)).view(B, C, H, W)
    nw_fr_val = torch.gather(image, 2, (iy_nw * IW * IN + ix_nw * IN + ir_fr).long().view(B, 1, H * W).repeat(1, C, 1)).view(B, C, H, W)
    ne_be_val = torch.gather(image, 2, (iy_ne * IW * IN + ix_ne * IN + ir_be).long().view(B, 1, H * W).repeat(1, C, 1)).view(B, C, H, W)
    ne_fr_val = torch.gather(image, 2, (iy_ne * IW * IN + ix_ne * IN + ir_fr).long().view(B, 1, H * W).repeat(1, C, 1)).view(B, C, H, W)
    sw_be_val = torch.gather(image, 2, (iy_sw * IW * IN + ix_sw * IN + ir_be).long().view(B, 1, H * W).repeat(1, C, 1)).view(B, C, H, W)
    sw_fr_val = torch.gather(image, 2, (iy_sw * IW * IN + ix_sw * IN + ir_fr).long().view(B, 1, H * W).repeat(1, C, 1)).view(B, C, H, W)
    se_be_val = torch.gather(image, 2, (iy_se * IW * IN + ix_se * IN + ir_be).long().view(B, 1, H * W).repeat(1, C, 1)).view(B, C, H, W)
    se_fr_val = torch.gather(image, 2, (iy_se * IW * IN + ix_se * IN + ir_fr).long().view(B, 1, H * W).repeat(1, C, 1)).view(B, C, H, W)

    out_val = nw_be_val * nw_be + nw_fr_val * nw_fr \
            + ne_be_val * ne_be + ne_fr_val * ne_fr \
            + sw_be_val * sw_be + sw_fr_val * sw_fr \
            + se_be_val * se_be + se_fr_val * se_fr

    if jac is not None:

        dout_dpx =    nw_be_val * (iy_se - iy) * (ir_be - ir) * mask + nw_fr_val * (iy_se - iy) * (ir_fr - ir) * mask \
                    + ne_be_val * (iy_sw - iy) * (ir_be - ir) * mask + ne_fr_val * (iy_sw - iy) * (ir - ir_be) * mask \
                    + sw_be_val * (iy - iy_ne) * (ir_be - ir) * mask + sw_fr_val * (iy - iy_ne) * (ir_fr - ir) * mask \
                    + se_be_val * (iy - iy_nw) * (ir_be - ir) * mask + se_fr_val * (iy - iy_nw) * (ir - ir_be) * mask
        
        dout_dpy =    nw_be_val * (ix_se - ix) * (ir_be - ir) * mask   + nw_be_val * (ix_se - ix) * (ir_fr - ir) * mask \
                    + ne_be_val * (ix - ix_sw) * (ir_be - ir) * mask + ne_fr_val * (ix - ix_sw) * (ir - ir_be) * mask   \
                    + sw_be_val * (ix_ne - ix) * (ir_be - ir) * mask + sw_fr_val * (ix_ne - ix) * (ir_fr - ir) * mask   \
                    + se_be_val * (ix - ix_nw) * (ir_be - ir) * mask + se_fr_val * (ix - ix_nw) * (ir - ir_be) * mask
        
        dout_dpr =    nw_be_val * (ix_se - ix) * (iy_se - iy) * mask + nw_fr_val * (ix_se - ix) * (iy - iy_ne) * mask \
                    + ne_be_val * (ix - ix_sw) * (iy_se - iy) * mask + ne_fr_val * (ix - ix_sw) * (iy - iy_ne) * mask   \
                    + sw_be_val * (ix_ne - ix) * (iy_se - iy) * mask + sw_fr_val * (ix_ne - ix) * (iy - iy_ne) * mask  \
                    + se_be_val * (ix - ix_nw) * (iy_se - iy) * mask + se_fr_val * (ix - ix_nw) * (iy - iy_ne) * mask
        
        dout_dpxyr = torch.stack([dout_dpx, dout_dpy,dout_dpr], dim=-1)  # [N, C, H, W, 3]

        # assert jac.shape[1:] == [N, H, W, 2]
        jac_new = dout_dpxyr[None, :, :, :, :, :] * jac[:, :, None, :, :, :] # [1,N,C,H,W,3] * [3,N,1,H,W,3] -> [3,N,C,H,W,3]
        jac_new1 = torch.sum(jac_new, dim=-1) # [3,N,C,H,W]

        if torch.any(torch.isnan(jac)) or torch.any(torch.isnan(dout_dpxyr)):
            print('Nan occurs')

        return out_val, jac_new1 #jac_new1 #jac_new.permute(4, 0, 1, 2, 3)
    else:
        return out_val, None


def integrate_observation(
    source, # source_belief map
    target, # target_belief map
    xy_source,
    xy_target, # xy_gt
    yaw_source,
    yaw_target, # yaw_gt
    canvas_source,
    canvas_target,
    **kwargs
):
    Δ_xy = rotmat2d(deg2rad(yaw_target)) @ (xy_source - xy_target) # motion model (xy?)
    Δ_yaw = yaw_source - yaw_target # motion model (yaw?)
    prior, valid = propagate_belief(
        Δ_xy, Δ_yaw, canvas_target, canvas_source, source, **kwargs
    ) # prediction step?  from time step i-1 to time step i
    prior = prior[0, 0].masked_fill_(~valid[0], -np.inf) # mask out invalid reigon
    target.add_(prior) # prior_map + target_map ?
    target.sub_(target.max())  # normalize to avoid overflow (substrat in-place)
    return prior

class RigidAligner:
    def __init__(
        self,
        canvas_ref=None,
        xy_ref=None,
        yaw_ref=None,
        num_rotations=None,
        track_priors=False,
    ):
        self.canvas = canvas_ref
        self.xy_ref = xy_ref
        self.yaw_ref = yaw_ref
        self.rotmat_ref = None
        self.num_rotations = num_rotations
        self.belief = None
        self.priors = [] if track_priors else None

        self.yaw_slam2geo = None
        self.Rt_slam2geo = None

    def update(self, observation, canvas, xy, yaw):
        # initialization
        if self.canvas is None:
            self.canvas = canvas
        if self.xy_ref is None:
            self.xy_ref = xy # xy_gt (prior!)
            self.yaw_ref = yaw # yaw_gt (prior!)
            self.rotmat_ref = rotmat2d(deg2rad(self.yaw_ref)) # rotmat_gt (prior!)
        if self.num_rotations is None:
            self.num_rotations = observation.shape[-1]
        if self.belief is None:
            self.belief = observation.new_zeros(
                (self.canvas.h, self.canvas.w, self.num_rotations)
            ) # belief map of the canvas ?

        prior = integrate_observation(
            observation,
            self.belief, # update prior belif map ?
            xy,
            self.xy_ref, # xy_gt ?
            yaw,
            self.yaw_ref, # yaw_gt ?
            canvas, # new canvas map
            self.canvas, # prior canvas map
            num_rotations=self.num_rotations,
        ) # return prediction pose

        if self.priors is not None:
            self.priors.append(prior.cpu()) # update i-frame's probability map
        return prior

    def update_with_ref(self, observation, canvas, xy, yaw):
        if self.belief is not None:
            observation = observation.clone()
            integrate_observation(
                self.belief,
                observation,
                self.xy_ref,
                xy,
                self.yaw_ref,
                yaw,
                self.canvas,
                canvas,
                num_rotations=observation.shape[-1],
            )

        self.belief = observation
        self.canvas = canvas
        self.xy_ref = xy
        self.yaw_ref = yaw

    def compute(self):
        uvt_align_ref = argmax_xyr(self.belief) # select max probability pose on the belief map
        self.yaw_ref_align = uvt_align_ref[-1]  # yaw_ref
        self.xy_ref_align = self.canvas.to_xy(uvt_align_ref[:2].double()) # uvt_ref to xy_ref canvas map coordinate

        self.yaw_slam2geo = self.yaw_ref - self.yaw_ref_align # yaw residual of current frame and prev frame?
        R_slam2geo = rotmat2d(deg2rad(self.yaw_slam2geo)) # yaw (aligned) - yaw (init) ?
        t_slam2geo = self.xy_ref_align - R_slam2geo @ self.xy_ref # translation (aligned) - translation (init) ?
        self.Rt_slam2geo = (R_slam2geo, t_slam2geo)

    def transform(self, xy, yaw):
        if self.Rt_slam2geo is None or self.yaw_slam2geo is None:
            raise ValueError("Missing transformation, call `compute()` first!")
        xy_geo = self.Rt_slam2geo[1].to(xy) + xy @ self.Rt_slam2geo[0].T.to(xy) # t_slam2geo + xy @ R_slam2geo
        return xy_geo, (self.yaw_slam2geo.to(yaw) + yaw) % 360
    
class ParticleAligner(RigidAligner):
    def __init__(self,num_Particles, canvas_ref=None, xy_ref=None, yaw_ref=None, num_rotations=None, track_priors=False):
        super().__init__(canvas_ref, xy_ref, yaw_ref, num_rotations, track_priors)
        self.num_Particles = num_Particles
        self.is_converged = False
        self.particles = None
        # self.canvs_ref = canvas_prev
        # self.xy_ref = xy_prev
        # self.yaw_ref = yaw_pref
        
    def init_particles_given_coords(self,coords, init_weight=1.0):
        """ Initialize particles uniformly given the candidate coordinates.
        Args:
        # numParticles: number of particles.
        coords: candidate coordinates (high-prob poses), [N,3]->[x,y,yaw]
        init_weight: initialization weight.
        Return:
        particles. [N,4]->[x,y,yaw,weight]
        """
        particles = []
        # rand = torch.rand
        # args_coords = torch.arange(len(coords))
        selected_args = torch.randperm((coords.shape[0]))[:self.num_Particles] # [N,]
        coords = coords[selected_args,:]
        weight = torch.ones_like(coords[:,:1])
        particles = torch.cat([coords,weight],dim = 1) # [N,4]
  
        return particles # [N,4]
    
    def motion_model(self,xy,yaw,thrs = 0.1):
        """ MOTION performs the sampling from the proposal.
        distribution, here the rotation-translation-rotation motion model

        input:
            particles: the particles as in the main script
            xy: xy coordinates at current frame
            yaw: yaw at current frame

        output:
            the same particles, with updated poses.

        The position of the i-th particle is given by the 3D vector
        particles(i).pose which represents (x, y, theta).

        Assume Gaussian noise in each of the three parameters of the motion model.
        These three parameters may be used as standard deviations for sampling.
        """
        Δ_xy =  xy - self.xy_ref # dist from prev loc --> curr loc in world coordinate system
        Δ_yaw = yaw - self.yaw_ref # yaw angle from prev pose -> curr pose in world coordinate system
        num_particles = len(self.particles) # [N,]
        # noise in the [trans_x trans_y rot] commands when moving the particles
        MOTION_NOISE = [0.5, 0.5, 0.1]
        xNoise = MOTION_NOISE[0]
        yNoise = MOTION_NOISE[1]
        rNoise = MOTION_NOISE[2]
        
        # if vehicle moves:
        if (Δ_xy ** 2).sum() > thrs ** 2:
            Δ_x = Δ_xy[0] + xNoise * torch.randn(num_particles).to(Δ_xy) # x with noise
            Δ_y = Δ_xy[1] + yNoise * torch.randn(num_particles).to(Δ_xy) # y with noise 
            Δ_xy = torch.stack([Δ_x,Δ_y],dim = 1) # [N,2]
            Δ_yaw = Δ_yaw + rad2deg(rNoise * torch.randn(num_particles).to(Δ_xy)) # yaw with noise
        # else:

        # rot_matrix = rotmat2d(Δ_yaw_ps) # rotation matrix [N,2,2]
        self.particles[:,:2] = self.particles[:,:2] + Δ_xy
        # particles[:,:2] = particles[:,:2] + torch.einsum('nij,nj -> ni',rot_matrix,Δ_xy_ps) # [N,2]
        self.particles[:,2] = self.particles[:,2] + Δ_yaw
    
    def update_weights(self,canvas,log_probs):
        """ This function update the weight for each particle using the log_prob map.
        Input:
            particles: each particle has four properties [x, y, theta, weight]
            log_probs: logical probability map, [H,W,N]
        Output:
            particles ... same particles with updated particles(i).weight
        """
        scores = torch.ones(len(self.particles),device = log_probs.device) * 0.00001
    
        # for idx in range(len(particles)):
        #     particle = particles[idx]

        # first check whether the particle is inside the map or not
        inside_mask = (self.particles[:,0] > canvas.bbox.min_[0]) & (self.particles[:,0] < canvas.bbox.max_[0]) & (self.particles[:,1] > canvas.bbox.min_[1]) & (self.particles[:,1] < canvas.bbox.max_[1])
        
        if inside_mask.sum() < 1:
            # inside_mask < 1 means all of particles locate outside previous map, skip!
            print("inside particles < 1 !!!")
            return inside_mask
        
        particles_ins = self.particles[inside_mask,:] # [K,4]
        particles_ins_uv = canvas.to_uv(particles_ins[:,:2])
        particles_ins_yaw = particles_ins[:,2]

        # update the weight
        H,W,N = log_probs.shape
        log_probs_particles_list = [sample_xyr(log_probs[None,None],particles_ins_uv[None,None,None,i:i+1,:],particles_ins_yaw[None,None,None,i:i+1],nearest_for_inf=True)[0][0,0,0,0] \
                                for i in range(len(particles_ins))] # [1,1,H,W,N]
        log_probs_particles = torch.cat(log_probs_particles_list)
        # particle_indices = int(particle[:,0]) * (W * N) + int(particle[:,1]) * (N) + int(particle[:,2]) # h * W * N + w * N + n
        # log_probs_particles = log_probs.flatten()[particle_indices] # [N]
        
        scores[inside_mask] = torch.exp(0.5 * log_probs_particles  / (2.0 ** 2)) + 0.00001 # 1e-6 for stability，e^((0.5 * log_probs^2) / 4.0), higher prob --> higher scores, lower prob --> lower scores
    
        # normalization
        self.particles[:, 3] = self.particles[:, 3] * scores
        self.particles[:, 3] = self.particles[:, 3] / torch.max(self.particles[:, 3]) # 1e-6 for stability

        dist_particles = (self.particles[None,:,:2] - self.particles[:,None,:2]).norm(dim = 2,p = 2) # [1,N,2] - [N,1,2] -> [N,N,2] -> [N,N]
        # check convergence using supporting tile map idea
        if dist_particles.max() < 20.0 and not self.is_converged:
            self.is_converged = True
            # print('Converged!')
            # cutoff redundant particles and leave only num of particles
            idxes = torch.argsort(self.particles[:, 3],descending=True)
            self.particles = self.particles[idxes[:200]]
    
        # return particles, len(particles)

    def resample(self):
        """ Re-sampling module,
        here we use the classical low variance re-sampling.
        """
        weights = self.particles[:, 3]
  
        # normalize the weights
        weights = weights / sum(weights)
  
        # compute effective number of particles
        eff_N = 1 / sum(weights ** 2)
  
        # resample
        new_particles = torch.zeros(self.particles.shape,device = self.particles.device)
        i = 0
        if eff_N < len(self.particles) * 3.0 / 4.0:
            r = torch.rand(1,device = self.particles.device) * 1.0 / len(self.particles)
            c = weights[0]
            for idx in range(len(self.particles)):
                u = r + idx/len(self.particles)
                while u > c:
                    if i >= len(self.particles) - 1:
                        break
                    i += 1
                    c += weights[i]
                new_particles[idx] = self.particles[i]
        else:
            new_particles = self.particles
        
        self.particles = new_particles

    def update(self,observation,xyr_topk,canvas,xy,yaw,first = False):
        """
        observation: log_prob of current map, [H,W,N]
        xyr_topk: topk pose of current frame, [N,3]
        canvas: query canvas of current frame,
        xy: gt_xy of current frame, for motion model
        yaw: gt_yaw of current frame, for motion model
        first: bool, return the xyr_topk[0,:] for the first frame
        """
        # initialization (for ref frame)
        if self.canvas is None: 
            self.canvas = canvas
        if self.xy_ref is None:
            self.xy_ref = xy # xy_gt (prev!)
            self.yaw_ref = yaw # yaw_gt (prev!)
            self.rotmat_ref = rotmat2d(deg2rad(self.yaw_ref)) # rotmat_gt (prev!)
        if self.num_rotations is None:
            self.num_rotations = observation.shape[-1]
        if self.belief is None:
            self.belief = observation.new_zeros(
                (self.canvas.h, self.canvas.w, self.num_rotations)
            ) # belief map of the canvas ?
        if self.particles is None:
            self.particles = self.init_particles_given_coords(xyr_topk,init_weight = 1.0) # particles init
        if first == True:
            return xyr_topk[0,:]

        # update canvas of current frame
        # self.canvas = canvas

        # observation model of each pose
        log_probs = observation
        
        # init particles with topk pose
         # [N,4]

        # motion model
        self.motion_model(xy,yaw)

        self.update_weights(canvas,log_probs)

        # resampling
        self.resample()

        # particle samples
        particles_uv = canvas.to_uv(self.particles[:,:2])
        particles_r = (self.particles[:,2:3] % 360) / 360.0 * log_probs.shape[2]
        particles_uvr = torch.cat([particles_uv,particles_r],dim = 1).int()
        particles_score = self.particles[:,3]
        inside_mask = (particles_uv[:,0] >= 0) & (particles_uv[:,0] < log_probs.shape[1]) & (particles_uv[:,1] >= 0) & (particles_uv[:,1] < log_probs.shape[0])
        particles_uvr = particles_uvr[inside_mask,:]
        particles_score = particles_score[inside_mask]

        self.belief = torch.zeros_like(log_probs) + 1e-32
        self.belief[particles_uvr[:,1],particles_uvr[:,0],particles_uvr[:,2]] = particles_score
        # self.belief = log_probs # update observation
        self.canvas = canvas # update map
        self.xy_ref = xy # update xy of prev frame
        self.yaw_ref = yaw # update yaw of prev frame
        if inside_mask.sum() >= 1:
            xyr_est = (self.particles[inside_mask,:3] * self.particles[inside_mask,3:]).sum(dim = 0) / self.particles[inside_mask,3:].sum()
        else:  # inside_mask.sum() < 1
            # resample particles on the plane
            self.particles = self.init_particles_given_coords(xyr_topk,init_weight = 1.0)
            print("resample!")
            # the maximal probability pose
            xyr_est = xyr_topk[0,:3]

        return xyr_est # [3]
    
class EKFAligner(RigidAligner):
    def __init__(self,canvas_ref=None, xy_ref=None, yaw_ref=None, num_rotations=None, track_priors=False):
        super().__init__(canvas_ref, xy_ref, yaw_ref, num_rotations, track_priors)
        # self.xy_ref: GT xy coordinate of prev frame
        # self.yaw_ref: GT heading of prev frame
        # self.num_Particles = num_Particles
        self.xy_prev = None # estimated xy of prev frame
        self.yaw_prev = None # estimated yaw of prev frame
        self.is_converged = False
        self.P = None # prior P for covariance matrix
        self.Q = torch.zeros(3) # Covariance for EKF simulation
        self.R = 1.0 # Observation x,y,heading position covariance
        self.MOTION_NOISE = [0.00, 0.00, 0.00]

        
    
    def jacob_f(self,xy):
        """
        Jacobian of Motion Model

        motion model
        x_{t+1} = x_t + Δ_x
        y_{t+1} = y_t + Δ_y
        yaw_{t+1} = yaw_t + Δ_yaw
        so
        dx/dx = 1
        dy/dy = 1
        dyaw/dyaw = 1
        else = 0 (x,y,yaw are relatively irrelevant)
        """
        jF = torch.eye(3).to(xy)

        return jF
    
    def jacob_H(self,scores,canvas,xy,yaw):
        """
        score: score map of template matching, [H,W,N]
        xy: [B,2]
        yaw: [B,]
        return: jacobin matrix of observation model, # [3,B,H,W]
        """
        uv = canvas.to_uv(xy)
        r = yaw / 360.0 * scores.shape[-1]
        xyr = torch.cat([uv,r[None]],dim = -1) # [B,3]
        dxyr = torch.eye(3).to(xyr)
        _,jac_xyr = grid_sample_jac(scores[None,None,...],xyr[None,None,None,:],jac = dxyr[:,None,None,None,:])
        jac_xyr = jac_xyr[:,0,0,0,:].permute(1,0) # [1,3]
        return jac_xyr 
    
    def motion_model(self,xy,yaw,thrs = 0.1):
        """ MOTION performs the sampling from the proposal.
        distribution, here the rotation-translation-rotation motion model

        input:
            particles: the particles as in the main script
            xy: GT xy coordinates at current frame
            yaw: GT yaw at current frame

        output:
            the same particles, with updated poses.

        The position of the i-th particle is given by the 3D vector
        particles(i).pose which represents (x, y, theta).

        Assume Gaussian noise in each of the three parameters of the motion model.
        These three parameters may be used as standard deviations for sampling.
        """
        Δ_xy =  xy - self.xy_ref # dist from prev loc --> curr loc in world coordinate system
        Δ_yaw = yaw - self.yaw_ref # yaw angle from prev pose -> curr pose in world coordinate system
        # noise in the [trans_x trans_y rot] commands when moving the particles
        
        xNoise = self.MOTION_NOISE[0]
        yNoise = self.MOTION_NOISE[1]
        rNoise = self.MOTION_NOISE[2]
        
        # if vehicle moves:
        if (Δ_xy ** 2).sum() > thrs ** 2:
            Δ_x = Δ_xy[0] + xNoise * torch.randn(Δ_xy[0].shape).to(Δ_xy) # x with noise
            Δ_y = Δ_xy[1] + yNoise * torch.randn(Δ_xy[1].shape).to(Δ_xy) # y with noise 
            Δ_xy = torch.tensor([Δ_x,Δ_y]).to(xy) # [2]
            Δ_yaw = Δ_yaw + rad2deg(rNoise * torch.randn_like(Δ_yaw).to(Δ_yaw)) # yaw with noise
        # else:
        
        # add noise for simulation (motion)
        xy_noise = self.xy_prev + Δ_xy
        yaw_noise = self.yaw_prev + Δ_yaw

        # predict xy/yaw
        # self.xy_ref = xy
        # self.yaw_ref = yaw

        return xy_noise,yaw_noise
       
    def observation_model(self,log_probs,canvas,xy,yaw):
        """This function calculate the observated log_prob value with observation model (log_probs) and pose
        Input:
            log_probs: logistic probability map, [H,W,N]
            xy: observed xy at current frame, [2]
            yaw: observed heading angle at current frame
        """
        uv = canvas.to_uv(xy)
        score,_ = sample_xyr(log_probs[None,None,...],uv[None,None,None,None,:],yaw.reshape(-1)[None,None,None,:],nearest_for_inf = True) 
        if torch.isfinite(score) == False:
            score = -torch.ones_like(score).to(score) * 1e3
        return score

    def update(self,observation,canvas,xy,yaw,xy_obs,yaw_obs):
        """
        observation: log_prob of current map, [H,W,N]
        xy: GT xy at current frame, for motion model
        yaw: GT yaw at current frame, for motion model
        xy_obs: observed xy at current frame, for observation model
        yaw_obs: observed yaw at current frame, for observation model
        P: covariance matrix of prev frame
        """
        # initialization
        if self.xy_ref is None:
            self.xy_ref = xy # gt_xy (prev!)
        if self.yaw_ref is None:
            self.yaw_ref = yaw # gt_yaw (prev!)
        if self.xy_prev is None:
            self.xy_prev = xy_obs
        if self.yaw_prev is None:
            self.yaw_prev = yaw_obs
        if self.P is None:
            self.P = torch.eye(3).to(xy)
            self.Q = self.Q.to(xy)
            # self.R = self.R.to(xy)

            xyr_est = torch.cat([xy,yaw[...,None]],dim = -1)
            return xyr_est

        # motion model (from prev frame to curr frame)
        xy_pred,yaw_pred = self.motion_model(xy,yaw) # curr frame (motion)
        xyr_pred = torch.cat([xy_pred,(yaw_pred / 180.0 * torch.pi)[...,None]],dim = -1) # [3?]
        jF = self.jacob_f(xy) # jacobin matrix of motion model at prev frame (est)
        P_pred = jF @ self.P @ jF.T + self.Q # prior covariance 
        
        # score from motion model (prediction)
        s_pred = self.observation_model(observation,canvas,xy_pred,yaw_pred)
        # score from observation model (log_prob value of curr pose (observed))
        s_obs = self.observation_model(observation,canvas,xy_obs,yaw_obs)
        # update
        res = (s_obs - s_pred).item()
        jH = self.jacob_H(observation,canvas,xy_pred,yaw_pred)
        S = jH @ P_pred @ jH.T + self.R
        K = P_pred @ jH.T @ torch.linalg.inv(S)
        xyr_est = xyr_pred + (K * res)[:,0]
        

        xyr_est[...,2] = xyr_est[...,2] * 180.0 / torch.pi
        self.xy_ref = xy
        self.yaw_ref = yaw
        self.xy_prev = xyr_est[...,:2]
        self.yaw_prev = xyr_est[...,2]
        self.P = (torch.eye(xyr_est.shape[-1]).to(xyr_est) - K @ jH) @ P_pred

        return xyr_est

class GPSAligner(RigidAligner):
    def __init__(self, distribution=log_laplace, **kwargs):
        self.distribution = distribution
        super().__init__(**kwargs)
        if self.num_rotations is None:
            raise ValueError("Rotation number is required.")
        angles = torch.arange(0, 360, 360 / self.num_rotations)
        self.rotmats = rotmat2d(deg2rad(-angles))
        self.xy_grid = None

    def update(self, xy_gps, accuracy, canvas, xy, yaw):
        # initialization
        if self.canvas is None:
            self.canvas = canvas
        if self.xy_ref is None:
            self.xy_ref = xy
            self.yaw_ref = yaw
            self.rotmat_ref = rotmat2d(deg2rad(self.yaw_ref))
        if self.xy_grid is None:
            self.xy_grid = self.canvas.to_xy(make_grid(self.canvas.w, self.canvas.h))
        if self.belief is None:
            self.belief = xy_gps.new_zeros(
                (self.canvas.h, self.canvas.w, self.num_rotations)
            )

        # integration
        Δ_xy = self.rotmat_ref @ (xy - self.xy_ref)
        Δ_xy_world = torch.einsum("nij,j->ni", self.rotmats.to(xy), Δ_xy)
        xy_grid_prev = (
            self.xy_grid.to(xy)[..., None, :] + Δ_xy_world[..., None, None, :, :]
        )
        prior = self.distribution(xy_grid_prev, xy_gps, accuracy)
        self.belief.add_(prior)
        self.belief.sub_(self.belief.max())  # normalize to avoid overflow

        if self.priors is not None:
            self.priors.append(prior.cpu())
        return prior
