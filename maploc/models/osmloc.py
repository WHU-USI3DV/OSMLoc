# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.nn.functional import normalize
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection, PolarProjectionDepth,ProjectionScale, ProjectionScalePolar
from .voting import (
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
    nll_normal_loss_xyr,
    res_l1loss_xyr,
    depth_loss,
    res_sem_xyr,
    topk_xyr,
    TemplateSampler,
)
from .map_encoder import MapEncoder
from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall
from .depth_anything.dpt import DepthAnything

class ConvModule(nn.Module):
    def __init__(self,in_dim, latent_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim,latent_dim,1)
        self.norm1 = nn.BatchNorm2d(latent_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(latent_dim, out_dim, 1)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(self.norm1(x))
        x = self.conv2(x)
        
        return x

class botten_block(nn.Module):
    def __init__(self,num_blocks,in_dim,out_dim):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            blocks.append(Bottleneck(in_dim,in_dim // Bottleneck.expansion))
        self.blocks = nn.Sequential(*blocks)
        self.adaptation = nn.Conv2d(in_dim,out_dim,1)
    
    def forward(self,x):
        x = self.blocks(x)
        x = self.adaptation(x)

        return x

class osmloc(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "bev_net": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "scale_range": [0, 9],
        "num_scale_bins": "???",
        "z_min": None,
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": True,
        "do_label_smoothing": False,
        # depcreated
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
    }

    def _init(self, conf):
        assert not self.conf.norm_depth_scores
        assert self.conf.depth_parameterization == "scale"
        assert not self.conf.normalize_scores_by_dim
        assert self.conf.normalize_scores_by_num_valid
        assert self.conf.prior_renorm

        # Encoder = DepthAnything(conf.image_encoder)
        self.image_encoder = DepthAnything(conf.image_encoder)
        # frozen
        for n,p in self.image_encoder.pretrained.named_parameters():
            p.requires_grad = False
        for n,p in self.image_encoder.depth_head.named_parameters():
            p.requires_grad = False
        # self.botteneck = botten_block(2,conf.latent_dim,conf.latent_dim)
        self.map_encoder = MapEncoder(conf.map_encoder)
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)


        ppm = conf.pixel_per_meter
        self.projection_polar = ProjectionScalePolar(
            conf.z_max,
            ppm,
            conf.scale_range,
            conf.z_min,
        )

        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min
        )

        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_xz, ppm, conf.num_rotations
        )
        
        self.scale_classifier = torch.nn.Linear(conf.latent_dim, conf.num_scale_bins)
        # self.scale_classifier = ConvModule(conf.latent_dim, 64, 64)
        self.sem_classifier = torch.nn.Sequential(torch.nn.Conv2d(conf.latent_dim,conf.map_encoder.embedding_dim,1),
                                                              torch.nn.BatchNorm2d(conf.map_encoder.embedding_dim),torch.nn.ReLU(),
                                                              torch.nn.Conv2d(conf.map_encoder.embedding_dim,conf.map_encoder.embedding_dim * 3 ,1))

        if conf.bev_net is None:
            self.feature_projection = torch.nn.Linear(
                conf.latent_dim, conf.matching_dim
            )
        if conf.add_temperature:
            temperature = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)

    def exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev=None):
        # f_bev: [4, 8, 64, 129]
        # f_map: [4, 8, 256, 256]
        if self.conf.normalize_features:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)

        # Build the templates and exhaustively match against the map.
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        templates = self.template_sampler(f_bev)
        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float(),
                padding_mode=self.conf.padding_matching,
            )
        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)

        # Reweight the different rotations based on the number of valid pixels
        # in each template. Axis-aligned rotation have the maximum number of valid pixels.
        valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        scores = scores / num_valid[..., None, None]
        return scores,templates,valid_templates

    def _forward(self, data):
        pred = {}
        pred_map,embedding_map = self.map_encoder(data)
        pred["map"] = pred_map
        f_map = pred_map["map_features"][0]


        output = self.image_encoder(data["image"])
        f_image = output["features"]
        disparity = output["depth"]
        
        camera = data["camera"].scale(1 / 2.0)
        camera = camera.to(data["image"].device, non_blocking=True)

        # Estimate the monocular priors.
        pred["pixel_scales"] = scales = self.scale_classifier(f_image.moveaxis(1, -1))

        f_polar, abs_depth = self.projection_polar(
                                        f_image,
                                        data["valid"].unsqueeze(1),
                                        scales,
                                        # depth_logit,
                                        camera
                                        )
        # Map to the BEV.
        with torch.autocast("cuda", enabled=False):
            f_bev, valid_bev, _ = self.projection_bev(
                f_polar.float(), None, camera.float()
            )
        pred_bev = {}
        if self.conf.bev_net is None:
            # channel last -> classifier -> channel first
            f_bev = self.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)
        else:
            pred_bev = pred["bev"] = self.bev_net({"input": f_bev})
            f_bev = pred_bev["output"]
            f_bev_fea = self.sem_classifier(pred_bev["features"])

        scores, f_bev_template, valid_template = self.exhaustive_voting(
            f_bev, f_map, valid_bev, pred_bev.get("confidence")
        )
        scores = scores.moveaxis(1, -1)  # B,H,W,N
        if "log_prior" in pred_map and self.conf.apply_map_prior:
            scores = scores + pred_map["log_prior"][0].unsqueeze(-1)
        # pred["scores_unmasked"] = scores.clone()
        if "map_mask" in data:
            scores.masked_fill_(~data["map_mask"][..., None], -np.inf)
        if "yaw_prior" in data:
            mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)
        log_probs = log_softmax_spatial(scores)
        with torch.no_grad():
            uvr_max = argmax_xyr(scores).to(scores)
            uvr_avg, _ = expectation_xyr(log_probs.exp())
            uvr_topk = topk_xyr(scores, k = 1000).to(scores)

        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "uvr_topk": uvr_topk,
            "uv_topk": uvr_topk[..., :2],
            "yaw_topk": uvr_topk[..., 2],
            # "features_depth": f_depth,
            "features_image": f_image,
            "features_bev": f_bev, # cover 32x64.5m area front of the camera
            # "features_bev_template": f_bev_template, # cover 64.5mx64.5m area (camera-centric)
            "features_map":f_map,
            "depth": abs_depth,
            # "depth_bev": bev_depth,
            "disparity": disparity,
            "features_bev_sem": f_bev_fea,
            # "features_bev_template_sem": f_bev_fea_template,
            "features_map_sem": embedding_map,
            "valid_bev": valid_bev.squeeze(1),
            "valid_template": valid_template,
        }
    
    def _forward_wrapper(self, data):
        pred = {}
        pred_map,embedding_map = self.map_encoder(data)
        pred["map"] = pred_map
        # pred_map = pred["map"] = self.map_encoder(data)
        f_map = pred_map["map_features"][0]

        # Extract image features.
        # level = 0
        # with torch.no_grad(): # frozen image encoder (depth anything)
        output = self.image_encoder(data["image"])
        f_image = output["features"]
        disparity = output["depth"]
        
        c = data["c"]
        f = data["f"]

        f = f / 2.0
        f = f.to(data["image"].device, non_blocking=True)

        c = (c + 0.5) * 2.0 - 0.5
        c = c.to(data["image"].device, non_blocking=True)
        # f_image = self.botteneck(f_depth)
        # if type(data["camera"]) == torch.Tensor:
        #     camera = data["camera"] / 2.0
        #     camera = camera.to(data["image"].device, non_blocking=True)
        # else:
        # camera = data["camera"].scale(1 / 2.0)
        # camera = camera.to(data["image"].device, non_blocking=True)

        # Estimate the monocular priors.
        pred["pixel_scales"] = scales = self.scale_classifier(f_image.moveaxis(1, -1))
        # depth_logit = self.depth_classifier(f_image) # --> [0,+oo)
        # rel_depths = (rel_depths - rel_depths.amin(dim = (-2,-1),keepdims = True)) / (rel_depths.amax(dim = (-2,-1),keepdims = True) - rel_depths.amin(dim = (-2,-1),keepdims = True)) # [0,+oo) --> [0,1]
        # pred["pixel_depths"] = rel_depths

        f_polar, abs_depth = self.projection_polar.forward_wrapper(
                                        f_image,
                                        data["valid"].unsqueeze(1),
                                        scales,
                                        # depth_logit,
                                        f
                                        )
        # Map to the BEV.
        with torch.autocast("cuda", enabled=False):
            f_bev, valid_bev, _ = self.projection_bev.forward_wrapper(
                f_polar.float(), None, c.float(),f.float()
            )
        pred_bev = {}
        if self.conf.bev_net is None:
            # channel last -> classifier -> channel first
            f_bev = self.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)
        else:
            pred_bev = pred["bev"] = self.bev_net({"input": f_bev})
            f_bev = pred_bev["output"]
            f_bev_fea = self.sem_classifier(pred_bev["features"])

        scores, _,_ = self.exhaustive_voting(
            f_bev, f_map, valid_bev, pred_bev.get("confidence")
        )
        # f_bev_fea = self.sem_classifier(f_bev_fea) # latent_dim --> out_dim
        # f_bev_fea_template = self.template_sampler(f_bev_fea) # template sampling, [B,N,C,H,W]
        scores = scores.moveaxis(1, -1)  # B,H,W,N
        if "log_prior" in pred_map and self.conf.apply_map_prior:
            scores = scores + pred_map["log_prior"][0].unsqueeze(-1)
        # pred["scores_unmasked"] = scores.clone()
        if "map_mask" in data:
            scores.masked_fill_(~data["map_mask"][..., None], -np.inf)
        if "yaw_prior" in data:
            mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)
        log_probs = log_softmax_spatial(scores)
        with torch.no_grad():
            max_uvr = argmax_xyr(scores).to(scores)
        return log_probs,max_uvr 
        # with torch.no_grad():
        #     argmax_xyr(scores).to(scores)
            # expectation_xyr(log_probs.exp())
            # topk_xyr(scores, k = 1000).to(scores)



    def forward(self, data):
        if "wrapper" not in data.keys():
            return self._forward(data)
        else:
            return self._forward_wrapper(data)

    def loss(self, pred, data):
        xy_gt = data["uv"]
        yaw_gt = data["roll_pitch_yaw"][..., -1]
        if self.conf.do_label_smoothing:
            nll = nll_loss_xyr_smoothed(
                pred["log_probs"],
                xy_gt,
                yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r,
                mask=data.get("map_mask"),
            )
        else:
            nll = nll_loss_xyr(pred["log_probs"], xy_gt, yaw_gt)
        loss_depth = 20.0 * depth_loss(pred["depth"],pred["disparity"],data["valid"].unsqueeze(1))
        res = 10.0 * res_l1loss_xyr(pred["features_bev_sem"],pred["features_map_sem"],xy_gt,yaw_gt,pred["valid_bev"])
        # res = res_l1loss_xyr(pred["features_bev_fea"],pred["embeddings_map"],xy_gt,yaw_gt,pred["valid_bev"])
        # res = res_l1loss_xyr(pred["features_bev"],pred["features_map"],xy_gt,yaw_gt,pred["valid_bev"])
        loss = {"total": nll + loss_depth + res, "nll": nll, "loss_depth":loss_depth, "loss_res": res}
        if self.training and self.conf.add_temperature:
            loss["temperature"] = self.temperature.expand(len(nll))
        return loss

    def metrics(self):
        return {
            "xy_max_error": Location2DError("uv_max", self.conf.pixel_per_meter),
            "xy_expectation_error": Location2DError(
                "uv_expectation", self.conf.pixel_per_meter
            ),
            "yaw_max_error": AngleError("yaw_max"),
            "xy_recall_1m": Location2DRecall(1.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_3m": Location2DRecall(3.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "uv_max"),
            "yaw_recall_1°": AngleRecall(1.0, "yaw_max"),
            "yaw_recall_3°": AngleRecall(3.0, "yaw_max"),
            "yaw_recall_5°": AngleRecall(5.0, "yaw_max"),
        }
