# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.nn.functional import normalize

from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection, PolarProjectionDepth
from .voting import (
    argmax_xyr,
    topk_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
    nll_normal_loss_xyr,
    res_l1loss_xyr,
    res_sem_xyr,
    TemplateSampler,
)
from .map_encoder import MapEncoder
from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall
from einops import rearrange



class OrienterNet(BaseModel):
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
        "sigma_xy": 1,
        "sigma_r": 2,
        # depcreated
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
        "topk": 20
    }

    def _init(self, conf):
        assert not self.conf.norm_depth_scores
        assert self.conf.depth_parameterization == "scale"
        assert not self.conf.normalize_scores_by_dim
        assert self.conf.normalize_scores_by_num_valid
        assert self.conf.prior_renorm

        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2")) # default = None, feature_extractor_v2 for encode+decoder
        self.image_encoder = Encoder(conf.image_encoder.backbone) # resnet-101:encoder in default, FCN:decoder
        self.map_encoder = MapEncoder(conf.map_encoder)
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)

        ppm = conf.pixel_per_meter
        self.projection_polar = PolarProjectionDepth(
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

        # Extract image features.
        level = 0
        f_image = self.image_encoder(data)["feature_maps"][level]
        camera = data["camera"].scale(1 / self.image_encoder.scales[level])
        camera = camera.to(data["image"].device, non_blocking=True)

        # Estimate the monocular priors.
        pred["pixel_scales"] = scales = self.scale_classifier(f_image.moveaxis(1, -1))
        f_polar,depth_prob = self.projection_polar(f_image, scales, camera)
        depth_prob = depth_prob.permute(0,-1,1,2) # .softmax(dim = 1)

        depth_mask = torch.arange(1,depth_prob.shape[1] + 1,1,device = depth_prob.device).reshape(1,-1,1,1).flip(1)
        depth_abs = (depth_prob * depth_mask).sum(dim = 1,keepdim = True)
        
        depth_abs_inv = 1.0 / depth_abs
        depth_rel_inv = (depth_abs_inv - depth_abs_inv.amin(dim = (1,2,3),keepdim = True)) / (depth_abs_inv.amax(dim = (1,2,3),keepdim = True) - depth_abs_inv.amin(dim = (1,2,3),keepdim = True))

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
            f_bev_fea = pred_bev["features"]

        scores, f_bev_template, valid_template = self.exhaustive_voting(
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
            uvr_max = argmax_xyr(scores).to(scores)
            uvr_topk = topk_xyr(scores, k = self.conf.topk).to(scores)
            uvr_avg, _ = expectation_xyr(log_probs.exp())

        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_topk": uvr_topk,
            "uv_topk": uvr_topk[..., :2],
            "yaw_topk": uvr_topk[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "features_image": f_image,
            "features_bev": f_bev, # cover 32x64.5m area front of the camera
            "features_map":f_map,
            "embeddings_map": embedding_map,
            "valid_bev": valid_bev.squeeze(1),
            "valid_template": valid_template,
            "depth": depth_rel_inv
        }
    
    def _forward_wrapper(self, data):
        pred = {}
        pred_map,embedding_map = self.map_encoder(data)
        pred["map"] = pred_map
        # pred_map = pred["map"] = self.map_encoder(data)
        f_map = pred_map["map_features"][0]

        # Extract image features.
        level = 0
        f_image = self.image_encoder(data)["feature_maps"][level]
        # disparity = output["depth"]
        
        c = data["c"]
        f = data["f"]

        f = f / 2.0
        f = f.to(data["image"].device, non_blocking=True)

        c = (c + 0.5) * 2.0 - 0.5
        c = c.to(data["image"].device, non_blocking=True)

        # Estimate the monocular priors.
        pred["pixel_scales"] = scales = self.scale_classifier(f_image.moveaxis(1, -1))

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
            # f_bev_fea = self.sem_classifier(pred_bev["features"])

        scores, _,_ = self.exhaustive_voting(
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
            max_uvr = argmax_xyr(scores).to(scores)
        return log_probs,max_uvr 
        # with torch.no_grad():
        #     argmax_xyr(scores).to(scores)
            # expectation_xyr(log_probs.exp())
            # topk_xyr(scores, k = 1000).to(scores)



    def forward(self, data):
        if "wrapper" not in data.keys() or data["wrapper"] == False:
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
        loss = {"total": nll, "nll": nll}
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

            "xy_recall_1m_top{:d}".format(self.conf.topk): Location2DRecall(1.0, self.conf.pixel_per_meter, "uv_topk",topk = True),
            "xy_recall_3m_top{:d}".format(self.conf.topk): Location2DRecall(3.0, self.conf.pixel_per_meter, "uv_topk",topk = True),
            "xy_recall_5m_top{:d}".format(self.conf.topk): Location2DRecall(5.0, self.conf.pixel_per_meter, "uv_topk",topk = True),
            "yaw_recall_1°_top{:d}".format(self.conf.topk): AngleRecall(1.0, "yaw_topk",topk = True),
            "yaw_recall_3°_top{:d}".format(self.conf.topk): AngleRecall(3.0, "yaw_topk",topk = True),
            "yaw_recall_5°_top{:d}".format(self.conf.topk): AngleRecall(5.0, "yaw_topk",topk = True),
        
        }
