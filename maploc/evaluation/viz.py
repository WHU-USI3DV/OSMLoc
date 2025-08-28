# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from torch.nn.functional import grid_sample
from ..models.utils import make_grid, rotmat2d

from ..utils.io import write_torch_image
from ..utils.viz_2d import plot_images, features_to_RGB, save_plot
from ..utils.viz_localization import (
    likelihood_overlay,
    plot_pose,
    plot_dense_rotations,
    add_circle_inset,
)
from ..osm.viz import Colormap, plot_nodes

def crop_map(map,uv,yaw,bev_size):
    map_size = map.new_tensor(map.shape[-2:][::-1])
    grid_uv = make_grid(bev_size[0],bev_size[1],step_x = 1,step_y = 1,orig_x = -(bev_size[0] // 2),orig_y = -bev_size[1],y_up = False,device = bev_size.device)
    rotmats = rotmat2d(yaw / 180 * torch.pi)
    grid_uv_rot = torch.einsum("...nij, ...hwj -> ...nhwi", rotmats, grid_uv)
    grid_uv_map = grid_uv_rot + uv.view(-1,1,1,2)
    grid_uv_map_norm = grid_uv_map / (map_size.view(1,-1) - 1)
    grid_uv_map_norm  = grid_uv_map_norm * 2 - 1
    feats_m_crop = grid_sample(map[None,...],grid_uv_map_norm,align_corners=True, mode = 'nearest')[0]
    mask_map = torch.ones_like(map[None,:1,:,:])
    mask_map = grid_sample(mask_map,grid_uv_map_norm,align_corners=True, mode = 'nearest')[0].squeeze(0) > -1
    mask_map[grid_uv[...,0].abs() > grid_uv[...,1].abs()] = False
    
    return feats_m_crop,mask_map


def plot_example_single(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=True,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
):
    scene, name, rasters, uv_gt = (data[k] for k in ("scene", "name", "map", "uv"))
    uv_gps = data.get("uv_gps")
    yaw_gt = data["roll_pitch_yaw"][-1].numpy()
    image = data["image"].permute(1, 2, 0)
    if "valid" in data:
        image = image.masked_fill(~data["valid"].unsqueeze(-1), 0.3)

    depth_pred = (pred["depth"] - pred["depth"].amin(dim = (-2,-1),keepdims = True)) / (pred["depth"].amax(dim = (-2,-1), keepdims = True) - pred["depth"].amin(dim = (-2,-1),keepdims = True))
    depth_pred = (depth_pred * 255.0)[0].numpy().astype(np.uint8)
    depth_pred = np.repeat(depth_pred[...,None],3,axis = -1)
    # depth_gt = (data["depth"] * 255.0)[0].numpy().astype(np.uint8)
    # depth_gt = np.repeat(depth_gt[...,None],3,axis = -1)
    # depth_color = cv2.applyColorMap(depth_norm,cv2.COLORMAP_INFERNO)
    lp_uvt = lp_uv = pred["log_probs"]
    if show_fused and "log_probs_fused" in pred:
        lp_uvt = lp_uv = pred["log_probs_fused"]
    elif not show_masked_prob and "scores_unmasked" in pred:
        lp_uvt = lp_uv = pred["scores_unmasked"]
    has_rotation = lp_uvt.ndim == 3
    if has_rotation:
        lp_uv = lp_uvt.max(-1).values
    if lp_uv.min() > -np.inf:
        lp_uv = lp_uv.clip(min=np.percentile(lp_uv, 1))
    prob = lp_uv.exp()
    uv_p, yaw_p = pred["uv_max"], pred.get("yaw_max")
    if show_fused and "uv_fused" in pred:
        uv_p, yaw_p = pred["uv_fused"], pred.get("yaw_fused")
    feats_map = pred["map"]["map_features"][0]
    feats_image = pred["features_image"]
    (feats_map_rgb,) = features_to_RGB(feats_map.numpy())
    (feats_image_rgb,) = features_to_RGB(feats_image.numpy())

    text1 = rf'$\Delta xy$: {results["xy_max_error"]:.1f}m'
    if has_rotation:
        text1 += rf', $\Delta\theta$: {results["yaw_max_error"]:.1f}°'
    if show_fused and "xy_fused_error" in results:
        text1 += rf', $\Delta xy_{{fused}}$: {results["xy_fused_error"]:.1f}m'
        text1 += rf', $\Delta\theta_{{fused}}$: {results["yaw_fused_error"]:.1f}°'
    if show_dir_error and "directional_error" in results:
        err_lat, err_lon = results["directional_error"]
        text1 += rf",  $\Delta$lateral/longitundinal={err_lat:.1f}m/{err_lon:.1f}m"
    if "xy_gps_error" in results:
        text1 += rf',  $\Delta xy_{{GPS}}$: {results["xy_gps_error"]:.1f}m'

    map_viz = Colormap.apply(rasters)
    overlay = likelihood_overlay(prob.numpy(), map_viz.mean(-1, keepdims=True))
    plot_images(
          [image, map_viz, overlay, feats_map_rgb,feats_image_rgb,depth_pred],
          titles=[text1, "map", "likelihood", "neural map","neural","depth_pred"],
          dpi=75,
          cmaps="jet",
    )
    # plot_images(
    #      [image, map_viz, overlay, feats_map_rgb, depth_pred],
    #      titles=[text1, "map", "likelihood", "neural map", "depth_pred"],
    #      dpi=75,
    #      cmaps="jet",
    #  )
    fig = plt.gcf()
    axes = fig.axes
    axes[1].images[0].set_interpolation("none")
    axes[2].images[0].set_interpolation("none")
    Colormap.add_colorbar()
    plot_nodes(1, rasters[2])

    if show_gps and uv_gps is not None:
        plot_pose([1], uv_gps, c="blue")
    plot_pose([1], uv_gt, yaw_gt, c="red")
    plot_pose([1], uv_p, yaw_p, c="k")
    plot_dense_rotations(2, lp_uvt.exp())
    inset_center = pred["uv_max"] if results["xy_max_error"] < 5 else uv_gt
    axins = add_circle_inset(axes[2], inset_center)
    axins.scatter(*uv_gt, lw=1, c="red", ec="k", s=50, zorder=15)
    axes[0].text(
        0.003,
        0.003,
        f"{scene}/{name}",
        transform=axes[0].transAxes,
        fontsize=3,
        va="bottom",
        ha="left",
        color="w",
    )
    plt.show()
    if out_dir is not None:
        name_ = name.replace("/", "_")
        p = str(out_dir / f"{scene}_{name_}_{{}}.pdf")
        save_plot(p.format("pred"))
        plt.close()

        if fig_for_paper:
            # !cp ../datasets/MGL/{scene}/images/{name}.jpg {out_dir}/{scene}_{name}.jpg
            plot_images([map_viz])
            plt.gca().images[0].set_interpolation("none")
            plot_nodes(0, rasters[2])
            plot_pose([0], uv_gt, yaw_gt, c="red")
            plot_pose([0], pred["uv_max"], pred["yaw_max"], c="k")
            save_plot(p.format("map"))
            plt.close()
            plot_images([lp_uv], cmaps="jet")
            plot_dense_rotations(0, lp_uvt.exp())
            save_plot(p.format("loglikelihood"), dpi=100)
            plt.close()
            plot_images([overlay])
            plt.gca().images[0].set_interpolation("none")
            axins = add_circle_inset(plt.gca(), inset_center)
            axins.scatter(*uv_gt, lw=1, c="red", ec="k", s=50)
            save_plot(p.format("likelihood"))
            plt.close()
            write_torch_image(
                p.format("neuralmap").replace("pdf", "jpg"), feats_map_rgb
            )
            write_torch_image(p.format("image").replace("pdf", "jpg"), image.numpy())

    if not plot_bev:
        return

    feats_q = pred["features_bev"]
    feats_s = pred["features_bev_sem"]
    feats_m = pred["features_map_sem"]
    # feats_d = pred["depth_bev"]
    mask_bev = pred["valid_bev"]

    # xy_gt = data["uv"]
    yaw_gt = data["roll_pitch_yaw"][..., -1:]
    bev_size = feats_q.new_tensor(feats_q.shape[-2:][::-1])
    feats_m_gt,mask_map_gt = crop_map(feats_m,uv_gt,yaw_gt,bev_size)
    feats_m_pr,mask_map_pr = crop_map(feats_m,uv_p,yaw_p[None],bev_size)
    grid_uv = make_grid(bev_size[0],bev_size[1],step_x = 1,step_y = 1,orig_x = -(bev_size[0] // 2),orig_y = -bev_size[1],y_up = False,device = bev_size.device)

    prior = None
    if "log_prior" in pred["map"]:
        prior = pred["map"]["log_prior"][0].sigmoid()
    if "bev" in pred and "confidence" in pred["bev"]:
        conf_q = pred["bev"]["confidence"]
    else:
        conf_q = torch.norm(feats_q, dim=0)
    conf_q = conf_q.masked_fill(~mask_bev, np.nan)
    (feats_q_rgb,) = features_to_RGB(feats_q.numpy(), masks=[mask_bev.numpy()])
    (feats_s_rgb,) = features_to_RGB(feats_s.numpy(), masks=[mask_bev.numpy()])
    (feats_m_gt_rgb,) = features_to_RGB(feats_m_gt.numpy(), masks=[mask_map_gt.numpy()])
    (feats_m_pr_rgb,) = features_to_RGB(feats_m_pr.numpy(), masks=[mask_map_pr.numpy()])
    # feats_d_rgb = (feats_d[0] - feats_d[0].amin()) / ((feats_d[0].amax() - feats_d[0].amin())).numpy() * 255.0
    # feats_d_rgb = feats_d_rgb.masked_fill(~mask_bev, np.nan)
    # feats_d_rgb[~mask_bev] = np.nan
    # import cv2
    # cv2.imwrite("temp.png",feats_q_rgb)
    # feats_map_rgb, feats_q_rgb, = features_to_RGB(
    #     feats_map.numpy(), feats_q.numpy(), masks=[None, mask_bev])
    norm_map = torch.norm(feats_map, dim=0)

    plot_images(
        [conf_q, feats_q_rgb, feats_s_rgb, feats_m_gt_rgb, feats_m_pr_rgb, norm_map] + ([] if prior is None else [prior]),
        titles=["BEV confidence", "BEV features", "BEV semantic", "map gt semantic", "map pr semantic", "map norm"]
        + ([] if prior is None else ["map prior"]),
        dpi=50,
        cmaps="jet",
    )
    plt.show()

    if out_dir is not None:
        save_plot(p.format("bev"))
        plt.close()


def plot_example_sequential(
    idx,
    model,
    pred,
    data,
    results,
    plot_bev=True,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
):
    return
