# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import OmegaConf, DictConfig

from .. import logger
from ..conf import data as conf_data_dir
from ..data import MapillaryDataModule
from .run import evaluate
import pickle


split_mgl = {
    "val": {
        "scenes": [
            "sanfrancisco_soma",
            "sanfrancisco_hayes",
            "amsterdam",
            "berlin",
            "lemans",
            "montrouge",
            "toulouse",
            "nantes",
            "vilnius",
            "avignon",
            "helsinki",
            "milan",
            "paris",
        ],
    },
}

split_taipei = {
             "val": {
                 "scenes": [
                     "taipei"
                 ],
            },
         }

split_brisbane = {
             "val": {
                 "scenes": [
                     "brisbane"
                 ],
            },
         }

split_detroit = {
             "val": {
                 "scenes": [
                     "detroit"
                 ],
            },
         }

split_munich = {
             "val": {
                 "scenes": [
                     "munich"
                 ],
            },
         }

split_configs = {
    "mgl": split_mgl,
    "taipei": split_taipei,
    "brisbane": split_brisbane,
    "detroit": split_detroit,
    "munich": split_munich,
}


def run(
    checkpoint: str,
    dataset: str,
    split: str,
    experiment: str,
    cfg: Optional[DictConfig] = None,
    sequential: bool = False,
    particle: bool = False,
    ekf: bool = False,
    thresholds: Tuple[int] = (1, 3, 5),
    **kwargs):  

    data_cfg_train = OmegaConf.load(Path(conf_data_dir.__file__).parent / f"mapillary_{dataset}.yaml")
    data_cfg = OmegaConf.merge(
        data_cfg_train,
        {
            "return_gps": True,
            "add_map_mask": True,
            "max_init_error": 32,
            "loading": {"val": {"batch_size": 1, "num_workers": 0}},
        },
    )
    default_cfg_single = OmegaConf.create({"data": data_cfg})
    default_cfg_sequential = OmegaConf.create(
        {
            **default_cfg_single,
            "chunking": {
                "min_length": 3,
                "max_length": 10
            },
        }
    )
    cfg = cfg or {}
    assert particle == False or (particle == True and sequential * particle == True), "if particle=True, sequential and particle must be true !"
    
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    default = default_cfg_sequential if sequential else default_cfg_single
    default = OmegaConf.merge(default, dict(data=split_configs[dataset][split]))
    cfg = OmegaConf.merge(default, cfg)
    dataset = MapillaryDataModule(cfg.get("data", {}))
    metrics = evaluate(checkpoint, cfg, dataset, split, sequential=sequential,particle = particle,ekf=ekf, experiment=experiment, **kwargs)

    keys = [
        "xy_max_error",
        # "xy_gps_error",
        "yaw_max_error",
        # "xy_scale_error",
        # "yaw_scale_error"
    ]
    if sequential:
        keys += [
            "xy_seq_error",
            # "xy_gps_seq_error",
            "yaw_seq_error",
            # "yaw_gps_seq_error",
        ]
    for k in keys:
        if k not in metrics:
            logger.warning("Key %s not in metrics.", k)
            continue
        rec = metrics[k].recall(thresholds).double().numpy().round(2).tolist()
        logger.info("Recall %s: %s at %s m/°", k, rec, thresholds)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--split", type=str, default="val", choices=["val"])
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--particle", action="store_true")
    parser.add_argument("--ekf", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--num", type=int)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_args()
    # print("args.ekf %s"%args.ekf)
    cfg = OmegaConf.from_cli(args.dotlist)
    run(
        args.checkpoint,
        args.dataset,
        args.split,
        args.experiment,
        cfg,
        args.sequential,
        args.particle,
        args.ekf,
        output_dir=args.output_dir,
        num=args.num,
    )
