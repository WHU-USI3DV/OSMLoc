from pathlib import Path
import torch
import yaml
from torchmetrics import MetricCollection
from omegaconf import OmegaConf as OC
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pytorch_lightning import seed_everything
import hydra
import os.path as osp
import itertools

# import maploc
# import sys
# sys.path.append('..')
import maploc
from maploc.data import MapillaryDataModule
from maploc.data.torch import unbatch_to_device
from maploc.module import GenericModule
from maploc.models.metrics import Location2DError
# from maploc.evaluation.run import resolve_checkpoint_path

torch.set_grad_enabled(False);
plt.rcParams.update({'figure.max_open_warning': 0})

conf = OC.load('maploc/conf/data/mapillary_munich.yaml')
conf = OC.merge(conf, OC.create(yaml.full_load("""
data_dir: "/data/OSM_Image/OrienterNet_reg/datasets/MGL"
loading:
    val: {batch_size: 1, num_workers: 0}
    train: ${.val}
add_map_mask: true
return_gps: true
""")))
OC.resolve(conf)
dataset = MapillaryDataModule(conf)
dataset.prepare_data()
dataset.setup()
sampler = None


path = "./checkpoints/osmloc_small.ckpt"
print(path)
cfg = {'model': {"num_rotations": 256, "apply_map_prior": True}}
model = GenericModule.load_from_checkpoint(
     path, strict=True, cfg=cfg)

model = model.eval().cuda()
assert model.cfg.data.resize_image == dataset.cfg.resize_image

def fps_params_flops(model, input):
    import time
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    iterations = None
    
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing (network inference)=========')
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(latency, ">>> latency. ")
    print(FPS, ">>> fps. ")
    

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis,flop_count_table

    flops = FlopCountAnalysis(model, input)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, input)

    print(f"module flops : {flop_count_table(flops)}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")


seed_everything(25) # best = 25
loader = dataset.dataloader("val", shuffle=sampler is None, sampler=sampler)
metrics = MetricCollection(model.model.metrics()).to(model.device)
metrics["xy_gps_error"] = Location2DError("uv_gps", model.cfg.model.pixel_per_meter)
data = next(itertools.islice(enumerate(loader),0,1),None)[1]
# pred = data = batch_ = None    
data["c"] = data["camera"].c
data["f"] = data["camera"].f
data.pop("camera")
data.pop("canvas")
data["wrapper"] = True
data = model.transfer_batch_to_device(data, model.device, 0)
fps_params_flops(model, data)