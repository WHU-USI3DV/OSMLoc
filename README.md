<h2> 
<a href="https://whu-usi3dv.github.io/OSMLoc/" target="_blank">OSMLoc: Single Image-Based Visual Localization in OpenStreetMap with Fused Geometric and Semantic Guidance</a>
</h2>

This is the official PyTorch implementation of the following publication:

> **OSMLoc: Single Image-Based Visual Localization in OpenStreetMap with Fused Geometric and Semantic Guidance**<br/>
> [Youqi Liao*](https://martin-liao.github.io/),[Xieyuanli Chen*](https://xieyuanli-chen.com/),[Shuhao Kang](https://kang-1-2-3.github.io/) [Jianping Li**](https://kafeiyin00.github.io/),  [Zhen Dong](https://dongzhenwhu.github.io/index.html), [Hongchao Fan](https://scholar.google.com/citations?user=VeH-I7AAAAAJ), [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm)<br/>
> *Technical Report*<br/>
> **Paper** | [**Arxiv**](https://arxiv.org/abs/2411.08665) | [**Project-page**](https://whu-usi3dv.github.io/OSMLoc/) | [**Video**](https://youtu.be/rjKRLYfCG-g)


## 🔭 Introduction
<p align="center">
<strong>TL;DR: OSMLoc is an image-to-OpenstreetMap (I2O) visual localization framework with geometric and semantic guidance.</strong>
</p>
<img src="./figs/motivation_repo.png" alt="Motivation" style="zoom:100%; display: block; margin-left: auto; margin-right: auto; max-width: 100%;">

<p align="justify">
<strong>Abstract:</strong> OpenStreetMap (OSM), a rich and versatile source of volunteered geographic information (VGI), facilitates human self-localization and scene understanding by integrating nearby visual observations with vectorized map data. However, the disparity in modalities and perspectives poses a major challenge for effectively matching camera imagery with compact map representations, thereby limiting the full potential of VGI data in real-world localization applications. Inspired by the fact that the human brain relies on the fusion of geometric and semantic understanding for spatial localization tasks, we propose the OSMLoc in this paper. OSMLoc is a brain-inspired visual localization approach based on first-person-view images against the OSM maps. It integrates semantic and geometric guidance to significantly improve accuracy, robustness, and generalization capability. First, we equip the OSMLoc with the visual foundational model to extract powerful image features. Second, a geometry-guided depth distribution adapter is proposed to bridge the monocular depth estimation and camera-to-BEV transform. Thirdly, the semantic embeddings from the OSM data are utilized as auxiliary guidance for image-to-OSM feature matching. To validate the proposed OSMLoc, we collect a worldwide cross-area and cross-condition (CC) benchmark for extensive evaluation. Experiments on the MGL dataset, CC validation benchmark, and KITTI dataset have demonstrated the superiority of our method. 
</p>

## 🆕 News
- 2024-11-20: [Project page](https://whu-usi3dv.github.io/OSMLoc/) (with introduction video) is available!🎉 
- 2024-11-20: The code, pre-trained models, and validation benchmark will be available upon acceptance of the paper.
- 2025-08-28: Code, pre-trained models, and validation benchmark are available now!

## 💻 Installation
An example for ```CUDA=11.8``` and ```pytorch=2.0.1```:
```
pip3 install -r requirements.txt
```
We will provide a Docker image for quick start in the near future.

## 🚅 Usage

### MGL dataset 
You could download the MGL dataset [here](https://pan.baidu.com/s/1dsP_Wl_m9MpLHimp8qch0A?pwd=22gj)(passwd: 22gj)(>120G). As the MGL dataset is extremely large, please reserve enough space.

### CC Benchmark
You could download the CC benchmark [here](https://pan.baidu.com/s/1yUI8SY-RWsz30LKMFkgseA?pwd=26ms)(passwd: 26ms).

### KITTI dataset

Download and prepare the dataset to `./datasets/kitti/`:

```
python -m maploc.data.kitti.prepare
```

### Evaluation
For the MGL dataset, CC benchmark and KITTI dataset, we provide pre-trained models on [googledrive](https://drive.google.com/drive/folders/142HxByjjPwY9Cbwyj6kEiut_sHxs9GSD?usp=sharing) and [Baidu Disk](https://pan.baidu.com/s/1CPOasGzZaCv95ppYyZgj_g?pwd=bu7b) (passwd:bu7b).
Please download the weights of OSMLoc from webdrive and put them in a folder like ```checkpoints```.

Example 1: evaluate ```OSMLOC``` on the MGL dataset

```
python -m maploc.evaluation.mapillary --checkpoint="./checkpoints/osmloc_small.ckpt" --dataset=mgl model.num_rotations=256 model.image_encoder.val=True
```
Above operation calculates the per-frame localization error, and the evaluation results on the MGL dataset should be close to:
```
PR@1m=14.04, PR@3m=44.00, PR@5m=57.31;
OR@1m=21.84, OR@3m=53.38, OR@5m=68.41;
```

Example 2: evaluate ```OSMLOC``` on the munich part of CC benchmark

```
python -m maploc.evaluation.mapillary --checkpoint="./checkpoints/osmloc_small.ckpt" --dataset=munich model.num_rotations=256 model.image_encoder.val=True
```
Above operation calculates the per-frame localization error, and the evaluation results on the MGL dataset should be close to:
```
PR@1m=3.20, PR@3m=19.53, PR@5m=38.27;
OR@1m=15.07, OR@3m=39.19, OR@5m=55.94;
```

Example 3: evaluate ```OSMLOC``` on the KITTI dataset

```
python -m maploc.evaluation.kitti kitti model.num_rotations=256  model.image_encoder.val=True
```
Above operation calculates the per-frame localization error, and the evaluation results on the MGL dataset should be close to:
```
LatR@1m=65.75, LatR@3m=93.79, Lat@5m=96.74;
LonR@1m=24.50, LonR@3m=59.53, LonR@5m=72.67;
OR@1m=37.43, OR@3m=79.69, OR@5m=91.67;
```

Example 4: evaluate ```OSMLOC``` on the MGL dataset with sequential frames

```
python -m maploc.evaluation.mapillary --checkpoint="./checkpoints/osmloc_small.ckpt" --dataset=mgl model.num_rotations=256  model.image_encoder.val=True --sequential --particle
```
The above operation employs the Monte Carlo Localization (MCL) framework with a simulated motion model to estimate per-frame localization, and the evaluation on the MGL dataset is expected to yield results close to:
```
PR@1m=14.04, PR@3m=44.00, PR@5m=57.31;
OR@1m=21.84, OR@3m=53.38, OR@5m=68.41;
```


### Training
Example: train ```OSMLoc``` on the MGL dataset
```
python -m maploc.train experiment.name=osmloc
```


## 💡 Citation
If you find this repo helpful, please give us a star~.Please consider citing OSMLoc if this program benefits your project.
```
@article{liao2024osmloc,
  title={OSMLoc: Single Image-Based Visual Localization in OpenStreetMap with Geometric and Semantic Guidances},
  author={Liao, Youqi and Chen, Xieyuanli and Kang, Shuhao and Li, Jianping and Dong, Zhen and Fan, Hongchao and Yang, Bisheng},
  journal={arXiv preprint arXiv:2411.08665},
  year={2024}
}
```

## 🔗 Related Projects
We sincerely thank the excellent projects:
- [OrienterNet](https://github.com/facebookresearch/OrienterNet) for pioneering I2O visual localization approach;
- [Retrieval](https://github.com/YujiaoShi/HighlyAccurate) for extensive evaluation;
- [Refine](https://github.com/tudelft-iv/CrossViewMetricLocalization) for extensive evaluation;
- [Range-MCL](https://github.com/PRBonn/range-mcl) for Monto Carlo localization framework;
- [Freereg](https://github.com/WHU-USI3DV/FreeReg) for excellent template; 