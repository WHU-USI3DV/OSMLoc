import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from ..feature_extractor_v2 import FPN, FPN_Bottleneck, FeatureEncoder

from .blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        # out = self.scratch.output_conv2[:2](out)
        depth = self.scratch.output_conv2(out) # remove relu layer
        
        return {"layer_1":layer_1,"layer_2":layer_2,"layer_3":layer_3,"layer_4":layer_4}, depth
        
        
class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256,out_dim = 128, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True, size = [518,518], ckpt = None,shallow_encoder = None, val = False):
        super().__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']
        print(encoder)
        
        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.pretrained = torch.hub.load('maploc/models/dinov2', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))

        if val == False:
            if ckpt is not None:
                ckpt = torch.load(ckpt)
                model_dict = self.state_dict()
                for name, params in ckpt.items():
                    if name in model_dict.keys():
                        model_dict[name] = params
                self.load_state_dict(model_dict)

        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

        dim_2 = self.pretrained.blocks[2].attn.qkv.in_features
        dim_5 = self.pretrained.blocks[2].attn.qkv.in_features
        dim_8 = self.pretrained.blocks[2].attn.qkv.in_features
        dim_11 = self.pretrained.blocks[2].attn.qkv.in_features
        
        self.fpn_bottleneck = FPN_Bottleneck([dim_2,dim_5,dim_8,dim_11],out_dim,align_corners = True)
        
        self.size = size


        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.register_buffer("mean_", torch.tensor(mean), persistent=False)
        self.register_buffer("std_", torch.tensor(std), persistent=False)
        
    def forward(self, x):

        h, w = x.shape[-2:]

        # x = torch.cat([x,x.flip(-2)],-2)
        
        # h_new, w_new = x.shape[-2:]
        
        t,b = self.size[0] - h, 0
        l,r = self.size[1] - w, 0
        x = F.pad(x,[l,r,t,b], mode="replicate")

        x = (x - self.mean_[:, None, None]) / self.std_[:, None, None]

        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = self.size[0] // 14, self.size[1] // 14

        feats, depth = self.depth_head(features, patch_h, patch_w)

        feats = self.pretrained.get_intermediate_layers(x,[2,5,8,11], return_class_token = False,reshape = True)
        feats = {"layer_0": feats[0],
                  "layer_1": feats[1],
                  "layer_2": feats[2],
                  "layer_3": feats[3]}

        # feat_loc = self.fpn(feats)
        feat_loc = self.fpn_bottleneck(feats)
        feat_loc = F.interpolate(feat_loc,(int(patch_h * 7), int(patch_w * 7)), mode="bilinear", align_corners=True)

        # out = self.depth_conv1(feat_depth)
        # out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        # out = self.scratch.output_conv2[:2](out)
        # depth = self.depth_conv2(out) # remove relu layer

        # disparity = disparity[...,t:,l:]
        feat_loc = feat_loc[...,t // 2:t // 2 + h // 2,l // 2: l // 2 + w // 2]
        depth = depth[...,t:,l:]

        out_dict = {"features": feat_loc, "depth": depth}

        return out_dict


class DepthAnything(DPT_DINOv2):
    def __init__(self, config):
        super().__init__(**config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        default="vits",
        type=str,
        choices=["vits", "vitb", "vitl"],
    )
    args = parser.parse_args()
    
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_{:}14".format(args.encoder))
    
    print(model)
    