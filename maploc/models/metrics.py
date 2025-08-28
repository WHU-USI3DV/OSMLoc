import torch
import torchmetrics
from torchmetrics.utilities.data import dim_zero_cat

from .utils import deg2rad, rotmat2d


def location_error(uv, uv_gt, ppm=1,topk = False):
    error = torch.norm(uv - uv_gt.to(uv), dim=-1) / ppm
    if topk:
        error = error.min(dim = 1).values
    return error


def angle_error(t, t_gt, topk: bool = False):
    error = torch.abs(t % 360 - t_gt.to(t) % 360)
    error = (torch.minimum(error, 360 - error))
    if topk:
        error = error.min(dim = 1).values
    return error


class Location2DRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, pixel_per_meter, key="uv_max", topk = False, *args, **kwargs):
        self.threshold = threshold
        self.ppm = pixel_per_meter
        self.key = key
        self.topk = topk
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        error = location_error(pred[self.key], data["uv"], self.ppm, self.topk)
        super().update((error < self.threshold).float())


class AngleRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, key="yaw_max", topk = False, *args, **kwargs):
        self.threshold = threshold
        self.key = key
        self.topk = topk
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        error = angle_error(pred[self.key], data["roll_pitch_yaw"][..., -1], self.topk)
        super().update((error < self.threshold).float())


class MeanMetricWithRecall(torchmetrics.Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("value", default=[], dist_reduce_fx="cat")

    def compute(self):
        return dim_zero_cat(self.value).mean(0)

    def get_errors(self):
        return dim_zero_cat(self.value)

    def recall(self, thresholds):
        error = self.get_errors()
        thresholds = error.new_tensor(thresholds)
        return (error.unsqueeze(-1) < thresholds).float().mean(0) * 100


class AngleError(MeanMetricWithRecall):
    def __init__(self, key,topk = False):
        super().__init__()
        self.key = key
        self.topk = topk

    def update(self, pred, data):
        value = angle_error(pred[self.key], data["roll_pitch_yaw"][..., -1], topk = self.topk)
        if value.numel():
            self.value.append(value)


class Location2DError(MeanMetricWithRecall):
    def __init__(self, key, pixel_per_meter, topk = False):
        super().__init__()
        self.key = key
        self.ppm = pixel_per_meter
        self.topk = topk

    def update(self, pred, data):
        value = location_error(pred[self.key], data["uv"], self.ppm,self.topk)
        if value.numel():
            self.value.append(value)


class LateralLongitudinalError(MeanMetricWithRecall):
    def __init__(self, pixel_per_meter, key="uv_max"):
        super().__init__()
        self.ppm = pixel_per_meter
        self.key = key

    def update(self, pred, data):
        yaw = deg2rad(data["roll_pitch_yaw"][..., -1])
        shift = (pred[self.key] - data["uv"]) * yaw.new_tensor([-1, 1])
        shift = (rotmat2d(yaw) @ shift.unsqueeze(-1)).squeeze(-1)
        error = torch.abs(shift) / self.ppm
        value = error.view(-1, 2)
        if value.numel():
            self.value.append(value)

class rmse(torchmetrics.MeanMetric):
    def __init__(self, pixel_per_meter, key="uv_max"):
        super().__init__()
        self.ppm = pixel_per_meter
        self.key = key

    def update(self, pred, data):
        # print(pred[self.key].shape, data["uv"].shape)
        error = torch.sqrt((torch.sum(pred[self.key]/self.ppm - data["uv"].to(pred[self.key])/self.ppm)**2))
        super().update(error.float())