"""Datasets: unpaired low-light folder, paired (LOL-style low/high), or synthetic degradation of
normally exposed images (useful for pre-training / when no low-light data is available)."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .classical import illumination_estimate

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(d):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXT)


def read_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def synth_low_light(img, rng):
    """Physically-motivated degradation: exposure drop + gamma + non-uniform illumination
    (random linear gradient / vignette) + signal-dependent shot noise + read noise + colour cast."""
    h, w = img.shape[:2]
    lin = np.power(img, 2.2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    vig = 1.0 - rng.uniform(0.0, 0.8) * np.clip(np.hypot(xx - cx, yy - cy) / np.hypot(w, h), 0, 1)
    scale = np.exp(rng.uniform(np.log(0.01), np.log(0.25)))
    cast = 1 + rng.uniform(-0.12, 0.12, 3).astype(np.float32)
    dark = lin * scale * vig[..., None] * cast
    peak = rng.uniform(30, 400)
    noisy = rng.poisson(dark * peak).astype(np.float32) / peak + rng.normal(0, rng.uniform(0.001, 0.01), dark.shape)
    return np.clip(np.power(np.clip(noisy, 0, 1), 1 / 2.2), 0, 1).astype(np.float32)


class LowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir=None, crop=256, with_prior=True, synthetic=False, seed=0):
        self.low = list_images(low_dir)
        self.high = {p.name: p for p in list_images(high_dir)} if high_dir else {}
        self.crop, self.prior, self.synth = crop, with_prior, synthetic
        self.rng = np.random.default_rng(seed)
        assert self.low, f"no images in {low_dir}"

    def __len__(self):
        return len(self.low)

    def _crop(self, *imgs):
        h, w = imgs[0].shape[:2]
        c = self.crop
        if min(h, w) < c:
            s = c / min(h, w) + 1e-3
            imgs = [cv2.resize(i, (int(w * s) + 1, int(h * s) + 1)) for i in imgs]
            h, w = imgs[0].shape[:2]
        y, x = self.rng.integers(0, h - c + 1), self.rng.integers(0, w - c + 1)
        out = [i[y:y + c, x:x + c] for i in imgs]
        if self.rng.random() < 0.5:
            out = [i[:, ::-1] for i in out]
        return [np.ascontiguousarray(i) for i in out]

    def __getitem__(self, i):
        p = self.low[i]
        img = read_rgb(p)
        gt = None
        if self.synth:
            (img,) = self._crop(img)
            gt, img = img, synth_low_light(img, self.rng)
        elif p.name in self.high:
            img, gt = self._crop(img, read_rgb(self.high[p.name]))
        else:
            (img,) = self._crop(img)
        item = {"low": torch.from_numpy(img.transpose(2, 0, 1)).float()}
        if self.prior:
            item["prior"] = torch.from_numpy(illumination_estimate(img)[None]).float()
        if gt is not None:
            item["gt"] = torch.from_numpy(gt.transpose(2, 0, 1)).float()
        return item
