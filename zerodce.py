"""Zero-DCE++ (Li et al., TPAMI 2021) in PyTorch, extended with an optional 4th input channel
carrying the classical illumination prior (hybrid mode). Includes the non-reference losses
and an optional supervised L1+SSIM loss for fine-tuning on paired data (LOL / SICE)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)
        self.pw = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class DCENet(nn.Module):
    """Predicts 8 curve-parameter maps (24 channels). scale>1 estimates them on a downscaled
    copy (Zero-DCE++ speed-up) and upsamples before applying curves at full resolution."""

    def __init__(self, in_ch=3, nf=32, n_iter=8):
        super().__init__()
        self.in_ch, self.n_iter = in_ch, n_iter
        self.c1, self.c2, self.c3, self.c4 = DSConv(in_ch, nf), DSConv(nf, nf), DSConv(nf, nf), DSConv(nf, nf)
        self.c5, self.c6 = DSConv(nf * 2, nf), DSConv(nf * 2, nf)
        self.c7 = DSConv(nf * 2, 3 * n_iter)

    def curve_maps(self, x):
        x1 = F.relu(self.c1(x)); x2 = F.relu(self.c2(x1)); x3 = F.relu(self.c3(x2)); x4 = F.relu(self.c4(x3))
        x5 = F.relu(self.c5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.c6(torch.cat([x2, x5], 1)))
        return torch.tanh(self.c7(torch.cat([x1, x6], 1)))

    def forward(self, x, scale=1, iters=None):
        """x: Bx(in_ch)xHxW; first 3 channels are RGB. Returns (enhanced, curve_maps)."""
        iters = iters or self.n_iter
        rgb = x[:, :3]
        xin = x if scale == 1 else F.interpolate(x, scale_factor=1 / scale, mode="bilinear", align_corners=True)
        A = self.curve_maps(xin)
        if scale != 1:
            A = F.interpolate(A, size=rgb.shape[2:], mode="bilinear", align_corners=True)
        out = rgb
        for i in range(iters):
            a = A[:, 3 * i:3 * i + 3]
            out = out + a * (out ** 2 - out)
        return out, A


# ----------------------------- losses (non-reference) -----------------------------
class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = {"l": [[0, 0, 0], [-1, 1, 0], [0, 0, 0]], "r": [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
             "u": [[0, -1, 0], [0, 1, 0], [0, 0, 0]], "d": [[0, 0, 0], [0, 1, 0], [0, -1, 0]]}
        self.register_buffer("k", torch.tensor([v for v in k.values()], dtype=torch.float32).unsqueeze(1))

    def forward(self, enh, org):
        e, o = enh.mean(1, keepdim=True), org.mean(1, keepdim=True)
        e, o = F.avg_pool2d(e, 4), F.avg_pool2d(o, 4)
        return ((F.conv2d(o, self.k, padding=1) - F.conv2d(e, self.k, padding=1)) ** 2).mean()


def exposure_loss(enh, E=0.6, patch=16):
    return ((F.avg_pool2d(enh.mean(1, keepdim=True), patch) - E) ** 2).mean()


def color_constancy_loss(enh):
    m = enh.mean((2, 3))
    r, g, b = m[:, 0], m[:, 1], m[:, 2]
    return (((r - g) ** 2 + (r - b) ** 2 + (b - g) ** 2) ** 0.5).mean()


def illumination_smoothness_loss(A):
    dx = (A[:, :, :, 1:] - A[:, :, :, :-1]).pow(2).mean()
    dy = (A[:, :, 1:, :] - A[:, :, :-1, :]).pow(2).mean()
    return dx + dy


def ssim_loss(x, y, win=7):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = F.avg_pool2d(x, win, 1, win // 2), F.avg_pool2d(y, win, 1, win // 2)
    sx = F.avg_pool2d(x * x, win, 1, win // 2) - mx ** 2
    sy = F.avg_pool2d(y * y, win, 1, win // 2) - my ** 2
    sxy = F.avg_pool2d(x * y, win, 1, win // 2) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sx + sy + c2))
    return 1 - s.mean()


class ZeroDCELoss(nn.Module):
    def __init__(self, w=(1.0, 10.0, 5.0, 200.0, 1.0, 1.0)):
        super().__init__()
        self.w, self.spa = w, SpatialConsistencyLoss()

    def forward(self, enh, org, A, gt=None):
        parts = {
            "spa": self.w[0] * self.spa(enh, org),
            "exp": self.w[1] * exposure_loss(enh),
            "col": self.w[2] * color_constancy_loss(enh),
            "tv": self.w[3] * illumination_smoothness_loss(A),
        }
        if gt is not None:  # optional supervised term for paired fine-tuning
            parts["l1"] = self.w[4] * F.l1_loss(enh, gt)
            parts["ssim"] = self.w[5] * ssim_loss(enh, gt)
        return sum(parts.values()), parts
