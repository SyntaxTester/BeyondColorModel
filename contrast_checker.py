from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image


def _linearize(value: float) -> float:
    v = value / 255.0
    if v <= 0.04045:
        return v / 12.92
    return ((v + 0.055) / 1.055) ** 2.4


def relative_luminance(r, g, b):
    return 0.2126*_linearize(r) + 0.7152*_linearize(g) + 0.0722*_linearize(b)


def contrast_ratio(fg, bg):
    l1 = relative_luminance(*fg)
    l2 = relative_luminance(*bg)
    return (max(l1,l2)+0.05) / (min(l1,l2)+0.05)


@dataclass
class ImageContrastAudit:
    min_ratio: float
    max_ratio: float
    mean_ratio: float
    low_contrast_fraction: float
    violations: list
    overall_pass: bool


def audit_image_contrast(image, sample_count=500, threshold=4.5):
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    h, w = arr.shape[:2]
    rng = np.random.default_rng(42)
    ratios = []
    violations = []
    for _ in range(sample_count):
        y = int(rng.integers(0, h-1))
        x = int(rng.integers(0, w-1))
        px1 = tuple(int(v) for v in arr[y, x, :3])
        px2 = tuple(int(v) for v in arr[y, x+1, :3])
        r = contrast_ratio(px1, px2)
        ratios.append(r)
        if r < threshold:
            violations.append({"position":(x,y),"ratio":round(r,2)})
    ratios = np.array(ratios)
    low = float((ratios < threshold).mean())
    return ImageContrastAudit(
        min_ratio=float(ratios.min()),
        max_ratio=float(ratios.max()),
        mean_ratio=float(ratios.mean()),
        low_contrast_fraction=low,
        violations=violations[:20],
        overall_pass=bool(low < 0.3),
    )


def check_contrast(fg, bg):
    return contrast_ratio(_parse(fg), _parse(bg))


def _parse(c):
    if isinstance(c, (tuple,list)):
        return tuple(int(x) for x in c[:3])
    h = c.strip().lstrip("#")
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))