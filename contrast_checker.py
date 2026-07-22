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
    edges_checked: int


def _relative_luminance_array(arr: np.ndarray) -> np.ndarray:
    channels = arr.astype(np.float32) / 255.0
    linear = np.where(
        channels <= 0.04045,
        channels / 12.92,
        ((channels + 0.055) / 1.055) ** 2.4,
    )
    return (
        0.2126 * linear[:, :, 0]
        + 0.7152 * linear[:, :, 1]
        + 0.0722 * linear[:, :, 2]
    )


def _contrast_ratio_array(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lighter = np.maximum(left, right)
    darker = np.minimum(left, right)
    return (lighter + 0.05) / (darker + 0.05)


def _regular_sample(values: np.ndarray, sample_count: int | None) -> np.ndarray:
    if sample_count is None or len(values) <= sample_count:
        return values
    indexes = np.linspace(0, len(values) - 1, sample_count, dtype=np.intp)
    return values[indexes]


def audit_image_contrast(
    image,
    sample_count: int | None = 5000,
    threshold: float = 3.0,
    edge_delta: int = 12,
):
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return ImageContrastAudit(
            min_ratio=1.0,
            max_ratio=1.0,
            mean_ratio=1.0,
            low_contrast_fraction=0.0,
            violations=[],
            overall_pass=True,
            edges_checked=0,
        )

    luminance = _relative_luminance_array(arr)
    horizontal_delta = np.abs(arr[:, 1:].astype(np.int16) - arr[:, :-1].astype(np.int16)).max(axis=2)
    vertical_delta = np.abs(arr[1:, :].astype(np.int16) - arr[:-1, :].astype(np.int16)).max(axis=2)
    horizontal_edges = horizontal_delta >= edge_delta
    vertical_edges = vertical_delta >= edge_delta

    horizontal_ratios = _contrast_ratio_array(luminance[:, 1:], luminance[:, :-1])[horizontal_edges]
    vertical_ratios = _contrast_ratio_array(luminance[1:, :], luminance[:-1, :])[vertical_edges]
    ratios = np.concatenate([horizontal_ratios, vertical_ratios])
    ratios = _regular_sample(ratios, sample_count)

    if len(ratios) == 0:
        return ImageContrastAudit(
            min_ratio=1.0,
            max_ratio=1.0,
            mean_ratio=1.0,
            low_contrast_fraction=0.0,
            violations=[],
            overall_pass=True,
            edges_checked=0,
        )

    violations = []
    horizontal_ratio_grid = _contrast_ratio_array(luminance[:, 1:], luminance[:, :-1])
    vertical_ratio_grid = _contrast_ratio_array(luminance[1:, :], luminance[:-1, :])
    for y, x in zip(*np.where(horizontal_edges & (horizontal_ratio_grid < threshold))):
        violations.append({"position": (int(x), int(y)), "ratio": round(float(horizontal_ratio_grid[y, x]), 2)})
        if len(violations) >= 20:
            break
    if len(violations) < 20:
        for y, x in zip(*np.where(vertical_edges & (vertical_ratio_grid < threshold))):
            violations.append({"position": (int(x), int(y)), "ratio": round(float(vertical_ratio_grid[y, x]), 2)})
            if len(violations) >= 20:
                break

    low = float((ratios < threshold).mean())
    return ImageContrastAudit(
        min_ratio=float(ratios.min()),
        max_ratio=float(ratios.max()),
        mean_ratio=float(ratios.mean()),
        low_contrast_fraction=low,
        violations=violations,
        overall_pass=bool(float(ratios.mean()) >= threshold),
        edges_checked=int(len(horizontal_ratios) + len(vertical_ratios)),
    )


def check_contrast(fg, bg):
    return contrast_ratio(_parse(fg), _parse(bg))


def _parse(c):
    if isinstance(c, (tuple,list)):
        return tuple(int(x) for x in c[:3])
    h = c.strip().lstrip("#")
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
