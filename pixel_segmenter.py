from __future__ import annotations
import numpy as np
from PIL import Image


_N_SECTORS = 16
_ANGLE_LUT = [
    0, 90, 22.5, 112.5, 45, 135, 67.5, 157.5,
    11.25, 101.25, 33.75, 123.75, 56.25, 146.25, 78.75, 168.75,
]
_PATTERN_FAMILIES = ["lines", "dots", "grid", "dashed", "waves", "zigzag", "crosshatch", "bricks"]


def _smooth_shadows(arr_rgb: np.ndarray) -> np.ndarray:
    
    import cv2
    median = cv2.medianBlur(arr_rgb, 5)
    return cv2.bilateralFilter(median, d=15, sigmaColor=80, sigmaSpace=80)


def _get_present_sectors(hue_smooth: np.ndarray, valid: np.ndarray) -> set[int]:
    sector = np.floor(hue_smooth / (360.0 / _N_SECTORS)).astype(np.int32) % _N_SECTORS
    present = set()
    total_valid = valid.sum()
    if total_valid == 0:
        return present
    for s in range(_N_SECTORS):
        count = ((sector == s) & valid).sum()
        if count > total_valid * 0.008:
            present.add(s)
    return present


def _assign_pattern_families(present_sectors: set[int]) -> dict[int, str]:

    n_families = len(_PATTERN_FAMILIES)
    sectors_sorted = sorted(present_sectors)
    return {s: _PATTERN_FAMILIES[i % n_families] for i, s in enumerate(sectors_sorted)}


def _draw_pattern(
    family: str,
    xx: np.ndarray, yy: np.ndarray,
    angle_rad: float,
    bold: np.ndarray,
    spacing: float = 12.0,
) -> np.ndarray:
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    proj = xx * cos_a + yy * sin_a
    perp = -xx * sin_a + yy * cos_a

    width_ratio = np.where(bold, 0.40, 0.20)

    if family == "lines":
        return (proj % spacing) < (spacing * width_ratio)

    if family == "dashed":
        line = (proj % spacing) < (spacing * width_ratio)
        dash_period = np.where(bold, spacing * 1.2, spacing * 2.2)
        dash = (perp % dash_period) < (dash_period * 0.5)
        return line & dash

    if family == "grid":
        line1 = (proj % spacing) < (spacing * width_ratio * 0.7)
        line2 = (perp % spacing) < (spacing * width_ratio * 0.7)
        return line1 | line2

    if family == "dots":
        gx = (proj % spacing)
        gy = (perp % spacing)
        cx = cy = spacing / 2
        r = spacing * np.where(bold, 0.30, 0.16)
        return (gx - cx) ** 2 + (gy - cy) ** 2 <= r ** 2

    if family == "waves":
        wave = np.sin(proj / spacing * 2 * np.pi) * (spacing * 0.3)
        return np.abs((perp % spacing) - spacing / 2 - wave) < (spacing * width_ratio)

    if family == "zigzag":
        tri = np.abs((proj % spacing) - spacing / 2)
        return np.abs((perp % spacing) - tri) < (spacing * width_ratio)

    if family == "crosshatch":
        d1 = ((proj + perp) % spacing) < (spacing * width_ratio * 0.7)
        d2 = ((proj - perp) % spacing) < (spacing * width_ratio * 0.7)
        return d1 | d2

    if family == "bricks":
        row = np.floor(perp / spacing)
        offset = (row % 2) * (spacing / 2)
        hline = (perp % spacing) < (spacing * width_ratio * 0.6)
        vline = ((proj + offset) % spacing) < (spacing * width_ratio * 0.6)
        return hline | vline

    return (proj % spacing) < (spacing * width_ratio)


def _procedural_pattern_mask(
    h_img: int, w_img: int,
    hue: np.ndarray, sat: np.ndarray, valid: np.ndarray,
) -> np.ndarray:
    import cv2

    hue_scaled = (hue / 360.0 * 255.0).astype(np.uint8)
    hue_smooth = cv2.medianBlur(hue_scaled, 15).astype(np.float32) / 255.0 * 360.0

    sat_smooth = cv2.medianBlur((sat * 255).astype(np.uint8), 9).astype(np.float32) / 255.0
    bold = sat_smooth > 0.5

    sector = np.floor(hue_smooth / (360.0 / _N_SECTORS)).astype(np.int32) % _N_SECTORS
    present = _get_present_sectors(hue_smooth, valid)
    family_map = _assign_pattern_families(present)

    yy, xx = np.mgrid[0:h_img, 0:w_img].astype(np.float32)
    result = np.zeros((h_img, w_img), dtype=bool)

    for s in present:
        sector_mask = (sector == s)
        if not sector_mask.any():
            continue
        angle_rad = np.deg2rad(_ANGLE_LUT[s])
        family = family_map.get(s, "lines")
        pattern = _draw_pattern(family, xx, yy, angle_rad, bold)
        result |= (sector_mask & pattern)

    return result


def _redraw_small_elements(
    arr: np.ndarray,
    valid: np.ndarray,
    result: np.ndarray, alpha: float,
    family_map: dict, sector_rgb: dict,
) -> np.ndarray:

    import cv2

    h_img, w_img = arr.shape[:2]
    valid_u8 = valid.astype(np.uint8)
    n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(valid_u8)

    MARKER = 18
    yy, xx = np.mgrid[0:h_img, 0:w_img].astype(np.float32)

    if not sector_rgb:
        return result

    for i in range(1, n_lab):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if not (4 <= cw <= 14 and 4 <= ch <= 14):
            continue
        if area < 12:
            continue
        aspect = cw / ch if ch > 0 else 0
        if not (0.5 < aspect < 2.0):
            continue

        elem_mask = (labels == i)
        elem_rgb = arr[elem_mask].mean(axis=0)

        elem_hsv = cv2.cvtColor(
            np.array([[elem_rgb.astype(np.uint8)]], dtype=np.uint8), cv2.COLOR_RGB2HSV
        )[0][0]
        elem_hue = float(elem_hsv[0]) * 2
        own_sector = int(elem_hue // (360.0 / _N_SECTORS)) % _N_SECTORS

        if own_sector in sector_rgb:
            best_sector = own_sector
        else:
            best_sector = min(
                sector_rgb.keys(),
                key=lambda s: np.linalg.norm(sector_rgb[s] - elem_rgb)
            )
        family = family_map.get(best_sector, "lines")
        angle_rad = np.deg2rad(_ANGLE_LUT[best_sector])

        cx = int(centroids[i][0])
        cy = int(centroids[i][1])
        x0 = max(0, cx - MARKER // 2)
        y0 = max(0, cy - MARKER // 2)
        x1 = min(w_img, x0 + MARKER)
        y1 = min(h_img, y0 + MARKER)

        result[y0:y1, x0:x1] = elem_rgb.astype(np.float32)

        rep = sector_rgb[best_sector].astype(np.uint8)
        rep_hsv = cv2.cvtColor(np.array([[rep]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]
        is_bold = (rep_hsv[1] / 255.0) > 0.5
        bold = np.full((h_img, w_img), is_bold, dtype=bool)

        pattern = _draw_pattern(family, xx, yy, angle_rad, bold, spacing=6.0)
        block = np.zeros((h_img, w_img), dtype=bool)
        block[y0:y1, x0:x1] = True
        draw_here = block & pattern
        result[draw_here] = result[draw_here] * (1 - alpha)

    return result


def _procedural_pattern_mask_with_families(
    h_img: int, w_img: int,
    hue: np.ndarray, sat: np.ndarray, valid: np.ndarray,
    arr_orig: np.ndarray,
):
    import cv2

    hue_scaled = (hue / 360.0 * 255.0).astype(np.uint8)
    hue_smooth = cv2.medianBlur(hue_scaled, 15).astype(np.float32) / 255.0 * 360.0

    sat_smooth = cv2.medianBlur((sat * 255).astype(np.uint8), 9).astype(np.float32) / 255.0
    bold = sat_smooth > 0.5

    sector = np.floor(hue_smooth / (360.0 / _N_SECTORS)).astype(np.int32) % _N_SECTORS
    present = _get_present_sectors(hue_smooth, valid)
    family_map = _assign_pattern_families(present)

    yy, xx = np.mgrid[0:h_img, 0:w_img].astype(np.float32)
    result = np.zeros((h_img, w_img), dtype=bool)

    sector_rgb: dict[int, np.ndarray] = {}
    for s in present:
        sector_mask = (sector == s) & valid
        if not sector_mask.any():
            continue
        comp_u8 = sector_mask.astype(np.uint8)
        n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(comp_u8)
        big_pixels = []
        for c in range(1, n_c):
            if st_c[c, cv2.CC_STAT_AREA] >= 300:  # крупная область = данные
                big_pixels.append(lab_c == c)
        if big_pixels:
            big_mask_s = np.any(big_pixels, axis=0)
            sector_rgb[s] = arr_orig[big_mask_s].mean(axis=0)

    for s in present:
        sector_mask = (sector == s)
        if not sector_mask.any():
            continue
        angle_rad = np.deg2rad(_ANGLE_LUT[s])
        family = family_map.get(s, "lines")
        pattern = _draw_pattern(family, xx, yy, angle_rad, bold)
        result |= (sector_mask & pattern)

    return result, present, family_map, sector_rgb


def apply_double_coding(
    image: Image.Image,
    opacity: int = 100,
    smooth_shadows: bool = True,
) -> Image.Image:
    import cv2

    arr = np.array(image.convert("RGB"))
    h_img, w_img = arr.shape[:2]

    arr_for_classification = _smooth_shadows(arr) if smooth_shadows else arr

    hsv = cv2.cvtColor(arr_for_classification, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32) * 2
    sat = hsv[:, :, 1].astype(np.float32) / 255
    val = hsv[:, :, 2].astype(np.float32)

    delta = arr_for_classification.astype(np.float32).max(axis=2) - \
            arr_for_classification.astype(np.float32).min(axis=2)

    valid = (
        (delta >= 25) &
        (arr_for_classification.max(axis=2) >= 30) &
        (arr_for_classification.min(axis=2) <= 248) &
        (sat >= 0.12)
    )

    hue_scaled_light = (hue / 360.0 * 255.0).astype(np.uint8)
    hue_legend = cv2.medianBlur(hue_scaled_light, 3).astype(np.float32) / 255.0 * 360.0

    big_mask, present, family_map, sector_rgb = _procedural_pattern_mask_with_families(
        h_img, w_img, hue, sat, valid, arr
    )
    draw_mask = big_mask & valid

    result = arr.astype(np.float32)
    alpha = opacity / 100.0
    result[draw_mask] = result[draw_mask] * (1 - alpha)

    result = _redraw_small_elements(
        arr, valid, result, alpha, family_map, sector_rgb
    )

    return Image.fromarray(result.astype(np.uint8))