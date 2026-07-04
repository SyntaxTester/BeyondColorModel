from __future__ import annotations
import numpy as np
from PIL import Image


# 8 углов вразброс  соседние по кругу сектора получают максимально разные углы
_ANGLE_LUT = [0, 90, 22.5, 112.5, 45, 135, 67.5, 157.5]

# Семейства паттернов  используются по очереди при конфликте похожих секторов
_PATTERN_FAMILIES = ["lines", "dots", "grid", "dashed"]


def _smooth_shadows(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Bilateral filter  сглаживает плавные градиенты яркости (тени, блики на 3D)
    но сохраняет резкие границы между физически разными объектами.
    """
    import cv2
    median = cv2.medianBlur(arr_rgb, 5)
    return cv2.bilateralFilter(median, d=15, sigmaColor=80, sigmaSpace=80)


def _get_present_sectors(hue_smooth: np.ndarray, valid: np.ndarray, n_sectors: int = 8) -> set[int]:
    """Какие сектора реально присутствуют на этом изображении в заметном объёме."""
    sector = np.floor(hue_smooth / (360.0 / n_sectors)).astype(np.int32) % n_sectors
    present = set()
    total_valid = valid.sum()
    if total_valid == 0:
        return present
    for s in range(n_sectors):
        count = ((sector == s) & valid).sum()
        if count > total_valid * 0.01:  # заметная доля изображения
            present.add(s)
    return present


def _assign_pattern_families(present_sectors: set[int]) -> dict[int, str]:
    """
    Семейство паттерна = номер сектора по модулю количества семейств.
    Это гарантирует что ЛЮБЫЕ соседние по кругу сектора (разница в 1)
    получат разные семейства математически, без отслеживания состояния 
    предыдущая версия помнила только последний обработанный сектор
    и застревала на одном семействе для всех несмежных секторов подряд.
    """
    n_families = len(_PATTERN_FAMILIES)
    return {s: _PATTERN_FAMILIES[s % n_families] for s in present_sectors}


def _draw_pattern(
    family: str,
    xx: np.ndarray, yy: np.ndarray,
    angle_rad: float,
    spacing: float = 12.0,
) -> np.ndarray:
    """Рисует один из 4 типов паттерна (lines/dots/grid/dashed) под заданным углом."""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    proj = xx * cos_a + yy * sin_a
    perp = -xx * sin_a + yy * cos_a

    if family == "lines":
        return (proj % spacing) < (spacing * 0.33)

    if family == "dashed":
        line = (proj % spacing) < (spacing * 0.33)
        dash = (perp % (spacing * 1.4)) < (spacing * 1.4 * 0.5)
        return line & dash

    if family == "grid":
        line1 = (proj % spacing) < (spacing * 0.25)
        line2 = (perp % spacing) < (spacing * 0.25)
        return line1 | line2

    if family == "dots":
        gx = (proj % spacing)
        gy = (perp % spacing)
        cx = cy = spacing / 2
        r = spacing * 0.28
        return (gx - cx) ** 2 + (gy - cy) ** 2 <= r ** 2

    return (proj % spacing) < (spacing * 0.33)


def _procedural_pattern_mask(
    h_img: int, w_img: int,
    hue: np.ndarray, val: np.ndarray, valid: np.ndarray,
) -> np.ndarray:
    """
    8 фиксированных секторов hue (устойчиво, без разрыва паттерна внутри
    одного физического объекта). Если на изображении реально присутствуют
    два соседних по кругу сектора одновременно  им назначаются разные
    СЕМЕЙСТВА паттернов (не просто углы), проверка идёт в рантайме по
    факту содержимого картинки, а не по заранее прописанному списку цветов.
    """
    import cv2

    hue_scaled = (hue / 360.0 * 255.0).astype(np.uint8)
    hue_smooth = cv2.medianBlur(hue_scaled, 15).astype(np.float32) / 255.0 * 360.0
    val_smooth = cv2.medianBlur(val.astype(np.uint8), 9).astype(np.float32)

    n_sectors = 8
    sector = np.floor(hue_smooth / (360.0 / n_sectors)).astype(np.int32) % n_sectors

    present = _get_present_sectors(hue_smooth, valid, n_sectors)
    family_map = _assign_pattern_families(present)

    yy, xx = np.mgrid[0:h_img, 0:w_img].astype(np.float32)
    result = np.zeros((h_img, w_img), dtype=bool)

    for s in present:
        sector_mask = (sector == s)
        if not sector_mask.any():
            continue
        angle_rad = np.deg2rad(_ANGLE_LUT[s])
        family = family_map.get(s, "lines")
        pattern = _draw_pattern(family, xx, yy, angle_rad)
        result |= (sector_mask & pattern)

    return result


def apply_double_coding(
    image: Image.Image,
    opacity: int = 100,
    smooth_shadows: bool = True,
) -> Image.Image:
    """
    Применяет дабл кодинг ко всему изображению как фотофильтр
    Каждый пиксель классифицируется по HSV и получает процедурно
    сгенерированную штриховку  без захардкоженного списка цветов
    """
    import cv2

    arr = np.array(image.convert("RGB"))
    h_img, w_img = arr.shape[:2]

    if smooth_shadows:
        arr_for_classification = _smooth_shadows(arr)
    else:
        arr_for_classification = arr

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

    draw_mask = _procedural_pattern_mask(h_img, w_img, hue, val, valid) & valid

    result = arr.astype(np.float32)
    alpha = opacity / 100.0
    result[draw_mask] = result[draw_mask] * (1 - alpha)

    return Image.fromarray(result.astype(np.uint8))