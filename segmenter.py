from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image

COLOR_TO_PATTERN: dict[str, tuple[str, tuple]] = {
    "red":           ("diagonal_stripes",  (220,  50,  30)),
    "red_orange":    ("diagonal_stripes",  (210,  70,  20)),
    "orange":        ("horizontal_lines",  (220, 110,   0)),
    "yellow_orange": ("horizontal_lines",  (210, 150,   0)),
    "yellow":        ("checkerboard",      (200, 180,   0)),
    "yellow_green":  ("checkerboard",      (120, 180,   0)),
    "green":         ("dots",              (  0, 160,  80)),
    "cyan":          ("dots",              (  0, 180, 160)),
    "blue":          ("crosshatch",        ( 50,  90, 210)),
    "blue_purple":   ("crosshatch",        ( 80,  60, 200)),
    "purple":        ("vertical_lines",    (120,  50, 200)),
    "magenta":       ("checkerboard",      (180,   0, 160)),
    "pink":          ("diagonal_stripes",  (220,  80, 160)),
}

_NO_PATTERN = {"unknown", "white", "black", "gray"}


def _pixel_hue_name(r: float, g: float, b: float) -> str | None:
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c

    if delta < 30:   return None
    if max_c < 30:   return None
    if min_c > 245:  return None

    saturation = delta / max_c if max_c > 0 else 0
    if saturation < 0.20:
        return None

    if max_c == r:
        h = 60.0 * (((g - b) / delta) % 6)
    elif max_c == g:
        h = 60.0 * (((b - r) / delta) + 2)
    else:
        h = 60.0 * (((r - g) / delta) + 4)
    if h < 0:
        h += 360.0

    if h < 15 or h >= 350:  return "red"
    if h < 35:               return "red_orange"
    if h < 55:               return "orange"
    if h < 75:               return "yellow_orange"
    if h < 95:               return "yellow"
    if h < 135:              return "yellow_green"
    if h < 165:              return "green"
    if h < 200:              return "cyan"
    if h < 255:              return "blue"
    if h < 285:              return "blue_purple"
    if h < 345:              return "magenta"
    return "pink"


def _color_name_from_hsv(hue360: int, sat: int) -> str:
    if sat < 30:
        return "gray"
    h = float(hue360)
    if h < 15 or h >= 350:  return "red"
    if h < 55:               return "orange"
    if h < 95:               return "yellow"
    if h < 165:              return "green"
    if h < 200:              return "cyan"
    if h < 255:              return "blue"
    if h < 285:              return "blue_purple"
    if h < 345:              return "magenta"
    return "pink"


def _find_legend_squares_opencv(
    arr: np.ndarray,
    excluded_colors: set[str],
    min_side: int = 5,
    max_side: int = 40,
) -> list["ColoredSegment"]:
    """
    Находит маленькие цветные квадратики легенды через OpenCV contours
    SAM их не видит используем классический CV подход
    """
    import cv2

    h_img, w_img = arr.shape[:2]
    # Ищем только в верхней трети изображения легенда обычно там
    top_region = arr[:h_img // 3, :, :]

    hsv_region = cv2.cvtColor(top_region, cv2.COLOR_RGB2HSV)

    # Маска насыщенных пикселей (sat > 80) исключаем серый/белый фон
    sat_mask = (hsv_region[:, :, 1] > 60).astype(np.uint8) * 255

    # Морфология закрываем дырки внутри квадратиков
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found: list[ColoredSegment] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Размер квадратика легенды
        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue

        # Почти квадратный
        aspect = w / h if h > 0 else 0
        if not (0.5 < aspect < 2.0):
            continue

        # Solidity 
        bbox_area = w * h
        cnt_area  = cv2.contourArea(cnt)
        if cnt_area / bbox_area < 0.6:
            continue

        # Берём пиксели этого квадратика
        roi = top_region[y:y+h, x:x+w]
        pixels = roi.reshape(-1, 3).astype(float)

        mean_rgb = pixels.mean(axis=0).astype(np.uint8)
        hsv_px   = cv2.cvtColor(
            np.array([[mean_rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV
        )[0][0]
        sat_px = int(hsv_px[1])

        if sat_px < 60:
            continue

        # Классифицируем цвет
        counts: dict[str, int] = {}
        for px in pixels:
            name = _pixel_hue_name(float(px[0]), float(px[1]), float(px[2]))
            if name:
                counts[name] = counts.get(name, 0) + 1

        if not counts:
            continue

        dominant = max(counts, key=counts.__getitem__)
        if dominant in excluded_colors:
            continue

        pattern, pat_color = COLOR_TO_PATTERN.get(
            dominant, ("diagonal_stripes", (120, 120, 120))
        )

        # Создаём маску для этого квадратика (в координатах полного изображения)
        full_mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=bool)
        full_mask[y:y+h, x:x+w] = True

        found.append(ColoredSegment(
            mask=full_mask,
            color_name=dominant,
            pattern=pattern,
            pattern_color=pat_color,
            mean_rgb=tuple(int(v) for v in mean_rgb),
            area=bbox_area,
            predicted_iou=1.0,
            stability_score=1.0,
        ))

    return found


@dataclass
class ColoredSegment:
    mask:            np.ndarray
    color_name:      str
    pattern:         str
    pattern_color:   tuple
    mean_rgb:        tuple[int, int, int]
    area:            int
    predicted_iou:   float
    stability_score: float


class SAMSegmenter:
    def __init__(self, mask_generator):
        self.mask_generator = mask_generator

    def segment(
        self,
        image: Image.Image,
        min_area: int = 500,
        max_area_ratio: float = 0.35,
    ) -> list[ColoredSegment]:
        import cv2

        arr = np.array(image.convert("RGB"))
        h_img, w_img, _ = arr.shape
        total_pixels = h_img * w_img

        # Фон по краям изображения 
        edges = np.concatenate([
            arr[:5,  :,  :].reshape(-1, 3),
            arr[-5:, :,  :].reshape(-1, 3),
            arr[:,  :5,  :].reshape(-1, 3),
            arr[:, -5:,  :].reshape(-1, 3),
        ], axis=0)
        bg_color_avg = edges.mean(axis=0)
        bg_mean = bg_color_avg.astype(np.uint8)
        bg_hsv  = cv2.cvtColor(
            np.array([[bg_mean]], dtype=np.uint8), cv2.COLOR_RGB2HSV
        )[0][0]
        bg_hue  = int(bg_hsv[0]) * 2
        bg_sat  = int(bg_hsv[1])
        edge_color_name = _color_name_from_hsv(bg_hue, bg_sat)

        raw_masks = self.mask_generator.generate(arr)

        # Определяем цвет plot area самый большой сегмент < 80% 
        # Это синяя область графика её тоже не красим
        sorted_by_area = sorted(raw_masks, key=lambda r: -r["segmentation"].sum())
        plot_bg_color_name: str | None = None

        for raw in sorted_by_area:
            area = int(raw["segmentation"].sum())
            if area > total_pixels * 0.80:
                continue  # слишком большой пропускаем
            if area < total_pixels * 0.10:
                break     # дальше только мелкие стоп

            pixels_bg = arr[raw["segmentation"]]
            if len(pixels_bg) == 0:
                continue
            mean_bg = pixels_bg.mean(axis=0).astype(np.uint8)
            hsv_bg  = cv2.cvtColor(
                np.array([[mean_bg]], dtype=np.uint8), cv2.COLOR_RGB2HSV
            )[0][0]
            sat_bg = int(hsv_bg[1])
            hue_bg = int(hsv_bg[0]) * 2
            # Если этот большой объект насыщен и отличается от края это plot area
            if sat_bg > 20 and np.linalg.norm(mean_bg.astype(float) - bg_color_avg) > 30:
                plot_bg_color_name = _color_name_from_hsv(hue_bg, sat_bg)
                break

        # Набор цветов которые не красим
        excluded_colors = {edge_color_name}
        if plot_bg_color_name:
            excluded_colors.add(plot_bg_color_name)

        all_found: list[ColoredSegment] = []

        # Квадратики легенды через OpenCV (SAM их не видит)
        legend_segments = _find_legend_squares_opencv(arr, excluded_colors)
        print(f"[DEBUG LEGEND] found {len(legend_segments)} squares: {[(s.color_name, s.area) for s in legend_segments]}")
        print(f"[DEBUG] excluded_colors={excluded_colors}")
        all_found.extend(legend_segments)

        for raw in raw_masks:
            mask  = raw["segmentation"]
            area  = int(mask.sum())
            bbox  = raw["bbox"]

            # Размер 
            if area > total_pixels * max_area_ratio:
                continue

            aspect = bbox[2] / bbox[3] if bbox[3] > 0 else 0

            # Квадратики легенды маленькие (50-500px) и квадратные
            is_legend_square = (
                50 <= area < min_area
                and 0.5 < aspect < 2.0
            )

            if area < 50:
                continue
            if not is_legend_square and area < min_area:
                continue

            # Solidity фильтр текста 
            bbox_area = bbox[2] * bbox[3]
            solidity  = area / bbox_area if bbox_area > 0 else 0
            if solidity < 0.65:
                continue

            # HSV 
            pixels = arr[mask]
            if len(pixels) == 0:
                continue

            mean_rgb = pixels.mean(axis=0).astype(np.uint8)
            hsv = cv2.cvtColor(
                np.array([[mean_rgb]], dtype=np.uint8),
                cv2.COLOR_RGB2HSV
            )[0][0]
            sat, val = int(hsv[1]), int(hsv[2])
            seg_hue  = int(hsv[0]) * 2

            if sat < 40:    continue
            if val > 252:   continue
            if val < 25:    continue

            # Дистанция до фона краёв 
            bg_dist = np.linalg.norm(mean_rgb.astype(float) - bg_color_avg)
            if bg_dist < 45:
                continue

            # Классификация 
            color_name, pattern, pat_color, res_mean_rgb, _ = \
                self._classify_region(pixels)

            if color_name in _NO_PATTERN:
                continue

            # Исключаем фон и plot area 
            if color_name in excluded_colors:
                continue

            all_found.append(ColoredSegment(
                mask=mask,
                color_name=color_name,
                pattern=pattern,
                pattern_color=pat_color,
                mean_rgb=tuple(int(v) for v in res_mean_rgb),
                area=area,
                predicted_iou=float(raw.get("predicted_iou", 0.0)),
                stability_score=float(raw.get("stability_score", 0.0)),
            ))

        return sorted(all_found, key=lambda s: -s.area)

    @staticmethod
    def _classify_region(pixels: np.ndarray, sample_n: int = 1000) -> tuple:
        if len(pixels) > sample_n:
            idx    = np.random.choice(len(pixels), sample_n, replace=False)
            pixels = pixels[idx]

        mean_rgb = tuple(int(v) for v in pixels.mean(axis=0))
        counts: dict[str, int] = {}
        total_colored = 0

        for px in pixels:
            name = _pixel_hue_name(float(px[0]), float(px[1]), float(px[2]))
            if name:
                counts[name] = counts.get(name, 0) + 1
                total_colored += 1

        if not counts:
            return ("unknown", "diagonal_stripes", (120, 120, 120), mean_rgb, 0.0)

        dominant = max(counts, key=counts.__getitem__)
        pattern, pat_color = COLOR_TO_PATTERN.get(
            dominant, ("diagonal_stripes", (120, 120, 120))
        )
        return (dominant, pattern, pat_color, mean_rgb, total_colored / len(pixels))


def deduplicate_segments(
    segments: list[ColoredSegment],
    iou_threshold: float = 0.7,
) -> list[ColoredSegment]:
    if len(segments) <= 1:
        return segments

    kept: list[ColoredSegment] = []
    for seg in segments:
        dominated = False
        for other in kept:
            intersection = int((seg.mask & other.mask).sum())
            union        = int((seg.mask | other.mask).sum())
            iou = intersection / union if union > 0 else 0.0
            if iou > iou_threshold:
                dominated = True
                break
        if not dominated:
            kept.append(seg)
    return kept