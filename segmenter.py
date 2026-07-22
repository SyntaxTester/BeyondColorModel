from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image

COLOR_TO_PATTERN: dict[str, tuple[str, tuple]] = {
    "red":           ("diagonal_stripes",  (220,  50,  30)),
    "red_orange":    ("horizontal_lines",  (210,  70,  20)),
    "orange":        ("horizontal_lines",  (220, 110,   0)),
    "yellow_orange": ("checkerboard",      (210, 150,   0)),
    "yellow":        ("checkerboard",      (200, 180,   0)),
    "yellow_green":  ("dots",              (120, 180,   0)),
    "green":         ("dots",              (  0, 160,  80)),
    "cyan":          ("crosshatch",        (  0, 180, 160)),
    "blue":          ("crosshatch",        ( 50,  90, 210)),
    "blue_purple":   ("vertical_lines",    ( 80,  60, 200)),
    "purple":        ("vertical_lines",    (120,  50, 200)),
    "magenta":       ("checkerboard",      (180,   0, 160)),
    "rose":          ("diagonal_stripes",  (220,  20, 120)),  # hue 315-330
    "pink":          ("dots",              (220,  80, 160)),
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

    v = max_c

    if h < 15 or h >= 350:  return "red"
    if h < 25:               return "red_orange"
    if h < 60:               return "orange"
    if h < 75:               return "yellow_orange"
    if h < 80:               return "yellow"
    if h < 135:              return "yellow_green"
    if h < 165:              return "green"
    if h < 200:              return "cyan"
    if h < 255:              return "blue"
    if h < 270:              return "blue_purple"
    if h < 315:
        return "purple" if v < 180 else "magenta"
    if h < 330:              return "rose"    # Chess (hue=322)
    if h < 345:
        return "pink" if v > 180 else "magenta"
    return "pink"


def _color_name_from_hsv(hue360: int, sat: int) -> str:
    if sat < 30:
        return "gray"
    h = float(hue360)
    if h < 15 or h >= 350:  return "red"
    if h < 60:               return "orange"
    if h < 80:               return "yellow"
    if h < 165:              return "green"
    if h < 200:              return "cyan"
    if h < 255:              return "blue"
    if h < 270:              return "blue_purple"
    if h < 315:              return "purple"
    if h < 330:              return "rose"
    if h < 345:              return "magenta"
    return "pink"


def _is_text_like(mask: np.ndarray, bbox: list, area: int, arr: np.ndarray) -> bool:
    import cv2
    bw, bh = bbox[2], bbox[3]
    if bw == 0 or bh == 0:
        return True

    aspect = bw / bh
    if aspect > 5.0 or aspect < 0.1:
        return True

    
    bbox_area = bw * bh
    solidity = area / bbox_area if bbox_area > 0 else 0
    if area < 8000 and solidity < 0.60:
        return True

    
    pixels = arr[mask]
    if len(pixels) == 0:
        return True
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    unsaturated = (hsv[:, 1] < 25).sum()
    if unsaturated / len(pixels) > 0.55 and area < 3000:
        return True

    return False


def _find_legend_squares_opencv(
    arr: np.ndarray,
    excluded_colors: set[str],
    min_side: int = 5,
    max_side: int = 40,
    is_pie: bool = False,
) -> list["ColoredSegment"]:
    import cv2
    h_img, w_img = arr.shape[:2]
    
    if is_pie:
        search_region = arr[:, w_img // 2:, :]
        x_offset = w_img // 2
        y_offset = 0
    else:
        search_region = arr[:h_img // 3, :, :]
        x_offset = 0
        y_offset = 0
    hsv_region = cv2.cvtColor(search_region, cv2.COLOR_RGB2HSV)
    sat_mask = (hsv_region[:, :, 1] > 60).astype(np.uint8) * 255
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found: list[ColoredSegment] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        aspect = w / h if h > 0 else 0
        if not (0.4 < aspect < 2.5):
            continue
        bbox_area = w * h
        cnt_area  = cv2.contourArea(cnt)
        if cnt_area / bbox_area < 0.5:
            continue
        roi = search_region[y:y+h, x:x+w]
        pixels = roi.reshape(-1, 3).astype(float)
        mean_rgb = pixels.mean(axis=0).astype(np.uint8)
        hsv_px = cv2.cvtColor(
            np.array([[mean_rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV
        )[0][0]
        if int(hsv_px[1]) < 60:
            continue
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
        pattern, pat_color = COLOR_TO_PATTERN.get(dominant, ("diagonal_stripes", (120, 120, 120)))
        contour_mask = np.zeros(search_region.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=-1)

        full_mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=bool)
        full_mask[
            y_offset:y_offset + search_region.shape[0],
            x_offset:x_offset + search_region.shape[1],
        ] = contour_mask.astype(bool)
        actual_area = int(full_mask.sum())
        if actual_area == 0:
            continue

        found.append(ColoredSegment(
            mask=full_mask, color_name=dominant, pattern=pattern,
            pattern_color=pat_color, mean_rgb=tuple(int(v) for v in mean_rgb),
            area=actual_area, predicted_iou=1.0, stability_score=1.0,
            is_legend=True,
        ))
    return found


def _find_missing_legend_markers_by_column(
    arr: np.ndarray,
    existing: list["ColoredSegment"],
    excluded_colors: set[str],
) -> list["ColoredSegment"]:
    import cv2

    h_img, w_img = arr.shape[:2]
    x_offset = w_img // 2
    search_region = arr[:, x_offset:, :]
    hsv = cv2.cvtColor(search_region, cv2.COLOR_RGB2HSV)
    colored = (hsv[:, :, 1] > 45).astype(np.uint8)
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(colored)
    candidates: list[tuple[int, np.ndarray, float]] = []

    for label_id in range(1, labels_count):
        x, y, width, height, area = stats[label_id]
        if not (3 <= width <= 40 and 3 <= height <= 40 and area >= 9):
            continue
        aspect = width / height
        if not 0.35 <= aspect <= 2.8:
            continue
        if area / (width * height) < 0.35:
            continue
        candidates.append((label_id, labels == label_id, float(centroids[label_id][0])))

    if len(candidates) < 3:
        return []

    median_x = float(np.median([candidate[2] for candidate in candidates]))
    column = [candidate for candidate in candidates if abs(candidate[2] - median_x) <= 6]
    if len(column) < 3:
        return []

    recovered: list[ColoredSegment] = []
    for _, local_mask, _ in column:
        full_mask = np.zeros((h_img, w_img), dtype=bool)
        full_mask[:, x_offset:] = local_mask
        area = int(full_mask.sum())
        if any(int((full_mask & segment.mask).sum()) / area > 0.5 for segment in existing):
            continue

        mean_rgb = arr[full_mask].mean(axis=0).astype(np.uint8)
        color_name = _pixel_hue_name(*map(float, mean_rgb))
        if color_name is None or color_name in excluded_colors:
            continue
        pattern, pattern_color = COLOR_TO_PATTERN[color_name]
        recovered.append(ColoredSegment(
            mask=full_mask, color_name=color_name, pattern=pattern,
            pattern_color=pattern_color, mean_rgb=tuple(int(v) for v in mean_rgb),
            area=area, predicted_iou=1.0, stability_score=1.0,
            is_legend=True,
        ))

    return recovered


def _split_segment_by_brightness(seg, arr, color_to_pattern):
    import cv2
    pixels_in_seg = arr[seg.mask]
    if len(pixels_in_seg) == 0:
        return [seg]
    hsv_pixels = cv2.cvtColor(
        pixels_in_seg.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV
    ).reshape(-1, 3)
    values = hsv_pixels[:, 2]
    median_val = float(np.median(values))
    std_val = float(np.std(values))
    if std_val < 25:
        return [seg]
    arr_hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    val_channel = arr_hsv[:, :, 2]
    dark_mask  = seg.mask & (val_channel < median_val)
    light_mask = seg.mask & (val_channel >= median_val)
    min_sub_area = seg.area // 5
    result = []
    for sub_mask in [dark_mask, light_mask]:
        sub_area = int(sub_mask.sum())
        if sub_area < min_sub_area:
            continue
        sub_pixels = arr[sub_mask]
        sub_mean = sub_pixels.mean(axis=0).astype(np.uint8)
        color_name = _pixel_hue_name(float(sub_mean[0]), float(sub_mean[1]), float(sub_mean[2]))
        if color_name is None:
            color_name = seg.color_name
        pattern, pat_color = color_to_pattern.get(color_name, (seg.pattern, seg.pattern_color))
        result.append(ColoredSegment(
            mask=sub_mask, color_name=color_name, pattern=pattern,
            pattern_color=pat_color, mean_rgb=tuple(int(v) for v in sub_mean),
            area=sub_area, predicted_iou=seg.predicted_iou, stability_score=seg.stability_score,
        ))
    return result if len(result) > 1 else [seg]


def _find_light_bars_opencv(
    arr: np.ndarray,
    bg_color_avg: np.ndarray,
    existing_masks: list,
    min_area: int = 1500,
) -> list["ColoredSegment"]:
    
    import cv2
    h_img, w_img = arr.shape[:2]

    
    diff = np.linalg.norm(arr.astype(float) - bg_color_avg, axis=2)
    hsv_full = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat_full = hsv_full[:, :, 1]
    colored_mask = ((diff > 25) & (sat_full > 25)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(colored_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    covered = np.zeros((h_img, w_img), dtype=bool)
    for m in existing_masks:
        covered |= m

    found: list[ColoredSegment] = []
    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        if area < min_area:
            continue

        cnt_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        full_mask = cnt_mask.astype(bool)

        
        uncovered_mask = (full_mask & ~covered).astype(np.uint8)
        if int(uncovered_mask.sum()) < min_area:
            continue

        ys, xs = np.where(uncovered_mask > 0)
        if len(ys) == 0:
            continue
        sub_pixels = arr[ys, xs].astype(np.float32)
        hsv_sub = cv2.cvtColor(sub_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        hue_sub = hsv_sub[:, 0].astype(np.float32)

        hue_std = float(np.std(hue_sub))
        if hue_std < 8 or len(hue_sub) < 200:
            color_groups = [np.ones(len(ys), dtype=bool)]
        else:
            hue_for_kmeans = hue_sub.reshape(-1, 1).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            try:
                _, labels_km, centers = cv2.kmeans(
                    hue_for_kmeans, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
                )
                labels_km = labels_km.flatten()
                center_diff = abs(float(centers[0][0]) - float(centers[1][0]))
                if center_diff < 8:
                    color_groups = [np.ones(len(ys), dtype=bool)]
                else:
                    color_groups = [labels_km == 0, labels_km == 1]
            except cv2.error:
                color_groups = [np.ones(len(ys), dtype=bool)]

        for group_sel in color_groups:
            if group_sel.sum() < min_area:
                continue
            gy, gx = ys[group_sel], xs[group_sel]
            group_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            group_mask[gy, gx] = 255

                
            n_labels, labels = cv2.connectedComponents(group_mask)
            for label_id in range(1, n_labels):
                comp_mask = (labels == label_id)
                comp_area = int(comp_mask.sum())
                if comp_area < min_area:
                    continue

                cys, cxs = np.where(comp_mask)
                sub_w = cxs.max() - cxs.min() + 1
                sub_h = cys.max() - cys.min() + 1
                sub_bbox_area = sub_w * sub_h
                solidity = comp_area / sub_bbox_area if sub_bbox_area > 0 else 0
                if solidity < 0.5:
                    continue
                aspect = sub_w / sub_h if sub_h > 0 else 0
                if aspect > 4.0 or aspect < 0.1:
                    continue

                pixels = arr[comp_mask]
                mean_rgb = pixels.mean(axis=0).astype(np.uint8)
                hsv_px = cv2.cvtColor(np.array([[mean_rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]
                sat, val = int(hsv_px[1]), int(hsv_px[2])
                if sat < 15: continue
                if val < 60: continue
                if val > 252 and sat < 20: continue

                color_name = _pixel_hue_name(float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]))
                if color_name is None or color_name in _NO_PATTERN:
                    continue

                pattern, pat_color = COLOR_TO_PATTERN.get(color_name, ("diagonal_stripes", (120, 120, 120)))
                found.append(ColoredSegment(
                    mask=comp_mask, color_name=color_name, pattern=pattern,
                    pattern_color=pat_color, mean_rgb=tuple(int(v) for v in mean_rgb),
                    area=comp_area, predicted_iou=0.8, stability_score=0.8,
                ))
    return found


def _find_missing_pie_slices_opencv(
    arr: np.ndarray,
    existing_segments: list["ColoredSegment"],
    min_area: int,
) -> list["ColoredSegment"]:
    import cv2

    data_masks = [segment.mask for segment in existing_segments if not segment.is_legend]
    if not data_masks:
        return []

    covered = np.logical_or.reduce(data_masks)
    covered_y, covered_x = np.where(covered)
    if len(covered_x) == 0:
        return []

    x_min, x_max = int(covered_x.min()), int(covered_x.max())
    y_min, y_max = int(covered_y.min()), int(covered_y.max())
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    colored = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 35)).astype(np.uint8)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(colored)
    required_area = max(50, min_area)
    found: list[ColoredSegment] = []

    for label_id in range(1, labels_count):
        component_area = int(stats[label_id, cv2.CC_STAT_AREA])
        if component_area < required_area:
            continue

        component = labels == label_id
        ys, xs = np.where(component)
        if len(xs) == 0:
            continue
        if xs.min() < x_min - 4 or xs.max() > x_max + 4:
            continue
        if ys.min() < y_min - 4 or ys.max() > y_max + 4:
            continue

        missing = component & ~covered
        missing_area = int(missing.sum())
        if missing_area < required_area:
            continue

        mean_rgb = arr[component].mean(axis=0).astype(np.uint8)
        color_name = _pixel_hue_name(*map(float, mean_rgb))
        if color_name is None or color_name in _NO_PATTERN:
            continue
        pattern, pattern_color = COLOR_TO_PATTERN[color_name]
        found.append(ColoredSegment(
            mask=missing, color_name=color_name, pattern=pattern,
            pattern_color=pattern_color, mean_rgb=tuple(int(v) for v in mean_rgb),
            area=missing_area, predicted_iou=0.6, stability_score=0.6,
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
    is_legend:       bool = False
    series_id:       int | None = None


@dataclass
class PatternSeries:
    series_id: int
    mean_rgb: np.ndarray
    area: int
    members: list[ColoredSegment]


def _lab_color_distance(left: tuple | np.ndarray, right: tuple | np.ndarray) -> float:
    import cv2

    colors = np.array([left[:3], right[:3]], dtype=np.uint8).reshape(2, 1, 3)
    lab = cv2.cvtColor(colors, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(2, 3)
    return float(np.linalg.norm(lab[0] - lab[1]))


def assign_patterns_by_series(
    segments: list[ColoredSegment],
    color_distance_threshold: float = 42.0,
) -> list[ColoredSegment]:
    if not segments:
        return segments

    data_segments = sorted(
        (segment for segment in segments if not segment.is_legend),
        key=lambda segment: -segment.area,
    )
    legend_segments = [segment for segment in segments if segment.is_legend]
    series: list[PatternSeries] = []

    for segment in data_segments:
        closest = min(
            series,
            key=lambda candidate: _lab_color_distance(segment.mean_rgb, candidate.mean_rgb),
            default=None,
        )
        if closest is None or _lab_color_distance(segment.mean_rgb, closest.mean_rgb) > color_distance_threshold:
            closest = PatternSeries(
                series_id=len(series),
                mean_rgb=np.asarray(segment.mean_rgb, dtype=np.float32),
                area=0,
                members=[],
            )
            series.append(closest)
        total_area = closest.area + segment.area
        closest.mean_rgb = (
            closest.mean_rgb * closest.area
            + np.asarray(segment.mean_rgb, dtype=np.float32) * segment.area
        ) / total_area
        closest.area = total_area
        closest.members.append(segment)

    for segment in legend_segments:
        closest = min(
            series,
            key=lambda candidate: _lab_color_distance(segment.mean_rgb, candidate.mean_rgb),
            default=None,
        )
        legend_threshold = color_distance_threshold * 1.4
        if closest is None or _lab_color_distance(segment.mean_rgb, closest.mean_rgb) > legend_threshold:
            closest = PatternSeries(
                series_id=len(series),
                mean_rgb=np.asarray(segment.mean_rgb, dtype=np.float32),
                area=0,
                members=[],
            )
            series.append(closest)
        total_area = closest.area + segment.area
        closest.mean_rgb = (
            closest.mean_rgb * closest.area
            + np.asarray(segment.mean_rgb, dtype=np.float32) * segment.area
        ) / total_area
        closest.area = total_area
        closest.members.append(segment)

    for candidate in series:
        rgb = tuple(int(value) for value in np.rint(candidate.mean_rgb))
        color_name = _pixel_hue_name(*map(float, rgb))
        if color_name is None:
            color_name = "blue"
        pattern, pattern_color = COLOR_TO_PATTERN[color_name]
        for member in candidate.members:
            member.series_id = candidate.series_id
            member.color_name = color_name
            member.pattern = pattern
            member.pattern_color = pattern_color

    return segments


class SAMSegmenter:
    def __init__(self, mask_generator):
        self.mask_generator = mask_generator

    def segment(
        self,
        image: Image.Image,
        min_area: int = 500,
        max_area_ratio: float = 0.35,
        is_pie: bool = False,
    ) -> list[ColoredSegment]:
        import cv2

        arr = np.array(image.convert("RGB"))
        h_img, w_img, _ = arr.shape
        total_pixels = h_img * w_img

        edges = np.concatenate([
            arr[:5,  :,  :].reshape(-1, 3), arr[-5:, :,  :].reshape(-1, 3),
            arr[:,  :5,  :].reshape(-1, 3), arr[:, -5:,  :].reshape(-1, 3),
        ], axis=0)
        bg_color_avg = edges.mean(axis=0)
        bg_mean = bg_color_avg.astype(np.uint8)
        bg_hsv  = cv2.cvtColor(np.array([[bg_mean]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]
        bg_hue = int(bg_hsv[0]) * 2
        bg_sat = int(bg_hsv[1])

        if bg_sat > 30:
            arr_hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            light_mask = arr_hsv[:, :, 2] > 240
            if light_mask.sum() > 100:
                bg_color_avg = arr[light_mask].mean(axis=0)
                bg_mean = bg_color_avg.astype(np.uint8)
                bg_hsv  = cv2.cvtColor(np.array([[bg_mean]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]
                bg_hue = int(bg_hsv[0]) * 2
                bg_sat = int(bg_hsv[1])

        edge_color_name = _color_name_from_hsv(bg_hue, bg_sat)
        raw_masks = self.mask_generator.generate(arr)
        print(f"[DEBUG SAM] generated {len(raw_masks)} masks, image={arr.shape}")

        sorted_by_area = sorted(raw_masks, key=lambda r: -r["segmentation"].sum())
        plot_bg_color_name: str | None = None
        for raw in sorted_by_area:
            area = int(raw["segmentation"].sum())
            if area > total_pixels * 0.80: continue
            if area < total_pixels * 0.10: break
            pixels_bg = arr[raw["segmentation"]]
            if len(pixels_bg) == 0: continue
            mean_bg = pixels_bg.mean(axis=0).astype(np.uint8)
            hsv_bg = cv2.cvtColor(np.array([[mean_bg]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]
            sat_bg = int(hsv_bg[1])
            hue_bg = int(hsv_bg[0]) * 2
            if sat_bg > 40 and np.linalg.norm(mean_bg.astype(float) - bg_color_avg) > 30:
                plot_bg_color_name = _color_name_from_hsv(hue_bg, sat_bg)
                break

        excluded_colors: set[str] = set()
        if not is_pie:
            excluded_colors.add(edge_color_name)
            if plot_bg_color_name and bg_sat < 30:
                for raw in sorted_by_area:
                    a = int(raw["segmentation"].sum())
                    if a > total_pixels * 0.20:
                        px_bg = arr[raw["segmentation"]]
                        if len(px_bg) == 0: continue
                        m = px_bg.mean(axis=0).astype(np.uint8)
                        h2 = cv2.cvtColor(np.array([[m]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]
                        if _color_name_from_hsv(int(h2[0])*2, int(h2[1])) == plot_bg_color_name:
                            excluded_colors.add(plot_bg_color_name)
                    break

        print(f"[DEBUG is_pie={is_pie}] excluded={excluded_colors}")

        all_found: list[ColoredSegment] = []
        legend_segments = _find_legend_squares_opencv(arr, excluded_colors, is_pie=is_pie)
        if is_pie:
            recovered_markers = _find_missing_legend_markers_by_column(
                arr, legend_segments, excluded_colors
            )
            legend_segments.extend(recovered_markers)
        print(f"[DEBUG LEGEND] found {len(legend_segments)}: {[(s.color_name, s.area) for s in legend_segments]}")
        all_found.extend(legend_segments)

        for raw in raw_masks:
            mask  = raw["segmentation"]
            area  = int(mask.sum())
            bbox  = raw["bbox"]

            if area > total_pixels * max_area_ratio: continue
            if area < 50: continue

            aspect = bbox[2] / bbox[3] if bbox[3] > 0 else 0
            is_legend_square = (50 <= area < min_area and 0.4 < aspect < 2.5)
            if not is_legend_square and area < min_area: continue

            if _is_text_like(mask, bbox, area, arr):
                if area > 3000:
                    print(f"[DEBUG SKIP] area={area} bbox_wh=({bbox[2]},{bbox[3]}) TEXT_LIKE")
                continue

            pixels = arr[mask]
            if len(pixels) == 0: continue

            mean_rgb = pixels.mean(axis=0).astype(np.uint8)
            hsv = cv2.cvtColor(np.array([[mean_rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]
            sat, val = int(hsv[1]), int(hsv[2])

            sat_threshold = 20 if is_pie else 30
            if sat < sat_threshold:
                if area > 3000:
                    print(f"[DEBUG SKIP] area={area} RGB={tuple(mean_rgb)} sat={sat} < {sat_threshold} REJECTED")
                continue
            if val > 252: continue
            if val < 25:  continue

            dist_threshold = 15 if is_pie else 45
            bgd = np.linalg.norm(mean_rgb.astype(float) - bg_color_avg)
            if bgd < dist_threshold:
                if area > 3000:
                    print(f"[DEBUG SKIP] area={area} RGB={tuple(mean_rgb)} bg_dist={bgd:.0f} < {dist_threshold} REJECTED")
                continue

            color_name, pattern, pat_color, res_mean_rgb, _ = self._classify_region(pixels)
            if color_name in _NO_PATTERN: continue
            if color_name in excluded_colors: continue

            all_found.append(ColoredSegment(
                mask=mask, color_name=color_name, pattern=pattern,
                pattern_color=pat_color, mean_rgb=tuple(int(v) for v in res_mean_rgb),
                area=area, predicted_iou=float(raw.get("predicted_iou", 0.0)),
                stability_score=float(raw.get("stability_score", 0.0)),
            ))

        if is_pie:
            missing_slices = _find_missing_pie_slices_opencv(
                arr, all_found, min_area=min_area
            )
            print(f"[DEBUG PIE FALLBACK] found {len(missing_slices)}: {[(s.color_name, s.area) for s in missing_slices]}")
            all_found.extend(missing_slices)

        result = sorted(all_found, key=lambda s: -s.area)

        if is_pie:
            split_result = []
            for seg in result:
                if seg.area > total_pixels * 0.10:
                    split_result.extend(_split_segment_by_brightness(seg, arr, COLOR_TO_PATTERN))
                else:
                    split_result.append(seg)
            result = sorted(split_result, key=lambda s: -s.area)
        else:
            # Fallback: ищем светлые/бледные столбцы, которые SAM пропустил
            existing_masks = [s.mask for s in result]
            light_bars = _find_light_bars_opencv(arr, bg_color_avg, existing_masks, min_area=min_area)
            print(f"[DEBUG LIGHT BARS] found {len(light_bars)}: {[(s.color_name, s.area) for s in light_bars]}")
            result = sorted(result + light_bars, key=lambda s: -s.area)

        print(f"[DEBUG SEGMENTS] {[(s.color_name, s.area) for s in result]}")
        return result

    @staticmethod
    def _classify_region(pixels: np.ndarray, sample_n: int = 1000) -> tuple:
        if len(pixels) > sample_n:
            idx = np.linspace(0, len(pixels) - 1, sample_n, dtype=np.intp)
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
        pattern, pat_color = COLOR_TO_PATTERN.get(dominant, ("diagonal_stripes", (120, 120, 120)))
        return (dominant, pattern, pat_color, mean_rgb, total_colored / len(pixels))


def deduplicate_segments(segments: list[ColoredSegment], iou_threshold: float = 0.7) -> list[ColoredSegment]:
    if len(segments) <= 1:
        return segments
    kept: list[ColoredSegment] = []
    for seg in segments:
        dominated = False
        for other in kept:
            intersection = int((seg.mask & other.mask).sum())
            union = int((seg.mask | other.mask).sum())
            if union > 0 and intersection / union > iou_threshold:
                dominated = True
                break
        if not dominated:
            kept.append(seg)
    return kept
