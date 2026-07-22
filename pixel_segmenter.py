from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class _PatternSpec:
    family: str
    angle_degrees: float
    spacing_scale: float = 1.0


_PATTERN_LIBRARY = [
    _PatternSpec("lines", 0.0),
    _PatternSpec("dots", 0.0),
    _PatternSpec("dashed", 45.0),
    _PatternSpec("lines", 90.0),
    # A sine wave and a zigzag collapse into the same chevron shape at small
    # raster sizes.  Crosshatch remains visually distinct from zigzag.
    _PatternSpec("crosshatch", 22.5),
    _PatternSpec("zigzag", 0.0),
    _PatternSpec("lines", 45.0),
    _PatternSpec("dots", 0.0, 1.35),
    _PatternSpec("dashed", -45.0),
    _PatternSpec("lines", -45.0),
    _PatternSpec("waves", 90.0),
    _PatternSpec("zigzag", 90.0),
    _PatternSpec("crosshatch", 22.5),
    _PatternSpec("grid", 45.0),
    _PatternSpec("bricks", 0.0),
    _PatternSpec("lines", 22.5),
]
# A 2x cap keeps the source labels sharp.  Small swatches remain readable
# because legend patterns use their own denser cadence below; scaling further
# only makes the raster text visibly soft.  The target lets larger markers use
# a gentler scale rather than always hitting that cap.
_LEGEND_SWATCH_TARGET_SIZE = 14
_MARKER_SHAPES = ["circle", "square", "triangle", "diamond", "cross", "star"]

_CVD_MATRICES = {
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                             [0.114503, 0.786281, 0.099216],
                             [-0.003882, -0.048116, 1.051998]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779],
                             [-0.078411, 0.930809, 0.147602],
                             [0.004733, 0.691367, 0.303900]]),
}
_CONFUSABLE_DE = 15.0


def _pattern_spec_for_series(series_id: int) -> _PatternSpec:
    base = _PATTERN_LIBRARY[series_id % len(_PATTERN_LIBRARY)]
    variation = series_id // len(_PATTERN_LIBRARY)
    if variation == 0:
        return base

    angle_offset = 22.5 * (variation % 4)
    spacing_scale = base.spacing_scale * (1.0 + 0.12 * variation)
    return _PatternSpec(base.family, base.angle_degrees + angle_offset, spacing_scale)


@dataclass
class _PixelSeries:
    mean_rgb: np.ndarray
    area: int
    masks: list[np.ndarray] = field(default_factory=list)
    family: str = "lines"
    angle_rad: float = 0.0
    spacing_scale: float = 1.0


def _lab_distance(left: np.ndarray, right: np.ndarray) -> float:
    import cv2

    colors = np.array([left[:3], right[:3]], dtype=np.uint8).reshape(2, 1, 3)
    lab = cv2.cvtColor(colors, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(2, 3)
    return float(np.linalg.norm(lab[0] - lab[1]))


def _hue_sort_key(color: np.ndarray) -> tuple[int, int, int]:
    import cv2

    hsv = cv2.cvtColor(
        np.array([[color.astype(np.uint8)]], dtype=np.uint8), cv2.COLOR_RGB2HSV
    )[0][0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def _hue_distance(left: np.ndarray, right: np.ndarray) -> int:
    left_hue = _hue_sort_key(left)[0]
    right_hue = _hue_sort_key(right)[0]
    return min(abs(left_hue - right_hue), 180 - abs(left_hue - right_hue))


def _masks_are_adjacent(left_masks: list[np.ndarray], right_masks: list[np.ndarray]) -> bool:
    for left in left_masks:
        left_y, left_x = np.where(left)
        if not len(left_x):
            continue
        left_x1, left_x2 = left_x.min(), left_x.max() + 1
        left_y1, left_y2 = left_y.min(), left_y.max() + 1
        for right in right_masks:
            right_y, right_x = np.where(right)
            if not len(right_x):
                continue
            right_x1, right_x2 = right_x.min(), right_x.max() + 1
            right_y1, right_y2 = right_y.min(), right_y.max() + 1
            if (
                left_x1 <= right_x2 + 3
                and right_x1 <= left_x2 + 3
                and left_y1 <= right_y2 + 3
                and right_y1 <= left_y2 + 3
            ):
                return True
    return False


def _merge_adjacent_bar_shades(series: list[_PixelSeries]) -> list[_PixelSeries]:
    merged = list(series)
    candidate_index = 0
    while candidate_index < len(merged):
        other_index = candidate_index + 1
        while other_index < len(merged):
            left = merged[candidate_index]
            right = merged[other_index]
            should_merge = (
                _hue_distance(left.mean_rgb, right.mean_rgb) <= 8
                and _lab_distance(left.mean_rgb, right.mean_rgb) <= 92.0
                and _masks_are_adjacent(left.masks, right.masks)
            )
            if not should_merge:
                other_index += 1
                continue

            total_area = left.area + right.area
            left.mean_rgb = (
                left.mean_rgb * left.area + right.mean_rgb * right.area
            ) / total_area
            left.area = total_area
            left.masks.extend(right.masks)
            del merged[other_index]
        candidate_index += 1
    return merged


def _build_pie_color_series(
    arr: np.ndarray,
    valid: np.ndarray,
    min_data_area: int,
    max_data_area: int,
) -> list[_PixelSeries]:
    """Build one series per visible pie slice instead of per hue family.

    Pie charts often deliberately use neighbouring shades of the same colour.
    Grouping pixels by a wide hue bin therefore joins several slices and gives
    them a single pattern.  Vector charts also usually retain a solid fill
    colour for each slice, so use those dominant colours as independent seeds.
    """
    import cv2

    height, width = valid.shape
    pixels = arr[valid]
    if not len(pixels):
        return []

    colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
    # A seed may be smaller than the final accepted slice because anti-aliased
    # borders use neighbouring RGB values.  Keep the threshold conservative.
    min_seed_area = max(25, min_data_area // 4)
    seed_indexes = np.flatnonzero(counts >= min_seed_area)
    series: list[_PixelSeries] = []

    for seed_index in seed_indexes[np.argsort(counts[seed_indexes])[::-1]]:
        seed = colors[seed_index]
        color_mask = valid & np.all(arr == seed, axis=2)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            color_mask.astype(np.uint8), connectivity=4
        )
        for label_id in range(1, labels_count):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if not min_data_area <= area <= max_data_area:
                continue
            component = labels == label_id
            mean_rgb = np.median(arr[component], axis=0).astype(np.float32)
            series.append(_PixelSeries(mean_rgb=mean_rgb, area=area, masks=[component]))

    # Percentage labels can split one uniformly coloured slice into two
    # connected components.  Rejoin only fragments of the same colour whose
    # bounding boxes meet; separated slices retain their own pattern.
    merged: list[_PixelSeries] = []
    for candidate in series:
        matching = next(
            (
                existing
                for existing in merged
                if _lab_distance(existing.mean_rgb, candidate.mean_rgb) <= 3.0
                and _masks_are_adjacent(existing.masks, candidate.masks)
            ),
            None,
        )
        if matching is None:
            merged.append(candidate)
            continue
        total_area = matching.area + candidate.area
        matching.mean_rgb = (
            matching.mean_rgb * matching.area + candidate.mean_rgb * candidate.area
        ) / total_area
        matching.area = total_area
        matching.masks.extend(candidate.masks)

    # Thin white separators and percentage labels can survive the circular
    # pie mask.  They are not a chart series, and one such false series would
    # prevent a six-item legend from being recognised as complete.
    near_white_limit = max(min_data_area, int(height * width * 0.02))
    merged = [
        candidate
        for candidate in merged
        if not (
            candidate.mean_rgb.min() >= 242
            and candidate.area <= near_white_limit
        )
    ]

    # Raster charts may not have sufficiently frequent exact RGB values.  The
    # caller falls back to the regular colour grouping in that case.
    return merged


def _assign_patterns_by_color(series: list[_PixelSeries]) -> None:
    """Give visually identical series the same pattern everywhere on a canvas.

    A dashboard can contain several charts that reuse the same categorical
    palette.  Those occurrences must remain separate masks, but they are one
    visual category and therefore need the same hatch.  Pattern assignment by
    component order made this depend on chart position.
    """
    clusters: list[dict[str, object]] = []
    for candidate in sorted(series, key=lambda item: _hue_sort_key(item.mean_rgb)):
        matching_cluster: dict[str, object] | None = None
        for cluster in clusters:
            cluster_rgb = cluster["mean_rgb"]
            assert isinstance(cluster_rgb, np.ndarray)
            neutral_pair = (
                max(candidate.mean_rgb) - min(candidate.mean_rgb) < 35
                and max(cluster_rgb) - min(cluster_rgb) < 35
            )
            if (
                _lab_distance(candidate.mean_rgb, cluster_rgb)
                <= (14.0 if neutral_pair else 18.0)
                and (neutral_pair or _hue_distance(candidate.mean_rgb, cluster_rgb) <= 4)
            ):
                matching_cluster = cluster
                break

        if matching_cluster is None:
            clusters.append(
                {
                    "mean_rgb": candidate.mean_rgb.copy(),
                    "area": candidate.area,
                    "members": [candidate],
                }
            )
            continue

        cluster_area = int(matching_cluster["area"])
        cluster_rgb = matching_cluster["mean_rgb"]
        assert isinstance(cluster_rgb, np.ndarray)
        matching_cluster["mean_rgb"] = (
            cluster_rgb * cluster_area + candidate.mean_rgb * candidate.area
        ) / (cluster_area + candidate.area)
        matching_cluster["area"] = cluster_area + candidate.area
        members = matching_cluster["members"]
        assert isinstance(members, list)
        members.append(candidate)

    for series_id, cluster in enumerate(
        sorted(clusters, key=lambda item: _hue_sort_key(item["mean_rgb"]))
    ):
        spec = _pattern_spec_for_series(series_id)
        members = cluster["members"]
        assert isinstance(members, list)
        for candidate in members:
            candidate.family = spec.family
            candidate.angle_rad = np.deg2rad(spec.angle_degrees)
            candidate.spacing_scale = spec.spacing_scale


def _build_color_series(
    arr: np.ndarray,
    valid: np.ndarray,
    color_distance_threshold: float = 42.0,
    min_data_fraction: float = 0.0008,
    is_pie: bool = False,
) -> list[_PixelSeries]:
    import cv2

    h_img, w_img = valid.shape
    min_data_area = max(50, int(h_img * w_img * min_data_fraction))
    max_data_area = int(h_img * w_img * 0.45)

    if is_pie:
        # The chart can occupy only part of the canvas, so an otherwise valid
        # 1–2% slice would be lost when measured against the whole image.
        pie_min_data_area = max(100, min_data_area // 3)
        pie_series = _build_pie_color_series(
            arr,
            valid,
            pie_min_data_area,
            max_data_area,
        )
        if len(pie_series) >= 2:
            _assign_patterns_by_color(pie_series)
            return pie_series

    components: list[tuple[np.ndarray, np.ndarray, int]] = []

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hue_bins = hsv[:, :, 0] // 6
    for hue_bin in np.unique(hue_bins[valid]):
        color_mask = (valid & (hue_bins == hue_bin)).astype(np.uint8)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            color_mask, connectivity=4
        )
        for label_id in range(1, labels_count):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if not min_data_area <= area <= max_data_area:
                continue
            # One- or two-pixel anti-aliased borders of bars can be long
            # enough to pass the area threshold, but they are not a series.
            if not is_pie and min(width, height) <= 2:
                continue
            if width * height > h_img * w_img * 0.75 and area > h_img * w_img * 0.05:
                continue
            if width > w_img * 0.55 and height > h_img * 0.30:
                continue
            if (
                not is_pie
                and width >= w_img * 0.25
                and height >= h_img * 0.25
                and area / (width * height) < 0.4
            ):
                continue
            mask = labels == label_id
            mean_rgb = np.median(arr[mask], axis=0).astype(np.float32)
            is_neutral = float(mean_rgb.max() - mean_rgb.min()) < 25.0
            touches_edge = x == 0 or y == 0 or x + width >= w_img or y + height >= h_img
            if is_neutral and touches_edge:
                continue
            components.append((mask, mean_rgb, area))

    series: list[_PixelSeries] = []
    for mask, mean_rgb, area in sorted(components, key=lambda item: -item[2]):
        closest = min(
            series,
            key=lambda candidate: _lab_distance(mean_rgb, candidate.mean_rgb),
            default=None,
        )
        if closest is None or _lab_distance(mean_rgb, closest.mean_rgb) > color_distance_threshold:
            closest = _PixelSeries(mean_rgb=mean_rgb.copy(), area=0)
            series.append(closest)
        total_area = closest.area + area
        closest.mean_rgb = (closest.mean_rgb * closest.area + mean_rgb * area) / total_area
        closest.area = total_area
        closest.masks.append(mask)

    if not is_pie:
        series = _merge_adjacent_bar_shades(series)

    _assign_patterns_by_color(series)

    return series


def _draw_series_patterns(
    arr: np.ndarray,
    series: list[_PixelSeries],
    opacity: int,
    protected: np.ndarray,
    spacing: float,
) -> np.ndarray:
    import cv2

    h_img, w_img = arr.shape[:2]
    yy, xx = np.mgrid[0:h_img, 0:w_img].astype(np.float32)
    result = arr.astype(np.float32)
    alpha = min(1.0, max(0.0, opacity / 100.0))

    for candidate in series:
        bold = np.zeros((h_img, w_img), dtype=bool)
        pattern = _draw_pattern(
            candidate.family,
            xx,
            yy,
            candidate.angle_rad,
            bold,
            spacing=spacing * candidate.spacing_scale,
        )
        ink_rgb = _pattern_ink_color(candidate.mean_rgb)
        for mask in candidate.masks:
            draw_here = mask & pattern & ~protected
            # A very thin pie slice can fall entirely between global pattern
            # strokes.  Give it a compact contrast cue at its visual centre
            # so every represented series remains perceptible.
            if (
                draw_here.sum() < 16
                and mask.sum() <= h_img * w_img * 0.02
            ):
                available = (mask & ~protected).astype(np.uint8)
                distance = cv2.distanceTransform(available, cv2.DIST_L2, 5)
                if distance.max() > 0:
                    center_y, center_x = np.unravel_index(distance.argmax(), distance.shape)
                    radius = 2.0 if distance.max() >= 2.5 else 1.5
                    draw_here = available.astype(bool) & (
                        (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
                    )
            result[draw_here] = result[draw_here] * (1 - alpha) + ink_rgb * alpha

    return result


def _pattern_ink_color(background_rgb: np.ndarray) -> np.ndarray:
    srgb = np.clip(background_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    luminance = float(np.dot(linear, np.array([0.2126, 0.7152, 0.0722])))
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    if white_contrast > black_contrast:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def _legend_match_score(marker_rgb: np.ndarray, series: _PixelSeries) -> float:
    marker_hue, marker_saturation, _ = _hue_sort_key(marker_rgb)
    series_hue, series_saturation, _ = _hue_sort_key(series.mean_rgb)
    lab_distance = _lab_distance(marker_rgb, series.mean_rgb)
    if min(marker_saturation, series_saturation) < 60:
        return lab_distance

    hue_distance = min(abs(marker_hue - series_hue), 180 - abs(marker_hue - series_hue))
    return hue_distance * 20.0 + lab_distance * 0.5


def _assign_legend_series(
    arr: np.ndarray,
    marker_items: list[tuple[np.ndarray, int, int, int, int]],
    marker_indexes: list[int],
    series: list[_PixelSeries],
) -> list[tuple[int, int]]:
    marker_colors = [
        np.median(arr[marker_items[index][0]], axis=0).astype(np.float32)
        for index in marker_indexes
    ]
    if len(marker_indexes) <= len(series) <= 12:
        score_matrix = tuple(
            tuple(_legend_match_score(marker_rgb, candidate) for candidate in series)
            for marker_rgb in marker_colors
        )

        @lru_cache(maxsize=None)
        def find_minimum(marker_offset: int, used_series: int) -> tuple[float, tuple[int, ...]]:
            if marker_offset == len(marker_indexes):
                return 0.0, ()

            best_score = float("inf")
            best_assignment: tuple[int, ...] = ()
            for series_index in range(len(series)):
                if used_series & (1 << series_index):
                    continue
                remaining_score, remaining_assignment = find_minimum(
                    marker_offset + 1,
                    used_series | (1 << series_index),
                )
                total_score = score_matrix[marker_offset][series_index] + remaining_score
                if total_score < best_score:
                    best_score = total_score
                    best_assignment = (series_index,) + remaining_assignment
            return best_score, best_assignment

        _, assigned_series = find_minimum(0, 0)
        return list(zip(marker_indexes, assigned_series))

    # Several charts on one canvas can repeat the same legend.  Match every
    # marker independently in that case so all copies receive a pattern.
    return [
        (
            marker_index,
            min(
                range(len(series)),
                key=lambda series_index: _legend_match_score(
                    marker_rgb, series[series_index]
                ),
            ),
        )
        for marker_index, marker_rgb in zip(marker_indexes, marker_colors)
    ]


def _marker_has_adjacent_label(
    dark_text: np.ndarray,
    x: int,
    y: int,
    marker_width: int,
    marker_height: int,
) -> bool:
    """Return whether a colour swatch has neutral text beside it.

    Decorative coloured headings otherwise look like a horizontal legend to
    the marker detector and receive hatches letter-by-letter.
    """
    height, width = dark_text.shape
    pad_x = max(18, marker_width * 3)
    y1 = max(0, y - 3)
    y2 = min(height, y + marker_height + 4)
    left = dark_text[y1:y2, max(0, x - pad_x):x].sum()
    right = dark_text[y1:y2, x + marker_width:min(width, x + marker_width + pad_x)].sum()
    return bool(left + right >= 5)


def _safe_in_place_badge_bounds(
    marker_items: list[tuple[np.ndarray, int, int, int, int]],
    marker_indexes: list[int],
    marker_index: int,
    dark_text: np.ndarray,
    neutral_background: np.ndarray,
    target_size: int,
) -> tuple[int, int, int, int] | None:
    """Return a safe, square target box for one legend swatch.

    The caller chooses one shared size for the whole legend.  Returning
    ``None`` is intentional: a legend is left untouched rather than making
    its markers different sizes when one row has insufficient space.
    """
    height, width = dark_text.shape
    marker_mask, x, y, marker_width, marker_height = marker_items[marker_index]
    center_y = y + marker_height / 2
    source_box = np.zeros((height, width), dtype=bool)
    source_box[y:y + marker_height, x:x + marker_width] = True
    occupied = np.zeros((height, width), dtype=bool)
    for other_index in marker_indexes:
        if other_index == marker_index:
            continue
        _, other_x, other_y, other_width, other_height = marker_items[other_index]
        occupied[
            other_y:other_y + other_height,
            other_x:other_x + other_width,
        ] = True

    left_ink = dark_text[
        max(0, y - 2):min(height, y + marker_height + 2),
        max(0, x - marker_width):x,
    ].sum()
    right_ink = dark_text[
        max(0, y - 2):min(height, y + marker_height + 2),
        min(width, x + marker_width):min(width, x + 2 * marker_width),
    ].sum()
    label_on_right = right_ink >= left_ink

    badge_width = badge_height = target_size
    if label_on_right:
        x2 = x + marker_width
        x1 = x2 - badge_width
    else:
        x1 = x
        x2 = x1 + badge_width
    y1 = int(round(center_y - badge_height / 2))
    y2 = y1 + badge_height
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return None

    badge_area = np.zeros((height, width), dtype=bool)
    badge_area[y1:y2, x1:x2] = True
    if (badge_area & occupied).any():
        return None
    # A badge is repainted as a whole.  Partial repainting of gradients or
    # text anti-aliasing was the reason several legend cells appeared broken.
    # A few dark pixels can belong to the original tiny hatch, whose coloured
    # component is smaller than its complete visual cell.  Treat that as part
    # of the cell; a real label occupies materially more of the candidate box.
    if (badge_area & dark_text & ~source_box).mean() > 0.04:
        return None
    # Chart exports often anti-alias the icon edge into a few non-neutral
    # pixels.  The candidate grows away from the label and is repainted as a
    # whole, so those edge pixels are safe; actual text and other badges were
    # rejected above.
    if (neutral_background | source_box)[badge_area].mean() < 0.90:
        return None
    return x1, y1, badge_width, badge_height


def _uniform_legend_badge_bounds(
    marker_items: list[tuple[np.ndarray, int, int, int, int]],
    marker_indexes: list[int],
    dark_text: np.ndarray,
    neutral_background: np.ndarray,
) -> dict[int, tuple[int, int, int, int]] | None:
    """Find one safe square size that works for every item in a legend."""
    largest_original = max(
        max(marker_items[index][3], marker_items[index][4])
        for index in marker_indexes
    )
    maximum_size = max(_LEGEND_SWATCH_TARGET_SIZE, largest_original)
    for target_size in range(maximum_size, largest_original - 1, -1):
        bounds: dict[int, tuple[int, int, int, int]] = {}
        for marker_index in marker_indexes:
            candidate_bounds = _safe_in_place_badge_bounds(
                marker_items,
                marker_indexes,
                marker_index,
                dark_text,
                neutral_background,
                target_size,
            )
            if candidate_bounds is None:
                break
            bounds[marker_index] = candidate_bounds
        if len(bounds) != len(marker_indexes):
            continue

        # Every candidate was checked against the original marker boxes.
        # Check enlarged boxes against each other as well, otherwise two
        # adjacent legend rows can overwrite one another.
        boxes = list(bounds.values())
        if any(
            left_x < right_x + right_width
            and right_x < left_x + left_width
            and left_y < right_y + right_height
            and right_y < left_y + left_height
            for offset, (left_x, left_y, left_width, left_height) in enumerate(boxes)
            for right_x, right_y, right_width, right_height in boxes[offset + 1:]
        ):
            continue
        return bounds
    return None


def _draw_legend_patterns(
    arr: np.ndarray,
    result: np.ndarray,
    series: list[_PixelSeries],
    opacity: int,
    is_pie: bool,
) -> np.ndarray:
    import cv2

    if not series:
        return result

    height, width = arr.shape[:2]
    arr_lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    arr_hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    channel_span = arr.max(axis=2) - arr.min(axis=2)
    dark_text = (arr.max(axis=2) <= 150) & (channel_span <= 35)
    neutral_background = (arr.max(axis=2) >= 130) & (channel_span <= 30)
    # A bar chart has large labelled rectangles that must not be mistaken for
    # legend swatches.  Pie/donut charts have no bars, while their legends may
    # use deliberately large colour keys, so use a chart-type-aware geometric
    # bound instead of one global cap.
    max_marker_size = (
        max(16, min(96, int(min(height, width) * 0.12)))
        if is_pie
        else max(16, min(64, int(min(height, width) * 0.08)))
    )
    data_mask = np.zeros((height, width), dtype=np.uint8)
    for candidate in series:
        for mask in candidate.masks:
            mask_y, mask_x = np.where(mask)
            if not len(mask_x):
                continue
            mask_x1, mask_x2 = int(mask_x.min()), int(mask_x.max()) + 1
            mask_y1, mask_y2 = int(mask_y.min()), int(mask_y.max()) + 1
            mask_width = mask_x2 - mask_x1
            mask_height = mask_y2 - mask_y1
            # Do not put already-found compact legend swatches into the chart
            # exclusion mask.  Otherwise the true 2017/2018/2019 squares are
            # hidden from marker detection and fragments of their text are
            # mistaken for the legend instead.
            is_labelled_swatch = (
                4 <= mask_width <= max_marker_size
                and 4 <= mask_height <= max_marker_size
                and 0.35 <= mask_width / mask_height <= 4.0
                and _marker_has_adjacent_label(
                    dark_text, mask_x1, mask_y1, mask_width, mask_height
                )
            )
            if is_labelled_swatch:
                continue
            data_mask |= mask.astype(np.uint8)
    margin = max(15, min(35, int(round(min(height, width) * 0.025))))
    margin += 1 - margin % 2
    data_mask = cv2.dilate(
        data_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin)),
    ).astype(bool)
    markers: dict[tuple[int, int, int, int], tuple[np.ndarray, int, int, int, int]] = {}

    for candidate in series:
        marker = np.array([[candidate.mean_rgb.astype(np.uint8)]], dtype=np.uint8)
        marker_lab = cv2.cvtColor(marker, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        marker_hue, marker_saturation, marker_value = _hue_sort_key(candidate.mean_rgb)
        hue_delta = np.abs(arr_hsv[:, :, 0].astype(np.int16) - marker_hue)
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        hue_match = (
            (hue_delta <= 8)
            & (arr_hsv[:, :, 1] >= 70)
            & (np.abs(arr_hsv[:, :, 1].astype(np.int16) - marker_saturation) <= 60)
            & (np.abs(arr_hsv[:, :, 2].astype(np.int16) - marker_value) <= 65)
        )
        # For neutral swatches, a wide Lab radius also catches the white
        # holes inside black digits (for example the zero in "2019").  The
        # actual grey marker is close in RGB, so use a narrower neutral band.
        lab_radius = 18.0 if marker_saturation < 60 else 32.0
        close_color = (
            np.linalg.norm(arr_lab - marker_lab, axis=2) <= lab_radius
        ) | hue_match
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            close_color.astype(np.uint8), connectivity=4
        )
        for label_id in range(1, labels_count):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            marker_width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            marker_height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if not (
                4 <= marker_width <= max_marker_size
                and 4 <= marker_height <= max_marker_size
                and area >= 12
            ):
                continue
            if not 0.35 <= marker_width / marker_height <= 4.0:
                continue
            fill_ratio = area / (marker_width * marker_height)
            # Pie-chart legends often use small circular swatches with an
            # existing hatch.  Their coloured pixels cover only about half of
            # the box, unlike the rectangular swatches handled by the stricter
            # rule.  Keep the complete 8+ px icon instead of treating its
            # individual hatch fragments as separate markers.
            compact_icon = (
                min(marker_width, marker_height) >= 8
                and fill_ratio >= 0.45
            )
            if fill_ratio < 0.7 and not compact_icon:
                continue
            if data_mask[y:y + marker_height, x:x + marker_width].any():
                continue
            marker_key = (x, y, marker_width, marker_height)
            markers.setdefault(
                marker_key,
                (labels == label_id, x, y, marker_width, marker_height),
            )

    marker_items = []
    for item in sorted(markers.values(), key=lambda candidate: -int(candidate[0].sum())):
        marker_mask = item[0]
        _, x, y, marker_width, marker_height = item
        center_x = x + marker_width / 2
        center_y = y + marker_height / 2
        if any(
            (marker_mask & existing[0]).sum()
            / min(marker_mask.sum(), existing[0].sum())
            >= 0.8
            for existing in marker_items
        ):
            continue
        # A patterned circular swatch can be discovered once as its complete
        # coloured outline and again as a small solid hatch fragment.  They
        # need to behave as one marker; otherwise an otherwise complete
        # legend looks like it has more entries than chart series and cannot
        # be safely enlarged.
        if any(
            abs(center_x - (existing[1] + existing[3] / 2)) <= 4
            and abs(center_y - (existing[2] + existing[4] / 2)) <= 4
            for existing in marker_items
        ):
            continue
        marker_items.append(item)

    if len(marker_items) < 2:
        return result

    positions = [
        (x + marker_width / 2, y + marker_height / 2)
        for _, x, y, marker_width, marker_height in marker_items
    ]
    legend_marker_indexes = []
    for marker_index, (_, x, y, marker_width, marker_height) in enumerate(marker_items):
        if not _marker_has_adjacent_label(
            dark_text, x, y, marker_width, marker_height
        ):
            continue
        center_x = x + marker_width / 2
        center_y = y + marker_height / 2
        peers_in_column = sum(abs(center_x - peer_x) <= 8 for peer_x, _ in positions)
        peers_in_row = sum(abs(center_y - peer_y) <= 8 for _, peer_y in positions)
        if max(peers_in_column, peers_in_row) >= 2:
            legend_marker_indexes.append(marker_index)

    if len(legend_marker_indexes) < 2:
        return result

    assignments = _assign_legend_series(
        arr,
        marker_items,
        legend_marker_indexes,
        series,
    )
    uniform_badge_bounds = _uniform_legend_badge_bounds(
        marker_items,
        legend_marker_indexes,
        dark_text,
        neutral_background,
    )

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    alpha = min(1.0, max(0.0, opacity / 100.0))

    for marker_index, series_index in assignments:
        candidate = series[series_index]
        marker_mask, x, y, marker_width, marker_height = marker_items[marker_index]
        if uniform_badge_bounds is None:
            x1, y1, badge_width, badge_height = x, y, marker_width, marker_height
        else:
            x1, y1, badge_width, badge_height = uniform_badge_bounds[marker_index]
        x2 = x1 + badge_width
        y2 = y1 + badge_height

        badge_area = np.zeros((height, width), dtype=bool)
        badge_area[y1:y2, x1:x2] = True
        badge_mask = badge_area if uniform_badge_bounds is not None else marker_mask

        marker_rgb = np.median(arr[marker_mask], axis=0).astype(np.float32)
        result[badge_mask] = marker_rgb
        # The square grows, but the pattern does not: keeping the original
        # cadence makes additional repeats visible instead of magnifying a
        # single blurry motif.
        spacing = float(np.clip(min(marker_width, marker_height) * 0.5, 5.0, 12.0))
        pattern = _draw_pattern(
            candidate.family,
            xx - x1,
            yy - y1,
            candidate.angle_rad,
            np.zeros((height, width), dtype=bool),
            spacing=spacing * candidate.spacing_scale,
        )
        draw_here = badge_mask & pattern
        ink_rgb = _pattern_ink_color(candidate.mean_rgb)
        result[draw_here] = result[draw_here] * (1 - alpha) + ink_rgb * alpha

    return result


def _smooth_shadows(arr_rgb: np.ndarray) -> np.ndarray:
    import cv2

    median = cv2.medianBlur(arr_rgb, 5)
    return cv2.bilateralFilter(median, d=15, sigmaColor=80, sigmaSpace=80)


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_lab(rgb):
    lin = _srgb_to_linear(rgb)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T
    t = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _simulate_cvd(rgb, kind):
    lin = _srgb_to_linear(rgb) @ _CVD_MATRICES[kind].T
    lin = np.clip(lin, 0, 1)
    srgb = np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * lin ** (1 / 2.4) - 0.055) * 255
    return srgb


def _confusable_pairs(colors) -> set:
    bad = set()
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            for kind in _CVD_MATRICES:
                a = _to_lab(_simulate_cvd(colors[i][:3], kind))
                b = _to_lab(_simulate_cvd(colors[j][:3], kind))
                if float(np.linalg.norm(a - b)) < _CONFUSABLE_DE:
                    bad.add((i, j))
                    break
    return bad


def _draw_marker(canvas: np.ndarray, cx: int, cy: int, shape: str,
                 size: int, color: tuple, outline: tuple | None = None) -> None:
    import cv2
    r = size // 2
    c = tuple(int(v) for v in color)
    if outline is not None:
        _draw_marker(canvas, cx, cy, shape, size + 3, outline, None)

    if shape == "circle":
        cv2.circle(canvas, (cx, cy), r, c, -1)
    elif shape == "square":
        cv2.rectangle(canvas, (cx - r, cy - r), (cx + r, cy + r), c, -1)
    elif shape == "triangle":
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillPoly(canvas, [pts], c)
    elif shape == "diamond":
        pts = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], np.int32)
        cv2.fillPoly(canvas, [pts], c)
    elif shape == "cross":
        t = max(1, r // 2)
        cv2.rectangle(canvas, (cx - r, cy - t), (cx + r, cy + t), c, -1)
        cv2.rectangle(canvas, (cx - t, cy - r), (cx + t, cy + r), c, -1)
    elif shape == "star":
        cv2.circle(canvas, (cx, cy), r, c, -1)
        t = max(1, r // 3)
        cv2.rectangle(canvas, (cx - r - 1, cy - t), (cx + r + 1, cy + t), c, -1)


def _cluster_line_colors(arr: np.ndarray, lines_mask: np.ndarray,
                         max_k: int = 8, min_share: float = 0.04) -> list:
    import cv2

    px = arr[lines_mask].astype(np.float32)
    if len(px) < 50:
        return []

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(px, max_k, None, crit, 10, cv2.KMEANS_PP_CENTERS)

    counts = np.bincount(labels.flatten(), minlength=max_k)
    total = counts.sum()

    MIN_BLOB = 60
    COLOR_RADIUS = 45.0

    keep = []
    for i in range(max_k):
        if counts[i] / total < min_share:
            continue

        c = centers[i]
        dist = np.linalg.norm(arr.astype(np.float32) - c, axis=2)
        color_mask = lines_mask & (dist < COLOR_RADIUS)
        if color_mask.sum() < 50:
            continue

        n_cc, _, st_cc, _ = cv2.connectedComponentsWithStats(color_mask.astype(np.uint8))
        largest = max((st_cc[j, cv2.CC_STAT_AREA] for j in range(1, n_cc)), default=0)
        if largest < MIN_BLOB:
            continue

        keep.append((counts[i], c))

    keep.sort(key=lambda t: -t[0])
    distinct_colors: list[np.ndarray] = []
    for _, candidate_color in keep:
        if all(
            np.linalg.norm(candidate_color - existing_color) > 30
            for existing_color in distinct_colors
        ):
            distinct_colors.append(candidate_color)
    return distinct_colors


def _background_color(arr: np.ndarray) -> tuple:
    edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    vals, counts = np.unique(edges.reshape(-1, 3), axis=0, return_counts=True)
    return tuple(int(v) for v in vals[counts.argmax()])


def _is_ink(c, bg=(255, 255, 255)) -> bool:
    return max(abs(int(c[i]) - int(bg[i])) for i in range(3)) > 30


def _scan_runs(arr: np.ndarray, minlen: int, maxlen: int, tol: int,
               bg=(255, 255, 255)) -> list:
    h, w, _ = arr.shape
    runs = []
    for y in range(h):
        row = arr[y]
        x = 0
        while x < w:
            c = row[x]
            x0 = x
            while x < w and np.abs(row[x] - c).max() <= tol:
                x += 1
            L = x - x0
            if minlen <= L <= maxlen:
                ct = tuple(int(v) for v in c)
                if _is_ink(ct, bg):
                    runs.append((y, x0, L, ct))
            if x == x0:
                x += 1
    return runs


def _stack_runs(runs: list, maxh: int) -> list:
    by = defaultdict(list)
    for y, x0, L, c in runs:
        by[(x0, L, c)].append(y)
    out = []
    for (x0, L, c), ys in by.items():
        ys = sorted(ys)
        run = [ys[0]]
        for a, b in zip(ys, ys[1:]):
            if b - a <= 1:
                run.append(b)
            else:
                out.append(dict(x=x0, len=L, h=len(run), y=int(np.mean(run)), rgb=c))
                run = [b]
        out.append(dict(x=x0, len=L, h=len(run), y=int(np.mean(run)), rgb=c))
    return [e for e in out if e["h"] <= maxh]


def _distinct(colors, thr=30):
    keep = []
    for c in colors:
        if all(max(abs(c[i] - k[i]) for i in range(3)) > thr for k in keep):
            keep.append(c)
    return keep


def find_legend_handles(arr: np.ndarray,
                        minlen: int = 8, maxlen: int = 60,
                        tol: int = 12, maxh: int = 12,
                        min_series: int = 3) -> list:
    arr = np.asarray(arr, dtype=np.int32)
    bg = _background_color(arr)
    H = _stack_runs(_scan_runs(arr, minlen, maxlen, tol, bg), maxh)
    if not H:
        return []

    groups = defaultdict(list)
    for e in H:
        for dl in (-1, 0, 1):
            groups[(e["len"] + dl, e["h"])].append(e)
    for k in groups:
        seen, uniq = set(), []
        for e in groups[k]:
            sig = (e["x"], e["y"], e["len"], e["rgb"])
            if sig not in seen:
                seen.add(sig); uniq.append(e)
        groups[k] = uniq

    best = None
    for (L, hh), v in groups.items():
        cols = _distinct({e["rgb"] for e in v})
        if len(cols) < min_series:
            continue
        ys = [e["y"] for e in v]
        xs = [e["x"] for e in v]
        is_row = (max(ys) - min(ys)) <= 3
        is_col = len(set(xs)) <= 2 and len(set(ys)) >= min_series
        if not (is_row or is_col):
            continue
        seen, uniq = [], []
        for e in sorted(v, key=lambda e: (e["y"], e["x"])):
            if all(max(abs(e["rgb"][i] - s[i]) for i in range(3)) > 30 for s in seen):
                seen.append(e["rgb"])
                uniq.append(e)
        if best is None or len(uniq) > len(best):
            best = uniq
    return best or []


def _find_legend_handles_for_colors(
    arr: np.ndarray,
    line_colors: list[np.ndarray],
) -> list[dict]:
    import cv2

    height, width = arr.shape[:2]
    dark_text = arr.max(axis=2) <= 145
    handles: list[dict] = []

    for line_color in line_colors:
        color_distance = np.linalg.norm(
            arr.astype(np.float32) - line_color.astype(np.float32), axis=2
        )
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (color_distance <= 45).astype(np.uint8), connectivity=4
        )
        best_candidate: tuple[int, int, int, int, int, int] | None = None

        for label_id in range(1, labels_count):
            x, y, handle_width, handle_height, area = stats[label_id]
            if not (
                3 <= handle_width <= 70
                and 1 <= handle_height <= 18
                and area >= 3
            ):
                continue

            y1 = max(0, y - max(5, handle_height))
            y2 = min(height, y + handle_height + max(5, handle_height))
            x1 = min(width, x + handle_width)
            x2 = min(width, x1 + max(28, handle_width * 3))
            label_pixels = int(dark_text[y1:y2, x1:x2].sum())
            if label_pixels < 12:
                continue

            candidate = (label_pixels, int(x), int(y), int(handle_width), int(handle_height), int(area))
            if best_candidate is None or candidate[0] > best_candidate[0]:
                best_candidate = candidate

        if best_candidate is None:
            continue

        _, x, y, handle_width, handle_height, _ = best_candidate
        handles.append(
            {
                "x": x,
                "len": handle_width,
                "h": handle_height,
                "y": y + handle_height // 2,
                "rgb": tuple(int(value) for value in line_color),
            }
        )

    return sorted(handles, key=lambda handle: (handle["y"], handle["x"]))


def _on_own_line(arr: np.ndarray, x: int, y: int, centers: np.ndarray,
                 ci: int, tol: float) -> bool:
    h, w = arr.shape[:2]
    if not (0 <= y < h and 0 <= x < w):
        return False
    px = arr[y, x].astype(np.float32)
    d = np.linalg.norm(centers - px, axis=1)
    if int(d.argmin()) != ci:
        return False
    return float(d[ci]) < tol


def _collides(occupied: np.ndarray, cx: int, cy: int, size: int) -> bool:
    h, w = occupied.shape
    r = size // 2 + 2
    y1, y2 = max(0, cy - r), min(h, cy + r + 1)
    x1, x2 = max(0, cx - r), min(w, cx + r + 1)
    return bool(occupied[y1:y2, x1:x2].any())


def _mark_occupied(occupied: np.ndarray, cx: int, cy: int, size: int) -> None:
    h, w = occupied.shape
    r = size // 2 + 2
    occupied[max(0, cy - r):min(h, cy + r + 1),
             max(0, cx - r):min(w, cx + r + 1)] = True


def _marker_colors(series_rgb: np.ndarray) -> tuple:
    r, g, b = [float(v) for v in series_rgb[:3]]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    fill = (int(r), int(g), int(b))
    outline = (255, 255, 255) if lum < 90 else (0, 0, 0)
    return fill, outline


def _place_markers_on_lines(
    result: np.ndarray,
    lines_mask: np.ndarray,
    arr: np.ndarray,
    marker_size: int = 9,
    step: int = 60,
    legend_handles: list | None = None,
) -> np.ndarray:
    import cv2

    clustered_colors = _cluster_line_colors(arr, lines_mask)
    inferred_handles = _find_legend_handles_for_colors(arr, clustered_colors)
    if len(inferred_handles) == len(clustered_colors):
        handles = inferred_handles
    elif legend_handles and len(legend_handles) >= len(clustered_colors):
        handles = legend_handles
    else:
        handles = []

    line_colors = (
        [np.array(handle["rgb"], dtype=np.float32) for handle in handles]
        if handles
        else clustered_colors
    )
    if not line_colors:
        return result

    ys, xs = np.where(lines_mask)
    if len(ys) == 0:
        return result

    occupied = np.zeros(lines_mask.shape, dtype=bool)

    pixels = arr[ys, xs].astype(np.float32)

    centers = np.array(line_colors, dtype=np.float32)
    dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
    assign = dists.argmin(axis=1)
    best_dist = dists.min(axis=1)

    if len(centers) > 1:
        pair = [float(np.linalg.norm(centers[i] - centers[j]))
                for i in range(len(centers)) for j in range(i + 1, len(centers))]
        own_tol = min(60.0, min(pair) * 0.5)
    else:
        own_tol = 60.0
    MAX_COLOR_DIST = own_tol
    on_line = best_dist < MAX_COLOR_DIST

    series_mask = []
    for ci in range(len(line_colors)):
        m = np.zeros(lines_mask.shape, dtype=bool)
        sel_i = (assign == ci) & on_line
        m[ys[sel_i], xs[sel_i]] = True
        series_mask.append(m)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dil = [cv2.dilate(m.astype(np.uint8), kern).astype(bool) for m in series_mask]
    risky = _confusable_pairs([tuple(int(v) for v in c) for c in centers])

    crossings = []
    for i in range(len(series_mask)):
        for j in range(i + 1, len(series_mask)):
            if (i, j) not in risky:
                continue
            both = dil[i] & dil[j] & (series_mask[i] | series_mask[j])
            n_cc, _, st_cc, cen_cc = cv2.connectedComponentsWithStats(
                both.astype(np.uint8), 8)
            for q in range(1, n_cc):
                if st_cc[q, cv2.CC_STAT_AREA] >= 12:
                    crossings.append((int(cen_cc[q][0]), int(cen_cc[q][1]), i, j))

    offset = marker_size + 4
    keep_out = marker_size

    def _on_crossing(cx, cy):
        return any(abs(cx - qx) < keep_out and abs(cy - qy) < keep_out
                   for qx, qy, _, _ in crossings)

    foreign_px = []
    for ci_ in range(len(centers)):
        m = (np.abs(arr.astype(np.int32) - centers[ci_].astype(np.int32)
                    ).max(2) < 30)
        foreign_px.append(m)

    def _covers_foreign(ci, cx, cy, size, limit=6):
        r = size // 2
        y1, y2 = max(0, cy - r), min(foreign_px[0].shape[0], cy + r + 1)
        x1, x2 = max(0, cx - r), min(foreign_px[0].shape[1], cx + r + 1)
        foreign = sum(int(foreign_px[si][y1:y2, x1:x2].sum())
                      for si in range(len(foreign_px)) if si != ci)
        own = int(foreign_px[ci][y1:y2, x1:x2].sum())
        return foreign > own

    for ci, h in enumerate(handles):
        shape = _MARKER_SHAPES[ci % len(_MARKER_SHAPES)]
        fill, outline = _marker_colors(np.array(h["rgb"], dtype=np.float32))
        cx = h["x"] + h["len"] // 2
        _draw_marker(result, cx, h["y"], shape, marker_size + 2, fill, outline)
        _mark_occupied(occupied, cx, h["y"], marker_size + 2)

    for ci in range(len(line_colors)):
        sel = (assign == ci) & on_line
        if sel.sum() < 50:
            continue
        shape = _MARKER_SHAPES[ci % len(_MARKER_SHAPES)]
        fill, outline = _marker_colors(centers[ci])

        cx_all = xs[sel]
        cy_all = ys[sel]

        x_min, x_max = int(cx_all.min()), int(cx_all.max())

        cross_x = []
        for qx, qy, a, b in crossings:
            if ci in (a, b):
                cross_x.append(qx - offset)
                cross_x.append(qx + offset)
        cross_x.sort()

        anchors = list(range(x_min, x_max + 1, step))

        cap = len(anchors) * 2
        if len(cross_x) > cap:
            idx = np.linspace(0, len(cross_x) - 1, cap).astype(int)
            cross_x = [cross_x[i] for i in sorted(set(idx))]

        plan = [(0, x) for x in cross_x]
        plan.extend((1, x) for x in anchors)
        plan.sort()

        budget = len(anchors) * 2
        used = 0

        for prio, x_target in plan:
            if used >= budget:
                break
            if not (x_min <= x_target <= x_max):
                continue
            near = np.abs(cx_all - x_target) <= 2
            if not near.any():
                continue

            cand = np.unique(cy_all[near])
            y_med = float(np.median(cy_all[near]))

            y_here = None
            for y in sorted(cand, key=lambda y: abs(y - y_med)):
                if _on_crossing(x_target, int(y)):
                    continue
                if _covers_foreign(ci, x_target, int(y), marker_size):
                    continue
                if _on_own_line(arr, x_target, int(y), centers, ci, own_tol):
                    if not _collides(occupied, x_target, int(y), marker_size):
                        y_here = int(y)
                        break
            if y_here is None:
                continue

            _mark_occupied(occupied, x_target, y_here, marker_size)
            _draw_marker(result, x_target, y_here, shape, marker_size,
                         fill, outline)
            used += 1

    return result


def _is_line_chart(valid: np.ndarray, thin: np.ndarray, min_area: int = 150) -> bool:
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8))
    thick_area = 0
    thin_area = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp = (labels == i)
        if (comp & thin).sum() > area * 0.5:
            thin_area += area
        else:
            thick_area += area

    total = thick_area + thin_area
    if total == 0:
        return False
    return (thin_area / total) > 0.05


def _find_legend_squares(valid: np.ndarray) -> np.ndarray:
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8))
    squares = np.zeros_like(valid, dtype=bool)

    for i in range(1, n):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if not (4 <= cw <= 14 and 4 <= ch <= 14):
            continue
        if area < 12:
            continue
        aspect = cw / ch if ch > 0 else 0
        if not (0.8 <= aspect <= 1.25):
            continue
        fill = area / (cw * ch) if (cw * ch) > 0 else 0
        if fill < 0.75:
            continue
        squares |= (labels == i)

    return squares


def _find_thin_structures(valid: np.ndarray, erode_px: int = 2) -> np.ndarray:
    import cv2

    k_size = erode_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8))
    thin = np.zeros_like(valid, dtype=bool)

    SURVIVE_THRESHOLD = 0.55

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:
            continue
        comp = (labels == i).astype(np.uint8)
        survived = int(cv2.erode(comp, kernel).sum())
        if survived / area < SURVIVE_THRESHOLD:
            thin |= (labels == i)

    return thin


def _find_mixed_chart_lines(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    import cv2

    height, width = valid.shape
    min_line_area = max(150, int(height * width * 0.0006))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hue_bins = hsv[:, :, 0] // 6
    lines = np.zeros_like(valid)

    for hue_bin in np.unique(hue_bins[valid]):
        color_mask = valid & (hue_bins == hue_bin)
        if color_mask.sum() < min_line_area:
            continue
        thin_color = _find_thin_structures(color_mask)
        if thin_color.sum() / color_mask.sum() < 0.35:
            continue

        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            thin_color.astype(np.uint8), connectivity=4
        )
        for label_id in range(1, labels_count):
            component_area = int(stats[label_id, cv2.CC_STAT_AREA])
            component_width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            if component_area >= min_line_area and max(component_width, component_height) >= 40:
                lines |= labels == label_id

    return lines


def _adaptive_spacing(h_img: int, w_img: int) -> float:
    ref = max(h_img, w_img)
    return max(8.0, min(ref / 100.0, 40.0))
def _chart_pattern_spacing(series: list[_PixelSeries]) -> float:
    import cv2

    diameters = []
    for candidate in series:
        component_diameters = [
            2.0 * cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5).max()
            for mask in candidate.masks
            if mask.sum() >= 100
        ]
        if component_diameters:
            diameters.append(max(component_diameters))

    if not diameters:
        return 12.0

    typical_diameter = float(np.median(diameters))
    return float(np.clip(round(4.0 + 1.5 * np.sqrt(typical_diameter)), 10, 28))


def _protect_text_pixels(arr: np.ndarray) -> np.ndarray:
    import cv2

    height, width = arr.shape[:2]
    channel_min = arr.min(axis=2)
    channel_max = arr.max(axis=2)
    dark_neutral = (channel_max <= 110) & ((channel_max - channel_min) <= 25)
    # Do not protect every light pixel: pale grey data series (for example a
    # 2019 bar) would otherwise be removed before pattern detection.  Bright
    # text is handled below as compact connected components instead.
    protected = cv2.dilate(
        dark_neutral.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)

    bright_words = cv2.dilate(
        (channel_min >= 180).astype(np.uint8), np.ones((3, 7), dtype=np.uint8)
    )
    labels_count, _, stats, _ = cv2.connectedComponentsWithStats(bright_words, connectivity=4)
    for label_id in range(1, labels_count):
        x, y, box_width, box_height, area = stats[label_id]
        if not (
            4 <= box_width <= width * 0.75
            and 4 <= box_height <= height * 0.35
            and area <= height * width * 0.05
        ):
            continue
        fill_ratio = area / (box_width * box_height)
        # A solid light rectangle is normally chart data, not a word.
        if fill_ratio >= 0.75 and box_width >= 8 and box_height >= 8:
            continue
        x1 = max(0, int(x) - 1)
        y1 = max(0, int(y) - 1)
        x2 = min(width, int(x + box_width) + 1)
        y2 = min(height, int(y + box_height) + 1)
        protected[y1:y2, x1:x2] = True

    return protected


def _edge_neutral_background(arr: np.ndarray) -> np.ndarray:
    """Find light neutral canvas areas connected to the image boundary.

    Light-grey report backgrounds are not data.  Treating them as a series
    produces an enormous grey hatch behind charts, while genuine grey bars or
    pie slices normally sit away from the outer canvas edge.
    """
    import cv2

    height, width = arr.shape[:2]
    channel_min = arr.min(axis=2)
    channel_max = arr.max(axis=2)
    neutral_light = (channel_max - channel_min <= 22) & (channel_min >= 170)
    border_width = max(2, min(8, min(height, width) // 40))
    border = np.zeros((height, width), dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    edge_pixels = arr[border & neutral_light]
    if not len(edge_pixels):
        return np.zeros((height, width), dtype=bool)

    # Do not connect every light neutral pixel merely because it is light.
    # A pale grey bar and a white canvas then form one boolean component.
    # Instead use the actual edge colour of this image as the background seed.
    background_rgb = np.median(edge_pixels, axis=0).astype(np.float32)
    color_distance = np.linalg.norm(arr.astype(np.float32) - background_rgb, axis=2)
    near_edge_background = neutral_light & (color_distance <= 18.0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        near_edge_background.astype(np.uint8), connectivity=4
    )
    background = np.zeros((height, width), dtype=bool)
    min_area = max(64, int(height * width * 0.005))
    for label_id in range(1, count):
        x, y, box_width, box_height, area = stats[label_id]
        if area < min_area:
            continue
        touches_edge = (
            x == 0
            or y == 0
            or x + box_width >= width
            or y + box_height >= height
        )
        if touches_edge:
            background |= labels == label_id
    return background


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

    stroke_width = np.where(bold, 3.0, 2.0)

    if family == "lines":
        return (proj % spacing) < stroke_width

    if family == "dashed":
        line = (proj % spacing) < stroke_width
        dash_period = np.where(bold, spacing * 1.2, spacing * 1.8)
        dash = (perp % dash_period) < (dash_period * 0.5)
        return line & dash

    if family == "grid":
        line1 = (proj % spacing) < (stroke_width * 0.7)
        line2 = (perp % spacing) < (stroke_width * 0.7)
        return line1 | line2

    if family == "dots":
        gx = (proj % spacing)
        gy = (perp % spacing)
        cx = cy = spacing / 2
        r = np.where(bold, 2.8, 2.0)
        return (gx - cx) ** 2 + (gy - cy) ** 2 <= r ** 2

    if family == "waves":
        wave = np.sin(proj / spacing * 2 * np.pi) * (spacing * 0.3)
        return np.abs((perp % spacing) - spacing / 2 - wave) < stroke_width

    if family == "zigzag":
        tri = np.abs((proj % spacing) - spacing / 2)
        return np.abs((perp % spacing) - tri) < stroke_width

    if family == "crosshatch":
        d1 = ((proj + perp) % spacing) < (stroke_width * 0.7)
        d2 = ((proj - perp) % spacing) < (stroke_width * 0.7)
        return d1 | d2

    if family == "bricks":
        row = np.floor(perp / spacing)
        offset = (row % 2) * (spacing / 2)
        hline = (perp % spacing) < (stroke_width * 0.6)
        vline = ((proj + offset) % spacing) < (stroke_width * 0.6)
        return hline | vline

    return (proj % spacing) < stroke_width


def _pie_region_mask(colorful: np.ndarray) -> np.ndarray:
    """Return the filled chart regions for circular/donut charts."""
    import cv2

    height, width = colorful.shape
    kernel_size = max(5, min(15, min(height, width) // 80 * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    connected = cv2.morphologyEx(colorful.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=4)
    pie_regions = np.zeros((height, width), dtype=np.uint8)

    for label_id in range(1, labels_count):
        box_width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < height * width * 0.06:
            continue
        aspect = box_width / box_height if box_height else 0.0
        fill_ratio = area / (box_width * box_height)
        if 0.75 <= aspect <= 1.35 and 0.42 <= fill_ratio <= 0.90:
            component = (labels == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            points = np.vstack([contour.reshape(-1, 2) for contour in contours])
            (center_x, center_y), radius = cv2.minEnclosingCircle(points)
            cv2.circle(
                pie_regions,
                (round(center_x), round(center_y)),
                max(1, round(radius)),
                1,
                thickness=-1,
            )

    return pie_regions.astype(bool)


def _looks_like_pie(colorful: np.ndarray) -> bool:
    return bool(_pie_region_mask(colorful).any())

def apply_double_coding(
    image: Image.Image,
    opacity: int = 100,
    smooth_shadows: bool = True,
    figure_source: str = "unknown",
    exclude_text: bool = True,
) -> Image.Image:
    import cv2

    arr = np.array(image.convert("RGB"))
    arr_for_classification = _smooth_shadows(arr) if smooth_shadows else arr

    hsv = cv2.cvtColor(arr_for_classification, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255

    delta = arr_for_classification.astype(np.float32).max(axis=2) - \
            arr_for_classification.astype(np.float32).min(axis=2)

    channel_max = arr_for_classification.max(axis=2)
    colorful = (delta >= 25) & (sat >= 0.12)
    neutral = (delta < 25) & (channel_max >= 115) & (channel_max <= 235)
    line_valid = (
        (delta >= 25)
        & (channel_max >= 30)
        & (arr_for_classification.min(axis=2) <= 248)
        & (sat >= 0.12)
    )
    thin = np.zeros_like(line_valid)
    legend = np.zeros_like(line_valid)
    mixed_chart_lines = np.zeros_like(line_valid)
    line_chart = False
    lines_for_markers: np.ndarray | None = None
    if exclude_text:
        thin = _find_thin_structures(line_valid)
        legend = _find_legend_squares(line_valid)
        line_chart = _is_line_chart(line_valid, thin)
        if line_chart:
            labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                thin.astype(np.uint8), connectivity=4
            )
            lines_for_markers = np.zeros_like(thin)
            for label_id in range(1, labels_count):
                if stats[label_id, cv2.CC_STAT_AREA] >= 150:
                    lines_for_markers |= labels == label_id
        else:
            mixed_chart_lines = _find_mixed_chart_lines(arr, line_valid)

    valid = (colorful | neutral) & (channel_max >= 30)
    protected = _protect_text_pixels(arr)
    edge_background = _edge_neutral_background(arr_for_classification)
    valid &= ~protected
    valid &= ~edge_background
    if exclude_text:
        if line_chart:
            valid &= ~(_find_thin_structures(valid) & ~legend)
        else:
            valid &= ~(thin & ~legend)

    # Detect the pie outline from coloured regions only.  A grey background
    # can otherwise merge with a grey slice and hide the circular geometry.
    pie_regions = _pie_region_mask(colorful & ~protected)
    is_pie = figure_source == "piechart" or pie_regions.any()
    if is_pie and pie_regions.any():
        valid &= pie_regions
    color_distance_threshold = 12.0 if is_pie else 30.0
    min_data_fraction = 0.003 if is_pie else 0.0008
    series = _build_color_series(
        arr,
        valid,
        color_distance_threshold,
        min_data_fraction,
        is_pie=is_pie,
    )
    spacing = _chart_pattern_spacing(series)
    result = _draw_series_patterns(arr, series, opacity, protected, spacing)
    result = _draw_legend_patterns(
        arr,
        result,
        series,
        opacity,
        is_pie,
    )
    marker_lines = lines_for_markers if line_chart else mixed_chart_lines
    if marker_lines.any():
        result = _place_markers_on_lines(
            result.astype(np.uint8),
            marker_lines,
            arr,
            legend_handles=find_legend_handles(arr),
        ).astype(np.float32)

    return Image.fromarray(result.astype(np.uint8))
