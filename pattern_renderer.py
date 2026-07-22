from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from segmenter import ColoredSegment


# генерит узоры

def _tile_diagonal_stripes(tile: int, color: tuple, opacity: int) -> Image.Image:
    size = tile * 2
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    c = (*color, opacity)
    for i in range(-size, size * 2, tile):
        draw.line([(i, size), (i + size, 0)], fill=c, width=max(1, tile // 7))
        draw.line([(i + size, size), (i + size * 2, 0)], fill=c, width=max(1, tile // 7))
    return img.crop((0, 0, tile, tile))


def _tile_dots(tile: int, color: tuple, opacity: int) -> Image.Image:
    img = Image.new("RGBA", (tile, tile), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = max(1, tile // 5)
    cx, cy = tile // 2, tile // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, opacity))
    return img


def _tile_crosshatch(tile: int, color: tuple, opacity: int) -> Image.Image:
    img = Image.new("RGBA", (tile, tile), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    c = (*color, opacity)
    mid = tile // 2
    draw.line([(0, mid), (tile, mid)], fill=c, width=1)
    draw.line([(mid, 0), (mid, tile)], fill=c, width=1)
    draw.line([(0, 0), (tile, tile)], fill=c, width=1)
    draw.line([(tile, 0), (0, tile)], fill=c, width=1)
    return img


def _tile_checkerboard(tile: int, color: tuple, opacity: int) -> Image.Image:
    half = max(1, tile // 2)
    img = Image.new("RGBA", (tile, tile), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    c = (*color, opacity)
    draw.rectangle([0, 0, half - 1, half - 1], fill=c)
    draw.rectangle([half, half, tile - 1, tile - 1], fill=c)
    return img


def _tile_horizontal_lines(tile: int, color: tuple, opacity: int) -> Image.Image:
    img = Image.new("RGBA", (tile, tile), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    mid = tile // 2
    draw.line([(0, mid), (tile, mid)], fill=(*color, opacity), width=max(1, tile // 7))
    return img


def _tile_vertical_lines(tile: int, color: tuple, opacity: int) -> Image.Image:
    img = Image.new("RGBA", (tile, tile), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    mid = tile // 2
    draw.line([(mid, 0), (mid, tile)], fill=(*color, opacity), width=max(1, tile // 7))
    return img


TILE_FACTORIES = {
    "diagonal_stripes": _tile_diagonal_stripes,
    "dots":             _tile_dots,
    "crosshatch":       _tile_crosshatch,
    "checkerboard":     _tile_checkerboard,
    "horizontal_lines": _tile_horizontal_lines,
    "vertical_lines":   _tile_vertical_lines,
}


def _make_tiled_pattern(
    width: int,
    height: int,
    pattern: str,
    color: tuple,
    opacity: int,
    tile_size: int,
) -> Image.Image:
    tile  = TILE_FACTORIES[pattern](tile_size, color, opacity)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for y in range(0, height, tile.height):
        for x in range(0, width, tile.width):
            layer.paste(tile, (x, y))
    return layer


def _get_tile_size(seg: ColoredSegment, default_tile_size: int) -> int:
    """
    Динамически подбирает tile_size под размер сегмента
    Маленькие объекты (легенда) получают мелкий тайл
    Большие столбцы стандартный
    """
    import cv2

    if seg.area < 500:
        area_tile = 5
    elif seg.area < 2000:
        area_tile = 8
    else:
        area_tile = default_tile_size

    distance = cv2.distanceTransform(
        seg.mask.astype(np.uint8), cv2.DIST_L2, 3
    )
    max_thickness = max(2, int(np.ceil(distance.max() * 2)))
    return max(2, min(area_tile, max_thickness))


def _apply_mask_to_alpha(layer: Image.Image, mask: np.ndarray) -> None:
    """Ограничивает существующую прозрачность паттерна маской сегмента."""
    pattern_alpha = np.asarray(layer.getchannel("A"), dtype=np.uint16)
    mask_alpha = mask.astype(np.uint16) * 255
    alpha = (pattern_alpha * mask_alpha // 255).astype(np.uint8)
    layer.putalpha(Image.fromarray(alpha, mode="L"))


def _relative_luminance(color: tuple) -> float:
    channels = []
    for channel in color[:3]:
        value = channel / 255
        channels.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _high_contrast_color(background: tuple) -> tuple[int, int, int]:
    luminance = _relative_luminance(background)
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return (0, 0, 0) if black_contrast >= white_contrast else (255, 255, 255)


# рендер

def render_patterns_on_segments(
    image: Image.Image,
    segments: list[ColoredSegment],
    pattern_opacity: int = 160,
    tile_size: int = 14,
    stroke_outline: bool = True,
    outline_opacity: int = 180,
) -> Image.Image:
    """
    Apply double-coding patterns to specific segmented regions.
    tile_size адаптируется под размер каждого сегмента автоматически.
    """
    w, h = image.size
    result = image.convert("RGBA").copy()

    for seg in segments:
        if seg.pattern not in TILE_FACTORIES:
            continue

        if seg.is_legend:
            _draw_legend_marker(
                result, seg.mask, seg.mean_rgb, seg.pattern, pattern_opacity
            )
            continue

        # Динамический tile_size для столбцов
        adaptive_tile = _get_tile_size(seg, tile_size)

        pattern_color = _high_contrast_color(seg.mean_rgb)
        effective_opacity = min(255, max(210, pattern_opacity))
        pattern_layer = _make_tiled_pattern(
            w, h,
            seg.pattern,
            pattern_color,
            effective_opacity,
            adaptive_tile,
        )

        _apply_mask_to_alpha(pattern_layer, seg.mask)

        result = Image.alpha_composite(result, pattern_layer)

        if stroke_outline:
            outline_width = 1 if adaptive_tile < 6 else 2
            _draw_segment_outline(
                result, seg.mask, pattern_color,
                min(255, max(210, outline_opacity)), outline_width
            )

    return result


def _draw_legend_marker(
    image: Image.Image,
    mask: np.ndarray,
    marker_color: tuple,
    pattern: str,
    pattern_opacity: int = 200,
) -> None:
    """Наносит паттерн строго внутри исходного маркера легенды."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return

    marker_width = int(xs.max() - xs.min() + 1)
    marker_height = int(ys.max() - ys.min() + 1)
    pattern_color = _high_contrast_color(marker_color)
    tile_size = max(2, min(4, min(marker_width, marker_height) // 2))
    pattern_layer = _make_tiled_pattern(
        image.width,
        image.height,
        pattern,
        pattern_color,
        min(255, max(210, pattern_opacity)),
        tile_size,
    )
    _apply_mask_to_alpha(pattern_layer, mask)
    image.paste(Image.alpha_composite(image, pattern_layer))
    _draw_segment_outline(image, mask, pattern_color, 255, width=1)


def _draw_segment_outline(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple,
    opacity: int,
    width: int = 2,
) -> None:
    import cv2

    mask_u8   = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for cnt in contours:
        pts = [tuple(p[0]) for p in cnt]
        if len(pts) > 2:
            pts.append(pts[0])
            draw.line(pts, fill=(*color, opacity), width=width)

    combined = Image.alpha_composite(image, overlay)
    image.paste(combined)
