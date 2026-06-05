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
    area = seg.area
    if area < 100:      return 3   # квадратики легенды 7x7
    if area < 500:      return 5   # мелкие объекты
    if area < 2000:     return 8   # средние
    return default_tile_size       # крупные столбцы стандартный


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

        # Маленькие объекты легенды (7x7px) увеличенный квадрат с паттерном
        if seg.area < 200:
            _draw_legend_marker(result, seg.mask, seg.pattern_color, seg.pattern, pattern_opacity)
            continue

        # Динамический tile_size для столбцов
        adaptive_tile = _get_tile_size(seg, tile_size)

        pattern_layer = _make_tiled_pattern(
            w, h,
            seg.pattern,
            seg.pattern_color,
            pattern_opacity,
            adaptive_tile,
        )

        mask_arr = seg.mask.astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_arr, mode="L")
        pattern_layer.putalpha(mask_pil)

        result = Image.alpha_composite(result, pattern_layer)

        if stroke_outline:
            _draw_segment_outline(result, seg.mask, seg.pattern_color, outline_opacity, 2)

    return result


def _draw_legend_marker(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple,
    pattern: str,
    pattern_opacity: int = 200,
    marker_size: int = 16,
) -> None:
    """
    Для маленьких квадратиков легенды рисуем увеличенный квадрат
    с тем же паттерном что у столбцов поверх оригинала
    marker_size=16 гарантирует что паттерн будет читаем
    """
    import cv2

    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Центр оригинального квадратика
        cx = x + w // 2
        cy = y + h // 2

        # Координаты увеличенного квадрата
        half = marker_size // 2
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(image.width - 1, cx + half), min(image.height - 1, cy + half)

        # 1. Заливка цветом
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([x1, y1, x2, y2], fill=(*color, 255))

        # 2. Паттерн поверх заливки
        if pattern in TILE_FACTORIES:
            tile_size = max(4, marker_size // 3)
            r, g, b = color[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            pat_color = (0, 0, 0) if luminance > 100 else (255, 255, 255)
            legend_pattern = "checkerboard" if pattern in ("horizontal_lines", "vertical_lines") else pattern

            # Рисуем паттерн только в области квадратика напрямую
            pat_tile = TILE_FACTORIES[legend_pattern](tile_size, pat_color, 255)
            for py in range(y1, y2, pat_tile.height):
                for px in range(x1, x2, pat_tile.width):
                    overlay.paste(pat_tile, (px, py), pat_tile)

        # 3. Чёрная рамка
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0, 255), width=1)

    combined = Image.alpha_composite(image, overlay)
    image.paste(combined)


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