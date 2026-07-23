from __future__ import annotations
import numpy as np
from PIL import Image
from collections import defaultdict


_N_SECTORS = 16
_ANGLE_LUT = [
    0, 90, 22.5, 112.5, 45, 135, 67.5, 157.5,
    11.25, 101.25, 33.75, 123.75, 56.25, 146.25, 78.75, 168.75,
]
_PATTERN_FAMILIES = [
    "dots", "herringbone", "triangles", "lines", "rings", "star", "plus",
    "checker",
]

_PATTERN_FILL = {
    "dots": 0.2193,
    "herringbone": 0.1875,
    "triangles": 0.3594,
    "lines": 0.2969,
    "rings": 0.1320,
    "star": 0.3381,
    "plus": 0.6000,
    "checker": 0.4900,
}

_SHAPE_FAMILIES = {"star", "triangles", "rings", "plus"}

_LEGEND_SHAPE = {
    "star": (0.50, 1.00, 1.01),
    "plus": (0.50, 1.00, 0.72),
    "triangles": (0.50, 1.00, 0.75),
    "rings": (0.85, 1.70, 1.50),
}


_PATTERN_DIST = [
    [0.00, 0.32, 0.29, 0.99, 0.51, 0.54, 0.43, 0.78],
    [0.32, 0.00, 0.39, 1.14, 0.42, 0.52, 0.49, 0.69],
    [0.29, 0.39, 0.00, 0.97, 0.33, 0.32, 0.22, 0.58],
    [0.99, 1.14, 0.97, 0.00, 1.04, 0.97, 0.98, 1.15],
    [0.51, 0.42, 0.33, 1.04, 0.00, 0.26, 0.38, 0.39],
    [0.54, 0.52, 0.32, 0.97, 0.26, 0.00, 0.33, 0.42],
    [0.43, 0.49, 0.22, 0.98, 0.38, 0.33, 0.00, 0.60],
    [0.78, 0.69, 0.58, 1.15, 0.39, 0.42, 0.60, 0.00],
]


def _pick_families(n: int) -> list:
    total = len(_PATTERN_FAMILIES)
    if n >= total:
        return list(range(total))
    if n <= 1:
        return [0]
    import itertools
    best = None
    for combo in itertools.combinations(range(total), n):
        worst = min(_PATTERN_DIST[a][b] for a, b in itertools.combinations(combo, 2))
        if best is None or worst > best[0]:
            best = (worst, combo)
    return list(best[1])


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
        if count > total_valid * 0.020:
            present.add(s)
    return present


def _assign_pattern_families(present_sectors: set[int]) -> dict[int, str]:
    sectors_sorted = sorted(present_sectors)
    n = len(sectors_sorted)
    if n == 0:
        return {}
    picked = _pick_families(n)
    return {s: _PATTERN_FAMILIES[picked[i % len(picked)]]
            for i, s in enumerate(sectors_sorted)}


_MARKER_SHAPES = ["circle", "square", "triangle", "diamond", "cross", "star"]

_CVD_MATRICES = {
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "protanopia":   np.array([[0.152286, 1.052583, -0.204868],
                              [0.114503, 0.786281, 0.099216],
                              [-0.003882, -0.048116, 1.051998]]),
    "tritanopia":   np.array([[1.255528, -0.076749, -0.178779],
                              [-0.078411, 0.930809, 0.147602],
                              [0.004733, 0.691367, 0.303900]]),
}
_CONFUSABLE_DE = 15.0


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
    return [c for _, c in keep]


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


def _find_peaks(cx_all, cy_all, smooth: int = 9, prominence: float = 3.0,
                min_gap: int = 12) -> list:
    if len(cx_all) < 40:
        return []
    order = np.argsort(cx_all)
    xs_s = cx_all[order]
    ys_s = cy_all[order]
    ux = np.unique(xs_s)
    if len(ux) < 20:
        return []
    med = np.zeros(len(ux), dtype=np.float64)
    start = 0
    for i, x in enumerate(ux):
        end = start
        while end < len(xs_s) and xs_s[end] == x:
            end += 1
        med[i] = np.median(ys_s[start:end])
        start = end
    kern = np.ones(smooth) / smooth
    sm = np.convolve(med, kern, mode="same")
    d = np.diff(sm)
    raw = []
    for i in range(2, len(d) - 2):
        if d[i - 1] * d[i] < 0:
            lo = max(0, i - 15)
            hi = min(len(sm), i + 15)
            if abs(sm[i] - sm[lo:hi].mean()) >= prominence:
                raw.append(int(ux[i]))
    out = []
    for x in raw:
        if not out or x - out[-1] > min_gap:
            out.append(x)
    return out


def _place_markers_on_lines(
    result: np.ndarray,
    lines_mask: np.ndarray,
    arr: np.ndarray,
    marker_size: int = 9,
    step: int = 60,
    legend_handles: list | None = None,
) -> np.ndarray:
    import cv2

    handles = legend_handles if legend_handles else []
    if handles:
        line_colors = [np.array(h["rgb"], dtype=np.float32) for h in handles]
    else:
        line_colors = _cluster_line_colors(arr, lines_mask)
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

        peaks = _find_peaks(cx_all, cy_all)

        plan = [(0, x) for x in peaks]
        plan.extend((1, x) for x in cross_x)
        plan.extend((2, x) for x in anchors)
        plan.sort()

        budget = len(anchors) * 2 + len(peaks)
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


def _adaptive_spacing(h_img: int, w_img: int) -> float:
    ref = max(h_img, w_img)
    return max(13.0, min(ref / 60.0, 60.0))


def _draw_pattern(
    family: str,
    xx: np.ndarray, yy: np.ndarray,
    angle_rad: float,
    bold: np.ndarray,
    spacing: float = 12.0,
) -> np.ndarray:
    k = _PATTERN_FILL.get(family, 0.25)
    base = spacing * 0.72 if bool(np.asarray(bold).ravel()[0]) else spacing
    sp = float(max(4, int(round(base))))

    def _px(value):
        return float(max(1, int(round(value))))

    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    proj = xx * cos_a + yy * sin_a
    perp = -xx * sin_a + yy * cos_a

    if family == "dots":
        gx = proj % sp
        gy = perp % sp
        c = sp / 2
        r = _px(sp * k)
        return (gx - c) ** 2 + (gy - c) ** 2 <= r ** 2

    if family == "herringbone":
        row = np.floor(perp / (sp * 0.9))
        flip = (row % 2) == 1
        d = np.where(flip, proj + perp, proj - perp)
        return (d % sp) < _px(sp * k)

    if family == "triangles":
        u = (proj % sp) / sp
        v = (perp % sp) / sp
        cu = u - 0.5
        v0 = 0.5 - k
        v1 = 0.5 + k
        t = np.clip((v - v0) / (v1 - v0), 0.0, 1.0)
        return (v >= v0) & (v <= v1) & (np.abs(cu) <= k * t)

    if family == "lines":
        return (proj % sp) < _px(sp * k)

    if family == "rings":
        gx = (proj % (sp * 1.7)) - sp * 0.85
        gy = (perp % (sp * 1.7)) - sp * 0.85
        d = np.sqrt(gx ** 2 + gy ** 2)
        return np.abs(d - _px(sp * 0.62)) < _px(sp * k)

    if family == "plus":
        row = np.floor(perp / sp)
        off = (row % 2) * sp * 0.5
        u = ((proj + off) % sp) - sp * 0.5
        v = (perp % sp) - sp * 0.5
        arm = _px(sp * k * 0.60)
        th = _px(sp * k * 0.20)
        return ((np.abs(u) < th) & (np.abs(v) < arm)) | \
               ((np.abs(v) < th) & (np.abs(u) < arm))

    if family == "star":
        u = ((proj % sp) / sp) - 0.5
        v = ((perp % sp) / sp) - 0.5
        r = np.sqrt(u ** 2 + v ** 2) + 1e-6
        ang = np.arctan2(v, u)
        lobe = 0.5 + 0.5 * np.cos(ang * 5.0)
        return r < (k * (0.5 + 1.0 * lobe))

    if family == "checker":
        a = np.floor(proj / sp).astype(np.int64)
        b = np.floor(perp / sp).astype(np.int64)
        return ((a + b) % 2) == 0

    return (proj % sp) < (sp * 0.25)


def _stabilize_sectors(sector: np.ndarray, valid: np.ndarray,
                       win: int = 7) -> np.ndarray:
    import cv2
    best = np.zeros(sector.shape, dtype=np.float32)
    out = sector.copy()
    for k in range(_N_SECTORS):
        m = ((sector == k) & valid).astype(np.float32)
        c = cv2.boxFilter(m, -1, (win, win), normalize=True)
        upd = c > best
        best = np.where(upd, c, best)
        out = np.where(upd, k, out)
    return np.where(valid, out, sector).astype(np.int32)


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
    sector = _stabilize_sectors(sector, valid)
    present = _get_present_sectors(hue_smooth, valid)
    family_map = _assign_pattern_families(present)

    yy, xx = np.mgrid[0:h_img, 0:w_img].astype(np.float32)
    result = np.zeros((h_img, w_img), dtype=bool)

    for s in present:
        sector_mask = (sector == s)
        if not sector_mask.any():
            continue
        family = family_map.get(s, "lines")
        angle_rad = 0.0 if family in _SHAPE_FAMILIES \
            else np.deg2rad(_ANGLE_LUT[s])
        sector_bold = np.full(sector_mask.shape,
                              bool(bold[sector_mask].mean() > 0.5))
        sp_use = _adaptive_spacing(h_img, w_img)
        ys_s, xs_s = np.where(sector_mask)
        span = min(xs_s.max() - xs_s.min(), ys_s.max() - ys_s.min()) + 1
        if span < sp_use * 3.0:
            sp_use = max(6.0, span / 3.0)
        sp_use = float(max(4, int(round(sp_use))))
        if family in _SHAPE_FAMILIES:
            ox = float(int(round(xs_s.mean() - sp_use * 0.5)))
            oy = float(int(round(ys_s.mean() - sp_use * 0.5)))
        else:
            ox = 0.0
            oy = 0.0
        pattern = _draw_pattern(family, xx - ox, yy - oy, angle_rad,
                                sector_bold, spacing=sp_use)
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
        if not (0.8 <= aspect <= 1.25):
            continue
        fill = area / (cw * ch) if (cw * ch) > 0 else 0
        if fill < 0.75:
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
        angle_rad = 0.0 if family in _SHAPE_FAMILIES \
            else np.deg2rad(_ANGLE_LUT[best_sector])

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

        if family in _SHAPE_FAMILIES:
            centre_f, period_f, extent_f = _LEGEND_SHAPE[family]
            target = MARKER / 1.42
            leg_sp = target / extent_f
            ccx = (x0 + x1 - 1) * 0.5
            ccy = (y0 + y1 - 1) * 0.5
            ca, sa = np.cos(angle_rad), np.sin(angle_rad)
            rx = (xx - ccx) * ca + (yy - ccy) * sa
            ry = -(xx - ccx) * sa + (yy - ccy) * ca
            half = leg_sp * period_f * 0.5
            inside = (np.abs(rx) < half) & (np.abs(ry) < half)
            pattern = _draw_pattern(family, rx + leg_sp * centre_f,
                                    ry + leg_sp * centre_f, 0.0, bold,
                                    spacing=leg_sp) & inside
        else:
            pattern = _draw_pattern(family, xx - x0, yy - y0, angle_rad, bold,
                                    spacing=MARKER / 2.0)
        block = np.zeros((h_img, w_img), dtype=bool)
        block[y0:y1, x0:x1] = True
        draw_here = block & pattern
        el = sector_rgb[best_sector].astype(np.float32)
        el_lum = 0.2126 * el[0] + 0.7152 * el[1] + 0.0722 * el[2]
        el_ink = 255.0 if el_lum < 128.0 else 0.0
        result[draw_here] = (result[draw_here] * (1 - alpha) + el_ink * alpha)

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
    sector = _stabilize_sectors(sector, valid)
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
            if st_c[c, cv2.CC_STAT_AREA] >= 300:
                big_pixels.append(lab_c == c)
        if big_pixels:
            big_mask_s = np.any(big_pixels, axis=0)
            sector_rgb[s] = arr_orig[big_mask_s].mean(axis=0)

    for s in present:
        sector_mask = (sector == s)
        if not sector_mask.any():
            continue
        family = family_map.get(s, "lines")
        angle_rad = 0.0 if family in _SHAPE_FAMILIES \
            else np.deg2rad(_ANGLE_LUT[s])
        sector_bold = np.full(sector_mask.shape,
                              bool(bold[sector_mask].mean() > 0.5))
        sp_use = _adaptive_spacing(h_img, w_img)
        ys_s, xs_s = np.where(sector_mask)
        span = min(xs_s.max() - xs_s.min(), ys_s.max() - ys_s.min()) + 1
        if span < sp_use * 3.0:
            sp_use = max(6.0, span / 3.0)
        sp_use = float(max(4, int(round(sp_use))))
        if family in _SHAPE_FAMILIES:
            ox = float(int(round(xs_s.mean() - sp_use * 0.5)))
            oy = float(int(round(ys_s.mean() - sp_use * 0.5)))
        else:
            ox = 0.0
            oy = 0.0
        pattern = _draw_pattern(family, xx - ox, yy - oy, angle_rad,
                                sector_bold, spacing=sp_use)
        result |= (sector_mask & pattern)

    return result, present, family_map, sector_rgb


def _detect_background(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    import cv2
    h, w = arr.shape[:2]
    out = np.zeros((h, w), dtype=bool)

    limit = 900
    if max(h, w) > limit:
        sc = limit / float(max(h, w))
        small = cv2.resize(arr, (max(1, int(w * sc)), max(1, int(h * sc))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = arr

    sh, sw = small.shape[:2]
    stotal = sh * sw
    q = (small.astype(np.int32) // 24 * 24)
    packed = (q[:, :, 0] << 16) | (q[:, :, 1] << 8) | q[:, :, 2]
    vals, cnt = np.unique(packed.ravel(), return_counts=True)

    for v, area in zip(vals, cnt):
        if area < stotal * 0.15:
            continue
        c = np.array([(int(v) >> 16) & 255, (int(v) >> 8) & 255, int(v) & 255],
                     dtype=np.int32)
        mx, mn = int(c.max()), int(c.min())
        if mn > 230 or mx < 60 or (mx - mn) < 25:
            continue
        ms = (np.abs(small.astype(np.int32) - c).max(axis=2) < 30)
        ys, xs = np.where(ms)
        if len(xs) < 20:
            continue
        spread_x = (xs.max() - xs.min()) / sw
        spread_y = (ys.max() - ys.min()) / sh
        if max(spread_x, spread_y) > 0.75:
            out |= (np.abs(arr.astype(np.int32) - c).max(axis=2) < 30)
    return out


def _detect_text(arr: np.ndarray) -> np.ndarray:
    import cv2
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    dark = (gray < 90).astype(np.uint8)
    n, lab, st, cent = cv2.connectedComponentsWithStats(dark, 8)
    cands = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        wd = st[i, cv2.CC_STAT_WIDTH]
        ht = st[i, cv2.CC_STAT_HEIGHT]
        fill = a / max(1, wd * ht)
        if 4 <= a <= 400 and 3 <= wd <= 25 and 5 <= ht <= 28 and fill > 0.15:
            cands.append((int(cent[i][0]), int(cent[i][1]), ht, i))
    keep = set()
    for a in range(len(cands)):
        xa, ya, ha, ia = cands[a]
        nb = 0
        for b in range(len(cands)):
            if b == a:
                continue
            xb, yb, hb, ib = cands[b]
            if abs(ya - yb) <= ha and abs(xa - xb) <= ha * 4 and abs(ha - hb) <= 4:
                nb += 1
        if nb >= 2:
            keep.add(ia)
    m = np.zeros(arr.shape[:2], dtype=bool)
    for i in keep:
        m |= (lab == i)
    if m.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.dilate(m.astype(np.uint8), k).astype(bool)
    return m


def apply_double_coding(
    image: Image.Image,
    opacity: int = 100,
    smooth_shadows: bool = True,
    exclude_text: bool = True,
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

    bg_mask = _detect_background(arr_for_classification, valid)
    valid = valid & ~bg_mask

    line_chart = False
    lines_for_markers = None
    if exclude_text:
        thin = _find_thin_structures(valid)
        legend = _find_legend_squares(valid)
        line_chart = _is_line_chart(valid, thin)

        if line_chart:
            import cv2
            n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8))
            big_thin = np.zeros_like(thin)
            for i in range(1, n_c):
                if st_c[i, cv2.CC_STAT_AREA] >= 150:
                    big_thin |= (lab_c == i)
            lines_for_markers = big_thin
            valid = valid & ~(thin & ~legend)
        else:
            valid = valid & ~(thin & ~legend)

    hue_scaled_light = (hue / 360.0 * 255.0).astype(np.uint8)
    hue_legend = cv2.medianBlur(hue_scaled_light, 3).astype(np.float32) / 255.0 * 360.0

    big_mask, present, family_map, sector_rgb = _procedural_pattern_mask_with_families(
        h_img, w_img, hue, sat, valid, arr
    )
    import cv2
    _lh, _lw = arr.shape[:2]
    if max(_lh, _lw) > 1400:
        _sc = 1400.0 / float(max(_lh, _lw))
        _sm = cv2.resize(arr, (max(1, int(_lw * _sc)), max(1, int(_lh * _sc))),
                         interpolation=cv2.INTER_AREA)
        local = cv2.resize(cv2.medianBlur(_sm, 21), (_lw, _lh),
                           interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        local = cv2.medianBlur(arr, 21).astype(np.float32)

    draw_mask = big_mask & valid
    if exclude_text:
        deviation = np.abs(arr.astype(np.float32) - local).sum(axis=2)
        foreign = deviation > 100
        halo = max(2, int(round(_adaptive_spacing(h_img, w_img) * 0.22)))
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (halo * 2 + 1, halo * 2 + 1))
        n_f, lab_f, st_f, _ = cv2.connectedComponentsWithStats(
            foreign.astype(np.uint8), 8)
        compact = np.zeros_like(foreign)
        limit = _adaptive_spacing(h_img, w_img) * 2.5
        for i in range(1, n_f):
            if st_f[i, cv2.CC_STAT_WIDTH] <= limit and \
                    st_f[i, cv2.CC_STAT_HEIGHT] <= limit:
                compact |= (lab_f == i)
        foreign = foreign | cv2.dilate(compact.astype(np.uint8),
                                       kern).astype(bool)
        draw_mask = draw_mask & ~foreign

    result = arr.astype(np.float32)
    alpha = opacity / 100.0
    lum = (0.2126 * local[:, :, 0] + 0.7152 * local[:, :, 1]
           + 0.0722 * local[:, :, 2])
    ink = np.where(lum[..., None] < 128.0, 255.0, 0.0)
    result[draw_mask] = (result[draw_mask] * (1 - alpha)
                         + ink[draw_mask] * alpha)

    result = _redraw_small_elements(
        arr, valid, result, alpha, family_map, sector_rgb
    )

    if line_chart and lines_for_markers is not None:
        result_u8 = result.astype(np.uint8)
        result_u8 = _place_markers_on_lines(
            result_u8, lines_for_markers, arr,
            legend_handles=find_legend_handles(arr),
        )
        result = result_u8.astype(np.float32)

    return Image.fromarray(result.astype(np.uint8))