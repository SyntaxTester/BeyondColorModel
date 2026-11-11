from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pixel_segmenter as ps

FOREIGN_DEV = 100
MEDIAN_K = 21
PEAK_SLACK = 2.0
TILE = 128
SPACING = 12.0

FIXTURES = [
    "gen_dark.png",
    "gen_native.png",
    "gen_nolegend8.png",
    "gen_similar.png",
    "gen_thick.png",
    "bars_550.png",
    "test_chart.png",
    "test_chart2.png",
    "test_chart3.png",
    "test_chart4.png",
    "test_chart5.png",
]


def foreign_mask(arr: np.ndarray) -> np.ndarray:
    local = cv2.medianBlur(arr, MEDIAN_K).astype(np.float32)
    dev = np.abs(arr.astype(np.float32) - local).sum(axis=2)
    return dev > FOREIGN_DEV


def pattern_descriptors() -> list[tuple[str, np.ndarray]]:
    ax = np.arange(TILE, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    bold = np.zeros_like(xx, dtype=bool)
    out = []
    for i, fam in enumerate(ps._DEPLOY_PATTERN_FAMILIES):
        angle = 0.0 if fam in ps._DEPLOY_SHAPE_FAMILIES else ps._DEPLOY_ANGLES[i]
        ink = np.asarray(ps._draw_pattern(fam, xx, yy, angle, bold, SPACING), dtype=np.float32)
        if ink.ndim > 2:
            ink = ink.reshape(TILE, TILE, -1)[:, :, 0]
        out.append((f"{i}:{fam}", descriptor(ink)))
    return out


def descriptor(ink: np.ndarray) -> np.ndarray:
    density = float(ink.mean())
    win = np.outer(np.hanning(TILE), np.hanning(TILE))
    mag = np.abs(np.fft.fftshift(np.fft.fft2((ink - ink.mean()) * win)))
    c = TILE // 2
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    r = np.hypot(yy - c, xx - c)
    th = np.mod(np.arctan2(yy - c, xx - c), np.pi)
    feats = []
    edges = np.linspace(2, c, 9)
    for a, b in zip(edges[:-1], edges[1:]):
        feats.append(mag[(r >= a) & (r < b)].sum())
    tedges = np.linspace(0, np.pi, 9)
    ring = (r >= 2) & (r < c)
    for a, b in zip(tedges[:-1], tedges[1:]):
        feats.append(mag[ring & (th >= a) & (th < b)].sum())
    f = np.asarray(feats, dtype=np.float64)
    n = np.linalg.norm(f)
    f = f / n if n > 0 else f
    return np.concatenate([f, [density * 4.0]])


def pattern_legibility() -> dict[str, dict]:
    ax = np.arange(TILE, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    bold = np.zeros_like(xx, dtype=bool)
    out = {}
    for i, fam in enumerate(ps._DEPLOY_PATTERN_FAMILIES):
        angle = 0.0 if fam in ps._DEPLOY_SHAPE_FAMILIES else ps._DEPLOY_ANGLES[i]
        ink = np.asarray(ps._draw_pattern(fam, xx, yy, angle, bold, SPACING), dtype=np.float32)
        if ink.ndim > 2:
            ink = ink.reshape(TILE, TILE, -1)[:, :, 0]
        binary = (ink > 0.5).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = sorted(int(st[k, cv2.CC_STAT_AREA]) for k in range(1, n))
        out[f"{i}:{fam}"] = {
            "density": round(float(binary.mean()), 4),
            "elem_px": int(np.median(areas)) if areas else 0,
            "elems": len(areas),
        }
    return out


def worst_pattern_pair() -> tuple[float, str]:
    d = pattern_descriptors()
    worst, pair = 1e9, ""
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            dist = float(np.linalg.norm(d[i][1] - d[j][1]))
            if dist < worst:
                worst, pair = dist, f"{d[i][0]}/{d[j][0]}"
    return worst, pair


def measure(path: Path) -> dict:
    img = Image.open(path).convert("RGB")
    src = np.asarray(img, dtype=np.uint8)
    fgn = foreign_mask(src)

    peaks: list[tuple[int, int]] = []
    marks: list[tuple[int, int]] = []

    real_peaks = ps._find_peaks
    real_marker = ps._draw_marker

    def spy_peaks(cx_all, cy_all, *a, **k):
        xs = real_peaks(cx_all, cy_all, *a, **k)
        cx = np.asarray(cx_all)
        cy = np.asarray(cy_all)
        for x in xs:
            sel = cy[cx == x]
            if len(sel):
                peaks.append((int(x), int(np.median(sel))))
        return xs

    def spy_marker(canvas, cx, cy, shape, size, color, outline=None):
        marks.append((int(cx), int(cy), int(size)))
        return real_marker(canvas, cx, cy, shape, size, color, outline)

    ps._find_peaks = spy_peaks
    ps._draw_marker = spy_marker
    try:
        t0 = time.perf_counter()
        out = np.asarray(ps.apply_double_coding(img, opacity=100), dtype=np.uint8)
        dt = time.perf_counter() - t0
    finally:
        ps._find_peaks = real_peaks
        ps._draw_marker = real_marker

    changed = np.any(out != src, axis=2)
    text_hit = int((changed & fgn).sum())

    covered = 0
    if peaks and marks:
        m = np.asarray(marks, dtype=np.float32)
        reach = m[:, 2] * 0.5 + PEAK_SLACK
        for px, py in peaks:
            if np.any(np.hypot(m[:, 0] - px, m[:, 1] - py) <= reach):
                covered += 1

    return {
        "px": int(src.shape[0] * src.shape[1]),
        "thin_hit": text_hit,
        "peaks": len(peaks),
        "peaks_covered": covered,
        "markers": len(marks),
        "sec": round(dt, 3),
    }


def main() -> int:
    base = Path(__file__).resolve().parent
    images = [base / n for n in FIXTURES]
    missing = [p.name for p in images if not p.exists()]
    images = [p for p in images if p.exists()]
    if missing:
        print("нет файлов:", ", ".join(missing))
    if not images:
        print("нет тестовых картинок рядом с bench.py")
        return 2

    baseline_path = base / "bench_baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    record = "--record" in sys.argv

    worst, pair = worst_pattern_pair()
    rows, fails = [], []

    for p in images:
        try:
            r = measure(p)
        except Exception as e:
            fails.append(f"{p.name}: УПАЛ {type(e).__name__}: {e}")
            continue
        cov = 100 * r["peaks_covered"] // r["peaks"] if r["peaks"] else -1
        rows.append((p.name, r, cov))

    print(f"{'график':22} {'Мп':>5} {'тонкие':>7} {'пики':>6} {'покрыто':>9} {'маркеров':>9} {'сек':>6}")
    for name, r, cov in rows:
        covs = f"{cov}%" if cov >= 0 else "-"
        print(f"{name:22} {r['px']/1e6:5.2f} {r['thin_hit']:7d} {r['peaks']:6d} {covs:>9} {r['markers']:9d} {r['sec']:6.2f}")

    print(f"\nхудшая пара паттернов: {worst:.3f}  ({pair})   разных узоров: {len(set(ps._DEPLOY_PATTERN_FAMILIES))} из {len(ps._DEPLOY_PATTERN_FAMILIES)}")

    now = {name: {"thin_hit": r["thin_hit"], "peaks_covered_pct": cov, "markers": r["markers"]}
           for name, r, cov in rows}
    now["_legibility"] = pattern_legibility()
    now["_patterns"] = {
        "worst_pair": round(worst, 4),
        "pair": pair,
        "distinct": len(set(ps._DEPLOY_PATTERN_FAMILIES)),
        "total": len(ps._DEPLOY_PATTERN_FAMILIES),
    }

    if record:
        baseline_path.write_text(json.dumps(now, indent=2, ensure_ascii=False))
        print(f"\nэталон записан: {baseline_path}")
        return 0

    if not baseline:
        print("\nэталона нет. запусти с --record")
        return 0

    for name, cur in now.items():
        old = baseline.get(name)
        if old is None:
            print(f"  НОВОЕ: {name}")
            continue
        if name == "_legibility":
            for fam, cur_v in cur.items():
                old_v = old.get(fam)
                if old_v is None:
                    continue
                if cur_v["density"] < old_v["density"] * 0.6:
                    fails.append(f"узор {fam}: плотность {old_v['density']} -> {cur_v['density']}")
                if cur_v["elem_px"] < max(4, old_v["elem_px"] * 0.6):
                    fails.append(f"узор {fam}: элемент {old_v['elem_px']} -> {cur_v['elem_px']} px")
            continue
        if name == "_patterns":
            if cur["worst_pair"] < old["worst_pair"] - 0.01:
                fails.append(f"паттерны: худшая пара {old['worst_pair']} -> {cur['worst_pair']} ({cur['pair']})")
            if cur["distinct"] < old["distinct"]:
                fails.append(f"паттерны: разных узоров {old['distinct']} -> {cur['distinct']}")
            if cur["distinct"] < cur["total"]:
                fails.append(f"паттерны: в наборе повтор ({cur['total']} слотов, {cur['distinct']} разных)")
            continue
        if cur["thin_hit"] > max(old["thin_hit"] + 20, old["thin_hit"] * 1.5):
            fails.append(f"{name}: текст перекрыт {old['text_hit']} -> {cur['thin_hit']} px")
        if old["peaks_covered_pct"] >= 0 and cur["peaks_covered_pct"] < old["peaks_covered_pct"] - 5:
            fails.append(f"{name}: покрытие пиков {old['peaks_covered_pct']}% -> {cur['peaks_covered_pct']}%")
        if cur["markers"] < old["markers"] * 0.8:
            fails.append(f"{name}: маркеров {old['markers']} -> {cur['markers']}")

    if fails:
        print("\nРЕГРЕСС:")
        for f in fails:
            print(f"  {f}")
        return 1
    print("\nвсё в пределах эталона")
    return 0


if __name__ == "__main__":
    sys.exit(main())
