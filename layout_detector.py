from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image



@dataclass
class FigureBlock:
    """A detected figure/chart region on a page."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    source: str        # 'publaynet' or 'opencv'

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop(self.bbox)


class PubLayNetDetector:
    """
    Layout detector using PubLayNet-trained Faster R-CNN via layoutparser.

    Install:
        pip install layoutparser
        pip install 'git+https://github.com/facebookresearch/detectron2.git'
    """

    MODEL_CONFIG = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    MODEL_EXTRA  = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/model_final"
    LABEL_MAP    = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

    def __init__(self, score_threshold: float = 0.7, device: str = "cpu"):
        try:
            import layoutparser as lp
        except ImportError:
            raise ImportError(
                "layoutparser not installed.\n"
                "Run: pip install layoutparser\n"
                "     pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            )

        print("[BeyondColor] Loading PubLayNet model...")
        self.model = lp.Detectron2LayoutModel(
            config_path=self.MODEL_CONFIG,
            model_path=self.MODEL_EXTRA,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_threshold],
            label_map=self.LABEL_MAP,
            device=device,
        )
        print("[BeyondColor] PubLayNet ready.")

    def detect(
        self,
        image: Image.Image,
        min_area: int = 10_000,
        dpi: int = 150,
        text_boxes: list | None = None,
    ) -> list[FigureBlock]:
        
        import layoutparser as lp

        arr = np.array(image.convert("RGB"))
        layout = self.model.detect(arr)

        blocks = []
        for block in layout:
            if block.type != "Figure":
                continue
            x1, y1, x2, y2 = (
                int(block.block.x_1), int(block.block.y_1),
                int(block.block.x_2), int(block.block.y_2),
            )
        
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(image.width, x2); y2 = min(image.height, y2)
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue
            blocks.append(FigureBlock(x1, y1, x2, y2, block.score, "publaynet"))

        return sorted(blocks, key=lambda b: -b.area)


class OpenCVDetector:

    def detect(
        self,
        image: Image.Image,
        min_area: int = 10_000,
        dpi: int = 150,
        min_side_pt: float = 60.0,
        min_saturated: float = 0.004,
        min_palette_share: float = 0.7,
        text_boxes: list | None = None,
    ) -> list[FigureBlock]:
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]
        rgb = arr.astype(np.int16)
        spread = rgb.max(2) - rgb.min(2)
        grey = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        ink = (grey < 235) | (spread >= 30)
        saturated = (spread >= 60) & (rgb.max(2) >= 60)
        words = []
        if text_boxes:
            text_mask = np.zeros((h, w), dtype=bool)
            for x1, y1, x2, y2 in text_boxes:
                x1, y1 = max(0, int(x1) - 1), max(0, int(y1) - 1)
                x2, y2 = min(w, int(np.ceil(x2)) + 1), min(h, int(np.ceil(y2)) + 1)
                if x2 > x1 and y2 > y1:
                    text_mask[y1:y2, x1:x2] = True
                    words.append([x1, y1, x2, y2])
            ink &= ~text_mask
            saturated &= ~text_mask

        r = max(1, int(round(3 * dpi / 72)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
        grown = cv2.dilate(ink.astype(np.uint8), kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(grown, 8)

        colour_boxes, plain_boxes = [], []
        for i in range(1, n):
            x, y, bw, bh, _ = stats[i]
            if bw * bh < 400:
                continue
            box = [x, y, x + bw, y + bh]
            if saturated[y:y + bh, x:x + bw].mean() >= min_saturated:
                colour_boxes.append(box)
            else:
                plain_boxes.append(box)

        colour_boxes = _merge_close_boxes(colour_boxes, int(round(12 * dpi / 72)))

        near = int(round(20 * dpi / 72))
        max_label_h = 30 * dpi / 72
        for _ in range(3):
            for box in colour_boxes:
                for p in plain_boxes:
                    if p[3] - p[1] > max_label_h or p[2] - p[0] > 1.2 * (box[2] - box[0]):
                        continue
                    if _boxes_within(box, p, near):
                        box[:] = [min(box[0], p[0]), min(box[1], p[1]), max(box[2], p[2]), max(box[3], p[3])]

        label_gap = int(round(8 * dpi / 72))
        for box in colour_boxes:
            core = list(box)
            for p in words:
                if p[3] - p[1] <= max_label_h and _boxes_within(core, p, label_gap):
                    box[:] = [min(box[0], p[0]), min(box[1], p[1]), max(box[2], p[2]), max(box[3], p[3])]

        min_side = min_side_pt * dpi / 72
        blocks = []
        for x1, y1, x2, y2 in colour_boxes:
            bw, bh = x2 - x1, y2 - y1
            if bw < min_side or bh < min_side or bw * bh < min_area:
                continue
            if bw * bh > 0.9 * h * w:
                continue
            roi = arr[y1:y2, x1:x2]
            sat_share = float(saturated[y1:y2, x1:x2].mean())
            palette_share = _palette_share(roi)
            if sat_share < min_saturated or palette_share < min_palette_share:
                continue
            blocks.append(FigureBlock(int(x1), int(y1), int(x2), int(y2), round(palette_share, 3), "opencv"))

        return sorted(blocks, key=lambda b: -b.area)


def _boxes_within(a, b, gap):
    return a[0] - gap < b[2] and b[0] - gap < a[2] and a[1] - gap < b[3] and b[1] - gap < a[3]


def _merge_close_boxes(boxes, gap):
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _boxes_within(boxes[i], boxes[j], gap):
                    a, b = boxes[i], boxes.pop(j)
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    merged = True
                    break
            if merged:
                break
    return boxes


def _palette_share(roi):
    q = roi.astype(np.int16) // 32
    key = (q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]).ravel()
    counts = np.bincount(key, minlength=512)
    return float(np.sort(counts)[::-1][:6].sum()) / max(key.size, 1)


def get_detector(
    force_mode: str | None = None,
    score_threshold: float = 0.7,
    device: str = "cpu",
) -> PubLayNetDetector | OpenCVDetector:
    """
    Return the best available layout detector.

    Args:
        force_mode: 'publaynet' / 'opencv' / None (auto)
        score_threshold: PubLayNet confidence threshold
        device: 'cuda' / 'cpu'
    """
    mode = force_mode or os.environ.get("BEYONDCOLOR_LAYOUT_MODE", "auto")

    if mode == "opencv":
        print("[BeyondColor] Layout mode: OpenCV")
        return OpenCVDetector()

    if mode in ("publaynet", "auto"):
        try:
            detector = PubLayNetDetector(score_threshold, device)
            print("[BeyondColor] Layout mode: PubLayNet")
            return detector
        except ImportError as e:
            if mode == "publaynet":
                raise
            print(f"[BeyondColor] PubLayNet unavailable ({e}), falling back to OpenCV")
            return OpenCVDetector()

    raise ValueError(f"Unknown layout mode: {mode}. Use 'publaynet', 'opencv', or 'auto'")
