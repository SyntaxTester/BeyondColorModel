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
        min_color_variance: float = 300.0,  
    ) -> list[FigureBlock]:
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]

        
        grey = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        
        thresh = cv2.adaptiveThreshold(
            grey, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )

        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blocks = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < min_area:
                continue

            
            if area > 0.9 * h * w:
                continue

            
            aspect = cw / ch if ch > 0 else 0
            if aspect > 12 or aspect < 0.08:
                continue

            
            roi = arr[y:y+ch, x:x+cw]
            variance = float(np.var(roi))
            if variance < min_color_variance:
                continue

            
            confidence = min(1.0, variance / 5000.0)
            blocks.append(FigureBlock(x, y, x + cw, y + ch, confidence, "opencv"))

        return sorted(blocks, key=lambda b: -b.area)


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
