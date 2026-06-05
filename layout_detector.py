from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


# типы данных

@dataclass
class FigureBlock:
    """A detected figure/chart region on a page."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    source: str        # 'publaynet' или 'opencv'

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


# PubLayNet детектор

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
                "layoutparser not installed, у тебя ошибка\n"
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
        """
    

        Args:
            image:    PIL картинка страницы (RGB)
            min_area: минимальная площадь фигуры в пикселях

        Returns:
            список FigureBlock отсортированный по убыванию площади
        """
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
            # клампим координаты до границ изображения
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(image.width, x2); y2 = min(image.height, y2)
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue
            blocks.append(FigureBlock(x1, y1, x2, y2, block.score, "publaynet"))

        return sorted(blocks, key=lambda b: -b.area)


# это короче если нет detectron2, он берет слабую модель opencv
class OpenCVDetector:
    """
    Lightweight figure detector using OpenCV contours.

    Finds large, non-text rectangular regions by:
    1. Converting to LAB colorspace for better color separation
    2. Edge detection
    3. Finding large filled contours (likely charts/diagrams)
    4. Filtering by aspect ratio and color variance

    No ML required. Works on CPU with no model download.
    Not as accurate as PubLayNet for complex layouts, but
    reliable enough for typical textbook pages.
    """

    def detect(
        self,
        image: Image.Image,
        min_area: int = 10_000,
        min_color_variance: float = 300.0,  # минимальная дисперсия цвета
    ) -> list[FigureBlock]:
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]

        # конвертируем в оттенки серого для детекции边缘
        grey = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # адаптивный порог для нахождения distinct regions
        thresh = cv2.adaptiveThreshold(
            grey, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )

        # расширяем для соединения близлежащих элементов
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # находим контуры
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blocks = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < min_area:
                continue

            # пропускаем регионы, которые занимают > 90% страницы (вероятно full-page art)
            if area > 0.9 * h * w:
                continue

            # пропускаем очень тонкие полосы (вероятно текстовые линии или правила)
            aspect = cw / ch if ch > 0 else 0
            if aspect > 12 or aspect < 0.08:
                continue

            # проверяем дисперсию цвета, диаграммы имеют higher variance чем текстовые блоки
            roi = arr[y:y+ch, x:x+cw]
            variance = float(np.var(roi))
            if variance < min_color_variance:
                continue

            # нормализуем прокси-оценку уверенности
            confidence = min(1.0, variance / 5000.0)
            blocks.append(FigureBlock(x, y, x + cw, y + ch, confidence, "opencv"))

        return sorted(blocks, key=lambda b: -b.area)

class PieChartDetector:
    def detect(
        self,
        image: Image.Image,
        min_radius_ratio: float = 0.12,
    ) -> list[FigureBlock]:
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]

        grey = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        grey = cv2.GaussianBlur(grey, (5, 5), 0)

        circles = cv2.HoughCircles(
            grey,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(h, w) // 4,
            param1=80,
            param2=30,
            minRadius=int(min(h, w) * min_radius_ratio),
            maxRadius=int(min(h, w) * 0.48),
        )

        if circles is None:
            return []

        blocks = []

        for circle in circles[0]:
            cx, cy, radius = circle

            cx = int(cx)
            cy = int(cy)
            radius = int(radius)

            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(w, cx + radius)
            y2 = min(h, cy + radius)

            roi = arr[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            variance = float(np.var(roi))
            if variance < 200:
                continue

            confidence = min(1.0, variance / 3000.0)

            blocks.append(
                FigureBlock(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    source="piechart",
                )
            )

        blocks.sort(key=lambda b: b.area, reverse=True)

        dedup = []
        for block in blocks:
            keep = True

            for other in dedup:
                ix1 = max(block.x1, other.x1)
                iy1 = max(block.y1, other.y1)
                ix2 = min(block.x2, other.x2)
                iy2 = min(block.y2, other.y2)

                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter = (ix2 - ix1) * (iy2 - iy1)
                union = block.area + other.area - inter

                if union > 0 and inter / union > 0.7:
                    keep = False
                    break

            if keep:
                dedup.append(block)

        return dedup

# авто выбор детектора

def get_detector(
    force_mode: str | None = None,
    score_threshold: float = 0.7,
    device: str = "cpu",
) -> PubLayNetDetector | OpenCVDetector:
    """
    Возвращает лучший доступный детектор макета

    Args:
        force_mode: 'publaynet' / 'opencv' / None (auto)
        score_threshold: PubLayNet confidence threshold
        device: 'cuda' / 'cpu'
    """
    mode = force_mode or os.environ.get("BEYONDCOLOR_LAYOUT_MODE", "auto")

    if mode == "opencv":
        print("[BeyondColor] Layout mode: OpenCV (fallback)")
        return HybridOpenCVDetector()

    if mode in ("publaynet", "auto"):
        try:
            detector = PubLayNetDetector(score_threshold, device)
            print("[BeyondColor] Layout mode: PubLayNet")
            return detector
        except ImportError as e:
            if mode == "publaynet":
                raise
            print(f"[BeyondColor] PubLayNet unavailable ({e}), falling back to OpenCV")
            return HybridOpenCVDetector()

    raise ValueError(f"Unknown layout mode: {mode}. Use 'publaynet', 'opencv', or 'auto'")


class HybridOpenCVDetector:
    def __init__(self):
        self.cv_detector = OpenCVDetector()
        self.pie_detector = PieChartDetector()

    def detect(self, image, *args, **kwargs):
        blocks = []

        blocks.extend(
            self.cv_detector.detect(image, *args, **kwargs)
        )

        blocks.extend(
            self.pie_detector.detect(image)
        )

        blocks.sort(key=lambda b: b.area, reverse=True)

        result = []

        for block in blocks:
            duplicate = False

            for existing in result:
                ix1 = max(block.x1, existing.x1)
                iy1 = max(block.y1, existing.y1)
                ix2 = min(block.x2, existing.x2)
                iy2 = min(block.y2, existing.y2)

                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter = (ix2 - ix1) * (iy2 - iy1)
                union = block.area + existing.area - inter

                if union > 0 and inter / union > 0.7:
                    duplicate = True
                    break

            if not duplicate:
                result.append(block)

        return result