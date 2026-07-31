from __future__ import annotations

from pathlib import Path

from PIL import Image, PngImagePlugin


PROCESSING_MARKER = "beyond_color_model"
PROCESSING_VERSION = "series-patterns-v1"


class AlreadyProcessedImageError(ValueError):
    pass


def validate_image_input(image: Image.Image, input_path: str | Path, output_path: str | Path) -> None:
    source = Path(input_path).resolve()
    target = Path(output_path).resolve()
    if source == target:
        raise ValueError("Input and output paths must be different to preserve the original image.")
    if image.info.get(PROCESSING_MARKER) == PROCESSING_VERSION:
        raise AlreadyProcessedImageError(
            "This image was already processed by BeyondColor. Use the original image to change patterns."
        )


def save_processed_image(image: Image.Image, output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".png":
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text(PROCESSING_MARKER, PROCESSING_VERSION)
        image.save(target, pnginfo=png_info)
        return
    image.save(target)
