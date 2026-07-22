from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

DEVICE = "cpu"
from image_provenance import save_processed_image, validate_image_input
from layout_detector import get_detector
from pixel_segmenter import apply_double_coding_with_report
from contrast_checker import audit_image_contrast


def _supported_image_extensions() -> set[str]:
    Image.init()
    return set(Image.registered_extensions()) - {".pdf"}


IMAGE_SUFFIXES = _supported_image_extensions()
PROJECT_DIR = Path(__file__).resolve().parent
# Все результаты массового прогона тестовых картинок храним в одном каталоге,
# чтобы его можно было целиком передать или архивировать.
TEST_RESULTS_DIR = PROJECT_DIR / "processed_tests"
TEST_IMAGE_STEM = re.compile(r"^test(?:[_ -]?(\d+))?$", re.IGNORECASE)


def discover_test_images(input_dir: str | Path) -> list[Path]:
    """Return the existing numbered test images in natural numeric order.

    Test numbers are allowed to have gaps: for example, ``test14.png``,
    ``test16.png`` and ``test19.png`` are a valid suite.  Service files such
    as ``test_compilation.png`` are deliberately excluded.
    """
    input_dir = Path(input_dir)
    test_paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and TEST_IMAGE_STEM.fullmatch(path.stem)
    ]

    def sort_key(path: Path) -> tuple[int, str, str]:
        match = TEST_IMAGE_STEM.fullmatch(path.stem)
        assert match is not None
        number = match.group(1)
        return (int(number) if number is not None else -1, path.suffix.lower(), path.name.lower())

    return sorted(test_paths, key=sort_key)


def load_input_image(input_path: str | Path) -> Image.Image:
    """Загружает изображение, включая файлы с неверным расширением."""
    try:
        with Image.open(input_path) as source:
            source.load()
            return source.convert("RGB")
    except OSError as pillow_error:
        import cv2

        bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise pillow_error
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def create_contact_sheet(
    image_paths: list[Path],
    output_path: str | Path,
    columns: int = 4,
) -> Path:
    """Создаёт один обзорный PNG из обработанных изображений."""
    thumbnail_size = (280, 200)
    cell_width = thumbnail_size[0] + 16
    cell_height = thumbnail_size[1] + 38
    rows = max(1, (len(image_paths) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell_width + 16, rows * cell_height + 16), "white")
    draw = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            thumbnail = ImageOps.contain(source.convert("RGB"), thumbnail_size, Image.LANCZOS)
        column = index % columns
        row = index // columns
        x = 16 + column * cell_width
        y = 16 + row * cell_height
        sheet.paste(thumbnail, (x, y))
        draw.text((x, y + thumbnail_size[1] + 6), image_path.name, fill="black")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")
    return output_path


@dataclass
class SegmentInfo:
    color: str
    pattern: str
    area_px: int
    iou: float | None

@dataclass
class FigureInfo:
    page: int
    bbox: tuple
    figure_source: str
    chart_kind: str
    segments_found: int
    patterns_applied: list[str]
    contrast_before: float
    contrast_after:  float
    violation_before: bool
    violation_after:  bool
    segments: list[SegmentInfo] = field(default_factory=list)

@dataclass
class PipelineReport:
    input_path:          str
    output_path:         str
    layout_mode:         str
    device:              str
    pages_processed:     int
    figures_found:       int
    segments_total:      int
    violations_detected: int
    violations_fixed:    int
    compliance_score:    float
    processing_time_ms:  int
    figures: list[FigureInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def print_summary(self) -> None:
        print(f"""

     BeyondColor Processing Report    

  Input:       {self.input_path[:35]:<35} 
  Device:      {self.device:<35} 
  Layout:      {self.layout_mode:<35} 

  Pages:       {self.pages_processed:<35} 
  Figures:     {self.figures_found:<35} 
  Segments:    {self.segments_total:<35}

  Violations:  {self.violations_detected:<35} 
  Fixed:       {self.violations_fixed:<35} 
  Compliance:  {f'{self.compliance_score:.1f}%':<35} 
  Time:        {f'{self.processing_time_ms}ms':<35}
""")


class BeyondColorPipeline:
    def __init__(
        self,
        layout_mode: str = "opencv",
        pattern_opacity: int = 100,
        min_figure_area: int = 4000,
        page_dpi: int = 150,
    ):
        self.pattern_opacity = pattern_opacity
        self.min_figure_area = min_figure_area
        self.page_dpi = page_dpi
        self._layout_mode_request = layout_mode
        self._layout_detector = None
        self.layout_mode = "whole-image"

        print("[BeyondColor] Initialising pipeline (pixel mode)...")
        print("[BeyondColor] Pipeline ready.\n")

    def _get_layout_detector(self):
        if self._layout_detector is None:
            self._layout_detector = get_detector(
                force_mode=self._layout_mode_request,
                device=DEVICE,
            )
            self.layout_mode = type(self._layout_detector).__name__
        return self._layout_detector

    def process_image(self, input_path: str | Path, output_path: str | Path) -> PipelineReport:
        start_t = time.perf_counter()
        source_img = load_input_image(input_path)
        validate_image_input(source_img, input_path, output_path)
        img = source_img

        final_img, all_fig_infos, _ = self._process_figure(
            img,
            page=1,
            bbox=(0, 0, img.width, img.height),
            source="image",
        )

        save_processed_image(final_img, output_path)

        violations = sum(1 for f in all_fig_infos if f.violation_before)
        fixed      = sum(1 for f in all_fig_infos if f.violation_before and not f.violation_after)
        compliance = _compliance_score(all_fig_infos)

        return PipelineReport(
            input_path=str(input_path),
            output_path=str(output_path),
            layout_mode="whole-image",
            device=DEVICE,
            pages_processed=1,
            figures_found=len(all_fig_infos),
            segments_total=sum(f.segments_found for f in all_fig_infos),
            violations_detected=violations,
            violations_fixed=fixed,
            compliance_score=round(compliance, 1),
            processing_time_ms=int((time.perf_counter() - start_t) * 1000),
            figures=all_fig_infos
        )

    def process_pdf(self, input_path: str | Path, output_path: str | Path) -> PipelineReport:
        try:
            import fitz
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "PDF processing requires PyMuPDF. Install it with: pip install PyMuPDF"
            ) from error

        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t = time.perf_counter()
        doc = fitz.open(str(input_path))
        page_count = doc.page_count
        all_fig_infos: list[FigureInfo] = []
        total_segs = 0
        detector = self._get_layout_detector()

        for page_num, page in enumerate(doc):
            print(f"  Page {page_num + 1}/{page_count}...", end=" ")
            mat  = fitz.Matrix(self.page_dpi / 72, self.page_dpi / 72)
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            figure_blocks = detector.detect(page_img, min_area=self.min_figure_area)
            print(f"{len(figure_blocks)} figure(s)")

            for fig_block in figure_blocks:
                figure_crop = fig_block.crop(page_img)
                processed_crop, fig_info_list, n_segs = self._process_figure(
                    figure_crop,
                    page=page_num + 1,
                    bbox=fig_block.bbox,
                    source=fig_block.source,
                )
                all_fig_infos.extend(fig_info_list)
                total_segs += n_segs

                scale_x = page.rect.width  / page_img.width
                scale_y = page.rect.height / page_img.height
                pdf_rect = fitz.Rect(
                    fig_block.x1 * scale_x, fig_block.y1 * scale_y,
                    fig_block.x2 * scale_x, fig_block.y2 * scale_y,
                )
                buf = io.BytesIO()
                processed_crop.convert("RGB").save(buf, format="PNG")
                buf.seek(0)
                page.insert_image(pdf_rect, stream=buf.read(), keep_proportion=True)

        doc.save(str(output_path), garbage=4, deflate=True)
        doc.close()

        violations = sum(1 for f in all_fig_infos if f.violation_before)
        fixed      = sum(1 for f in all_fig_infos if f.violation_before and not f.violation_after)
        compliance = _compliance_score(all_fig_infos)

        report = PipelineReport(
            input_path=str(input_path),
            output_path=str(output_path),
            layout_mode=self.layout_mode,
            device=DEVICE,
            pages_processed=page_count,
            figures_found=len(all_fig_infos),
            segments_total=total_segs,
            violations_detected=violations,
            violations_fixed=fixed,
            compliance_score=round(compliance, 1),
            processing_time_ms=int((time.perf_counter() - t) * 1000),
            figures=all_fig_infos,
        )
        report.print_summary()
        return report

    def process_bulk(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        continue_on_error: bool = False,
    ) -> list[PipelineReport]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        for i, in_path in enumerate(input_paths):
            in_path = Path(in_path)
            out_path = output_dir / in_path.name
            print(f"\n[{i+1}/{len(input_paths)}] {in_path.name}")
            try:
                if in_path.suffix.lower() == ".pdf":
                    report = self.process_pdf(in_path, out_path)
                elif in_path.suffix.lower() in IMAGE_SUFFIXES:
                    report = self.process_image(in_path, out_path)
                else:
                    print("  SKIP: unsupported file type")
                    continue
                reports.append(report)
            except Exception as e:
                if not continue_on_error:
                    raise
                print(f"  ERROR: {e}")
        return reports

    def process_test_suite(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        compilation_path: str | Path | None = None,
    ) -> tuple[list[PipelineReport], Path]:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        test_paths = discover_test_images(input_dir)
        if not test_paths:
            raise ValueError(f"No test images found in {input_dir}")

        reports = self.process_bulk(test_paths, output_dir)
        output_paths = [
            Path(report.output_path)
            for report in reports
            if Path(report.output_path).exists()
        ]
        contact_sheet = create_contact_sheet(
            output_paths,
            compilation_path or output_dir / "test_compilation.png",
        )
        return reports, contact_sheet

    def _process_figure(
        self,
        figure_img: Image.Image,
        page: int = 1,
        bbox: tuple = (0, 0, 0, 0),
        source: str = "unknown",
    ) -> tuple[Image.Image, list[FigureInfo], int]:

        audit_before = audit_image_contrast(figure_img)
        coding = apply_double_coding_with_report(
            figure_img,
            opacity=self.pattern_opacity,
        )
        processed = coding.image

        audit_after = audit_image_contrast(processed)
        patterns = sorted({segment.pattern for segment in coding.segments})

        fig_info = FigureInfo(
            page=page, bbox=bbox, figure_source=source, chart_kind=coding.chart_kind,
            segments_found=len(coding.segments),
            patterns_applied=patterns,
            contrast_before=round(audit_before.mean_ratio, 2),
            contrast_after=round(audit_after.mean_ratio, 2),
            violation_before=not audit_before.overall_pass,
            violation_after=not audit_after.overall_pass,
            segments=[
                SegmentInfo(
                    color=segment.color,
                    pattern=segment.pattern,
                    area_px=segment.area_px,
                    iou=None,
                )
                for segment in coding.segments
            ],
        )
        return processed, [fig_info], len(coding.segments)


def _compliance_score(figures: list[FigureInfo]) -> float:
    if not figures:
        return 100.0
    passing = sum(1 for figure in figures if not figure.violation_after)
    return 100.0 * passing / len(figures)

def _get_pipeline(**kwargs) -> BeyondColorPipeline:
    return BeyondColorPipeline(**kwargs)

def process_image(input_path: str, output_path: str, **kwargs) -> PipelineReport:
    return _get_pipeline(**kwargs).process_image(input_path, output_path)

def process_pdf(input_path: str, output_path: str, **kwargs) -> PipelineReport:
    return _get_pipeline(**kwargs).process_pdf(input_path, output_path)

def process_bulk(
    input_paths: list[str],
    output_dir: str,
    continue_on_error: bool = False,
    **kwargs,
) -> list[PipelineReport]:
    return _get_pipeline(**kwargs).process_bulk(
        input_paths,
        output_dir,
        continue_on_error=continue_on_error,
    )


def process_test_suite(
    input_dir: str | Path,
    output_dir: str | Path,
    compilation_path: str | Path | None = None,
    **kwargs,
) -> tuple[list[PipelineReport], Path]:
    return _get_pipeline(**kwargs).process_test_suite(
        input_dir,
        output_dir,
        compilation_path,
    )


if __name__ == "__main__":
    import sys
    requested_input = Path(sys.argv[1]) if len(sys.argv) >= 2 else None
    missing_test_alias = (
        requested_input is not None
        and not requested_input.exists()
        and TEST_IMAGE_STEM.fullmatch(requested_input.stem) is not None
    )
    if len(sys.argv) == 1 or sys.argv[1] == "--test-suite" or missing_test_alias:
        if missing_test_alias:
            print(
                f"[BeyondColor] {requested_input.name} is absent; "
                "processing all available test images instead."
        )
        output_dir = (
            Path(sys.argv[2])
            if len(sys.argv) >= 3 and sys.argv[1] == "--test-suite"
            else TEST_RESULTS_DIR
        )
        reports, contact_sheet = process_test_suite(
            PROJECT_DIR,
            output_dir,
        )
        print(json.dumps({"total": len(reports), "contact_sheet": str(contact_sheet)}, indent=2))
        sys.exit(0)

    if len(sys.argv) < 3:
        print("Usage: python pipeline_pixel.py <input> <output>")
        print("   or: python pipeline_pixel.py --test-suite [output_dir]")
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])

    if inp.is_dir():
        files = [path for path in inp.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES | {".pdf"}]
        pipeline = BeyondColorPipeline()
        reports = pipeline.process_bulk(files, out)
        print(json.dumps({"total": len(reports)}, indent=2))
    elif inp.suffix.lower() == ".pdf":
        report = process_pdf(str(inp), str(out))
        out.with_suffix(".report.json").write_text(json.dumps(report.to_dict(), indent=2, default=str))
    elif inp.suffix.lower() in IMAGE_SUFFIXES:
        report = process_image(str(inp), str(out))
        report.print_summary()
    else:
        supported = ", ".join(sorted(IMAGE_SUFFIXES))
        print(f"Unsupported file type: {inp.suffix}")
        print(f"Supported: .pdf, {supported}")
        sys.exit(1)
