"""
BeyondColor — Full Pipeline

PubLayNet → SAM → Double Coding

Single image:
    result = process_image("chart.png", "chart_out.png")

Single PDF:
    result = process_pdf("textbook.pdf", "textbook_out.pdf")

Bulk PDFs:
    results = process_bulk(["a.pdf", "b.pdf"], output_dir="out/")

The pipeline initialises models once and reuses them across all documents.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import fitz                   # PyMuPDF
import numpy as np
from PIL import Image

from model_loader import load_sam, get_device, DEVICE
from layout_detector import get_detector, FigureBlock
from segmenter import SAMSegmenter, deduplicate_segments, ColoredSegment
from pattern_renderer import render_patterns_on_segments
from contrast_checker import audit_image_contrast


# ── Report types ──────────────────────────────────────────────────────────────

@dataclass
class SegmentInfo:
    color: str
    pattern: str
    area_px: int
    iou: float

@dataclass
class FigureInfo:
    page: int
    bbox: tuple
    figure_source: str          # 'publaynet' | 'opencv'
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
╔══════════════════════════════════════╗
║     BeyondColor Processing Report    ║
╠══════════════════════════════════════╣
║  Input:       {self.input_path[:35]:<35} ║
║  Device:      {self.device:<35} ║
║  Layout:      {self.layout_mode:<35} ║
╠══════════════════════════════════════╣
║  Pages:       {self.pages_processed:<35} ║
║  Figures:     {self.figures_found:<35} ║
║  Segments:    {self.segments_total:<35} ║
╠══════════════════════════════════════╣
║  Violations:  {self.violations_detected:<35} ║
║  Fixed:       {self.violations_fixed:<35} ║
║  Compliance:  {f'{self.compliance_score:.1f}%':<35} ║
║  Time:        {f'{self.processing_time_ms}ms':<35} ║
╚══════════════════════════════════════╝""")


# ── Pipeline class ────────────────────────────────────────────────────────────

class BeyondColorPipeline:
    """
    Initialise once, process many documents.

    pipeline = BeyondColorPipeline()
    pipeline.process_pdf("a.pdf", "a_out.pdf")
    pipeline.process_pdf("b.pdf", "b_out.pdf")
    """

    def __init__(
        self,
        layout_mode: str = "opencv",       # 'publaynet' | 'opencv' | 'auto'
        sam_points_per_side: int = 16,   # lower = faster
        pattern_opacity: int = 100,
        tile_size: int = 14,
        min_figure_area: int = 4000,    # px² on rendered page
        min_segment_area: int = 80,
        page_dpi: int = 150,             # PDF render resolution
    ):
        self.pattern_opacity   = pattern_opacity
        self.tile_size         = tile_size
        self.min_figure_area   = min_figure_area
        self.min_segment_area  = min_segment_area
        self.page_dpi          = page_dpi

        # Load models
        print("[BeyondColor] Initialising pipeline...")
        self._layout_detector = get_detector(force_mode=layout_mode, device=DEVICE)
        _, self._mask_generator = load_sam(points_per_side=sam_points_per_side)
        self._segmenter = SAMSegmenter(self._mask_generator)

        self.layout_mode = getattr(self._layout_detector, '__class__', type(self._layout_detector)).__name__
        print("[BeyondColor] Pipeline ready.\n")

    # ── Single image ──────────────────────────────────────────────────────────

# ── Single image ──────────────────────────────────────────────────────────

    def process_image(self, input_path: str | Path, output_path: str | Path) -> PipelineReport:
        start_t = time.perf_counter()
        img = Image.open(input_path).convert("RGB")
        
        # 1. Детекция блоков
        blocks = self._layout_detector.detect(img, min_area=self.min_figure_area)

        # Фильтруем мелкие и вытянутые блоки
        blocks = [
            b for b in blocks
            if b.width >= 50 and b.height >= 50
            and 0.1 < (b.width / b.height) < 10
        ]

        # Если блоков много — OpenCV дробит один график на части.
        # Берём только самый большой (весь график целиком).
        if len(blocks) > 3:
            blocks = blocks[:1]

        final_img = img.copy()
        all_fig_infos = []

        for block in blocks:

            # Обрабатываем только то, что похоже на график
            crop = block.crop(img)
            
            processed_crop, fig_info_list, n_segs = self._process_figure(
                crop, page=1, bbox=block.bbox, source=block.source
            )
            
            # Вклеиваем результат обратно на оригинал
            final_img.paste(processed_crop, (block.x1, block.y1))
            all_fig_infos.extend(fig_info_list) # Используем extend, так как возвращается список

        final_img.save(output_path)
        
        # Считаем статистику для отчета
        violations = sum(1 for f in all_fig_infos if f.violation_before)
        fixed      = sum(1 for f in all_fig_infos if f.violation_before and not f.violation_after)
        compliance = max(0.0, 100.0 - (violations - fixed) * 10.0) if all_fig_infos else 100.0
        
        # Правильный возврат репорта (совпадает с твоим dataclass)
        return PipelineReport(
            input_path=str(input_path),
            output_path=str(output_path),
            layout_mode=self.layout_mode,
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
    # ... вернуть репорт

    # ── Single PDF ────────────────────────────────────────────────────────────

    def process_pdf(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> PipelineReport:
        """
        Process a PDF: detect figures on each page, apply double coding,
        replace images back in the PDF, save.
        """
        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t = time.perf_counter()
        doc = fitz.open(str(input_path))

        all_fig_infos: list[FigureInfo] = []
        total_segs = 0

        for page_num, page in enumerate(doc):
            print(f"  Page {page_num + 1}/{doc.page_count}...", end=" ")

            # Render page to image
            mat  = fitz.Matrix(self.page_dpi / 72, self.page_dpi / 72)
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Detect figures on this page
            figure_blocks: list[FigureBlock] = self._layout_detector.detect(
                page_img, min_area=self.min_figure_area
            )
            print(f"{len(figure_blocks)} figure(s)")

            for fig_block in figure_blocks:
                figure_crop = fig_block.crop(page_img)

                # Segment + apply patterns
                processed_crop, fig_info_list, n_segs = self._process_figure(
                    figure_crop, page=page_num + 1, bbox=fig_block.bbox,
                    source=fig_block.source,
                )
                all_fig_infos.extend(fig_info_list)
                total_segs += n_segs

                # Scale bbox back to PDF units and replace the region
                scale_x = page.rect.width  / page_img.width
                scale_y = page.rect.height / page_img.height
                pdf_rect = fitz.Rect(
                    fig_block.x1 * scale_x, fig_block.y1 * scale_y,
                    fig_block.x2 * scale_x, fig_block.y2 * scale_y,
                )

                # Embed processed figure as PNG
                buf = io.BytesIO()
                processed_crop.convert("RGB").save(buf, format="PNG")
                buf.seek(0)

                # Insert image onto page (covers original figure)
                page.insert_image(pdf_rect, stream=buf.read(), keep_proportion=True)

        doc.save(str(output_path), garbage=4, deflate=True)
        doc.close()

        violations = sum(1 for f in all_fig_infos if f.violation_before)
        fixed      = sum(1 for f in all_fig_infos if f.violation_before and not f.violation_after)
        compliance = max(0.0, 100.0 - (violations - fixed) * 5.0)

        report = PipelineReport(
            input_path=str(input_path),
            output_path=str(output_path),
            layout_mode=self.layout_mode,
            device=DEVICE,
            pages_processed=doc.page_count if not doc.is_closed else len(all_fig_infos),
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

    # ── Bulk ──────────────────────────────────────────────────────────────────

    def process_bulk(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
    ) -> list[PipelineReport]:
        """Process multiple PDFs sequentially (models stay loaded)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        reports = []
        for i, in_path in enumerate(input_paths):
            in_path = Path(in_path)
            out_path = output_dir / in_path.name
            print(f"\n[{i+1}/{len(input_paths)}] {in_path.name}")
            try:
                report = self.process_pdf(in_path, out_path)
                reports.append(report)
            except Exception as e:
                print(f"  ERROR: {e}")
        return reports

    # ── Core: figure → segments → patterns ───────────────────────────────────

    def _process_figure(
        self,
        figure_img: Image.Image,
        page: int = 1,
        bbox: tuple = (0, 0, 0, 0),
        source: str = "unknown",
    ) -> tuple[Image.Image, list[FigureInfo], int]:
        """
        Run SAM segmentation + pattern rendering on one figure crop.

        Returns:
            (processed_image, [FigureInfo], segment_count)
        """
        # Contrast before
        audit_before = audit_image_contrast(figure_img, sample_count=150)

        # Segment
        raw_segments = self._segmenter.segment(
            figure_img, min_area=self.min_segment_area
        )
        segments = deduplicate_segments(raw_segments)

        # Apply patterns
        if segments:
            processed = render_patterns_on_segments(
                figure_img,
                segments,
                pattern_opacity=self.pattern_opacity,
                tile_size=self.tile_size,
            )
        else:
            # Fallback: whole-image overlay if SAM found nothing
            from double_coding import apply_double_coding
            result = apply_double_coding(figure_img, pattern_opacity=self.pattern_opacity)
            processed = result.output_image

        # Contrast after
        audit_after = audit_image_contrast(processed, sample_count=150)

        fig_info = FigureInfo(
            page=page,
            bbox=bbox,
            figure_source=source,
            segments_found=len(segments),
            patterns_applied=list({s.pattern for s in segments}),
            contrast_before=round(audit_before.mean_ratio, 2),
            contrast_after= round(audit_after.mean_ratio,  2),
            violation_before=not audit_before.overall_pass,
            violation_after= not audit_after.overall_pass,
            segments=[
                SegmentInfo(
                    color=s.color_name,
                    pattern=s.pattern,
                    area_px=s.area,
                    iou=round(s.predicted_iou, 3),
                )
                for s in segments[:20]   # top 20 in report
            ],
        )

        return processed, [fig_info], len(segments)


# ── Convenience functions ─────────────────────────────────────────────────────

_default_pipeline: BeyondColorPipeline | None = None

def _get_pipeline(**kwargs) -> BeyondColorPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = BeyondColorPipeline(**kwargs)
    return _default_pipeline


def process_image(input_path: str, output_path: str, **kwargs) -> PipelineReport:
    return _get_pipeline(**kwargs).process_image(input_path, output_path)

def process_pdf(input_path: str, output_path: str, **kwargs) -> PipelineReport:
    return _get_pipeline(**kwargs).process_pdf(input_path, output_path)

def process_bulk(
    input_paths: list[str],
    output_dir: str,
    **kwargs,
) -> list[PipelineReport]:
    return _get_pipeline(**kwargs).process_bulk(input_paths, output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python pipeline.py <input.pdf>   <output.pdf>")
        print("  python pipeline.py <input.png>   <output.png>")
        print("  python pipeline.py <input_dir/>  <output_dir/>")
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])

    if inp.is_dir():
        pdfs = list(inp.glob("*.pdf"))
        print(f"Bulk: {len(pdfs)} PDFs")
        pipeline = BeyondColorPipeline()
        reports = pipeline.process_bulk(pdfs, out)
        summary = {
            "total": len(reports),
            "mean_compliance": round(
                sum(r.compliance_score for r in reports) / len(reports), 1
            ) if reports else 0,
        }
        print(json.dumps(summary, indent=2))

    elif inp.suffix.lower() == ".pdf":
        report = process_pdf(str(inp), str(out))
        out_json = out.with_suffix(".report.json")
        out_json.write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"Report saved: {out_json}")

    elif inp.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        report = process_image(str(inp), str(out))
        report.print_summary()

    else:
        print("Unsupported file type. Use .pdf, .png, .jpg, or a directory.")
        sys.exit(1)