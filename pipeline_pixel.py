from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

DEVICE = "cpu"
from layout_detector import get_detector, FigureBlock
from pixel_segmenter import apply_double_coding
from contrast_checker import audit_image_contrast


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
    figure_source: str
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
        tile_size: int = 14,
        min_figure_area: int = 4000,
        min_segment_area: int = 80,
        page_dpi: int = 150,
    ):
        self.pattern_opacity   = pattern_opacity
        self.tile_size         = tile_size
        self.min_figure_area   = min_figure_area
        self.min_segment_area  = min_segment_area
        self.page_dpi          = page_dpi

        print("[BeyondColor] Initialising pipeline (pixel mode)...")
        self._layout_detector = get_detector(force_mode=layout_mode, device=DEVICE)
        self.layout_mode = type(self._layout_detector).__name__
        print("[BeyondColor] Pipeline ready.\n")

    def process_image(self, input_path: str | Path, output_path: str | Path) -> PipelineReport:
        start_t = time.perf_counter()
        img = Image.open(input_path).convert("RGB")

        blocks = self._layout_detector.detect(img, min_area=self.min_figure_area)

        pie_blocks = [b for b in blocks if b.source == "piechart"]
        if pie_blocks:
            w_img, h_img = img.size
            blocks = [type(pie_blocks[0])(0, 0, w_img, h_img, 1.0, "piechart")]
        else:
            blocks = [
                b for b in blocks
                if b.width >= 50 and b.height >= 50
                and 0.1 < (b.width / b.height) < 10
            ]
            if len(blocks) > 3:
                blocks = blocks[:1]

        final_img = img.copy()
        all_fig_infos = []

        for block in blocks:
            crop = block.crop(img)
            processed_crop, fig_info_list, n_segs = self._process_figure(
                crop, page=1, bbox=block.bbox, source=block.source
            )
            if block.source == "piechart":
                final_img = processed_crop.convert("RGB")
            else:
                final_img.paste(processed_crop.convert("RGB").resize(
                    (block.x2 - block.x1, block.y2 - block.y1), Image.LANCZOS
                ), (block.x1, block.y1))
            all_fig_infos.extend(fig_info_list)

        final_img.save(output_path)

        violations = sum(1 for f in all_fig_infos if f.violation_before)
        fixed      = sum(1 for f in all_fig_infos if f.violation_before and not f.violation_after)
        compliance = max(0.0, 100.0 - (violations - fixed) * 10.0) if all_fig_infos else 100.0

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

    def process_pdf(self, input_path: str | Path, output_path: str | Path) -> PipelineReport:
        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t = time.perf_counter()
        doc = fitz.open(str(input_path))
        all_fig_infos: list[FigureInfo] = []
        total_segs = 0

        for page_num, page in enumerate(doc):
            print(f"  Page {page_num + 1}/{doc.page_count}...", end=" ")
            mat  = fitz.Matrix(self.page_dpi / 72, self.page_dpi / 72)
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            figure_blocks = self._layout_detector.detect(page_img, min_area=self.min_figure_area)
            print(f"{len(figure_blocks)} figure(s)")

            for fig_block in figure_blocks:
                figure_crop = fig_block.crop(page_img)
                processed_crop, fig_info_list, n_segs = self._process_figure(
                    figure_crop, page=page_num + 1, bbox=fig_block.bbox, source=fig_block.source,
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
        compliance = max(0.0, 100.0 - (violations - fixed) * 5.0)

        report = PipelineReport(
            input_path=str(input_path),
            output_path=str(output_path),
            layout_mode=self.layout_mode,
            device=DEVICE,
            pages_processed=len(all_fig_infos),
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

    def process_bulk(self, input_paths: list[str | Path], output_dir: str | Path) -> list[PipelineReport]:
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

    def _process_figure(
        self,
        figure_img: Image.Image,
        page: int = 1,
        bbox: tuple = (0, 0, 0, 0),
        source: str = "unknown",
    ) -> tuple[Image.Image, list[FigureInfo], int]:

        audit_before = audit_image_contrast(figure_img, sample_count=150)

        # Pixel double coding - красим каждый пиксель по цвету
        processed = apply_double_coding(
            figure_img,
            opacity=self.pattern_opacity,
        )

        audit_after = audit_image_contrast(processed, sample_count=150)

        fig_info = FigureInfo(
            page=page, bbox=bbox, figure_source=source,
            segments_found=1,
            patterns_applied=["pixel_double_coding"],
            contrast_before=round(audit_before.mean_ratio, 2),
            contrast_after=round(audit_after.mean_ratio, 2),
            violation_before=not audit_before.overall_pass,
            violation_after=not audit_after.overall_pass,
            segments=[],
        )
        return processed, [fig_info], 1


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

def process_bulk(input_paths: list[str], output_dir: str, **kwargs) -> list[PipelineReport]:
    return _get_pipeline(**kwargs).process_bulk(input_paths, output_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <input> <output>")
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])

    if inp.is_dir():
        pdfs = list(inp.glob("*.pdf"))
        pipeline = BeyondColorPipeline()
        reports = pipeline.process_bulk(pdfs, out)
        print(json.dumps({"total": len(reports)}, indent=2))
    elif inp.suffix.lower() == ".pdf":
        report = process_pdf(str(inp), str(out))
        out.with_suffix(".report.json").write_text(json.dumps(report.to_dict(), indent=2, default=str))
    elif inp.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        report = process_image(str(inp), str(out))
        report.print_summary()
    else:
        print("Unsupported file type.")
        sys.exit(1)