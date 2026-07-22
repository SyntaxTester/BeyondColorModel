from __future__ import annotations

import json
from pathlib import Path

from pipeline_pixel import (
    BeyondColorPipeline,
    IMAGE_SUFFIXES,
    PROJECT_DIR,
    TEST_RESULTS_DIR,
    PipelineReport,
    process_bulk,
    process_image,
    process_pdf,
    process_test_suite,
)


def main() -> int:
    import sys

    arguments = sys.argv[1:]
    if not arguments or arguments[0] == "--test-suite":
        output_dir = Path(arguments[1]) if len(arguments) >= 2 else TEST_RESULTS_DIR
        reports, contact_sheet = process_test_suite(
            PROJECT_DIR,
            output_dir,
            PROJECT_DIR / "output2.png",
        )
        print(json.dumps({"total": len(reports), "contact_sheet": str(contact_sheet)}, indent=2))
        return 0

    if len(arguments) < 2:
        print("Usage: python pipeline.py <input> <output>")
        print("   or: python pipeline.py --test-suite [output_dir]")
        return 1

    input_path = Path(arguments[0])
    output_path = Path(arguments[1])

    if input_path.is_dir():
        paths = [
            path
            for path in input_path.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES | {".pdf"}
        ]
        reports = process_bulk(paths, output_path)
        print(json.dumps({"total": len(reports)}, indent=2))
        return 0

    if input_path.suffix.lower() == ".pdf":
        report = process_pdf(str(input_path), str(output_path))
        output_path.with_suffix(".report.json").write_text(
            json.dumps(report.to_dict(), indent=2, default=str)
        )
        return 0

    if input_path.suffix.lower() in IMAGE_SUFFIXES:
        report = process_image(str(input_path), str(output_path))
        report.print_summary()
        return 0

    supported = ", ".join(sorted(IMAGE_SUFFIXES))
    print(f"Unsupported file type: {input_path.suffix}")
    print(f"Supported: .pdf, {supported}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
