from __future__ import annotations
import os
from pathlib import Path
import torch

def get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[loader] GPU: {name} ({vram:.1f} GB)")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[loader] Apple MPS")
        return "mps"
    print("[loader] CPU - MobileSAM")
    return "cpu"

DEVICE = get_device()

_WEIGHTS = Path(__file__).parent / "weights"
MOBILE_SAM_PATH = Path(os.environ.get("MOBILE_SAM_CHECKPOINT", str(_WEIGHTS / "mobile_sam.pt")))
SAM_VIT_B_PATH  = Path(os.environ.get("SAM_VIT_B_CHECKPOINT",  str(_WEIGHTS / "sam_vit_b_01ec64.pth")))

def _check_weight(path: Path, name: str, url: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing: {name}\nExpected: {path}\n"
            f"Download:\n  mkdir -p {path.parent}\n  wget '{url}' -O {path}"
        )
    print(f"[loader] {name} ({path.stat().st_size/1e6:.0f} MB)")

# Cached instances
_sam = None
_mask_gen = None
_layout_model = None

def load_sam(
    points_per_side=12,          # Сетка 12x12 (достаточно для графиков)
    points_per_batch=24,         # ОБЯЗАТЕЛЬНО: обработка по 32 точки за раз (спасает RAM)
    pred_iou_thresh=0.8,
    stability_score_thresh=0.8,
    min_mask_region_area=10     # Игнорим совсем мелкий шум (текст)
):
    global _sam, _mask_gen
    if _sam is not None:
        return _sam, _mask_gen

    if DEVICE == "cpu":
        _check_weight(MOBILE_SAM_PATH, "MobileSAM",
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt")
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        print("[loader] Loading MobileSAM...")
        sam = sam_model_registry["vit_t"](checkpoint=str(MOBILE_SAM_PATH))
    else:
        _check_weight(SAM_VIT_B_PATH, "SAM vit_b",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        print(f"[loader] Loading SAM vit_b on {DEVICE}...")
        sam = sam_model_registry["vit_b"](checkpoint=str(SAM_VIT_B_PATH))

    sam.to(device=DEVICE)
    sam.eval()
    
    # Теперь передаем points_per_batch в генератор
    _mask_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,  # он жадный
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    _sam = sam
    print(f"[loader] SAM ready (batch size: {points_per_batch}).")
    return _sam, _mask_gen

# ... функция load_publaynet без изменений ...