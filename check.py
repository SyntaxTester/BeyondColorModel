import torch, PIL, cv2, fitz
from mobile_sam import sam_model_registry
from pathlib import Path

print('torch:', torch.__version__)
print('PIL:', PIL.__version__)
print('cv2:', cv2.__version__)
print('fitz:', fitz.__version__)
print('weights:', Path('weights/mobile_sam.pt').exists())
print('device:', 'cuda' if torch.cuda.is_available() else 'cpu')
print('ALL OK')