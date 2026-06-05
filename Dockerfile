FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# систем депсы
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# питон депсы
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# детектрон2 депсы
# пребилд детекрон2
RUN pip install --no-cache-dir \
    detectron2 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html

# поддержка модели
RUN pip install --no-cache-dir "layoutparser[layoutmodels]"

# СЭМ для ГПУ
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# MobileSAM для CPU
RUN pip install --no-cache-dir mobile-sam

# установка весов
RUN mkdir -p /app/weights

# всегда качать
RUN wget -q --show-progress \
    https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt \
    -O /app/weights/mobile_sam.pt

# это для ГПУ
RUN wget -q --show-progress \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    -O /app/weights/sam_vit_b_01ec64.pth

# веса для модели PubLayNet
# но мы их тут загружаем чтобы образ был полностью готов
RUN python -c "
import layoutparser as lp
try:
    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')
    print('PubLayNet weights downloaded OK')
except Exception as e:
    print(f'PubLayNet preload: {e}')
" || true

# копирование кода приложения
COPY . /app/

# переменные окружения
ENV BEYONDCOLOR_LAYOUT_MODE=auto
ENV MOBILE_SAM_CHECKPOINT=/app/weights/mobile_sam.pt
ENV SAM_VIT_B_CHECKPOINT=/app/weights/sam_vit_b_01ec64.pth
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

# healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD wget -qO- http://localhost:8001/status || exit 1

CMD ["uvicorn", "api_integration:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
