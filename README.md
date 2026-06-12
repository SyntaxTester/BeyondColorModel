Установка

bashpython -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install Pillow numpy opencv-python PyMuPDF fastapi uvicorn python-multipart mobile-sam timm

mkdir weights
# Скачай weights/mobile_sam.pt:
# https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

python check.py  # проверка установки

Использование

bash# Изображение
python pipeline.py input.png output.png

# PDF
python pipeline.py input.pdf output.pdf

# API сервер
uvicorn api_integration:app --host 0.0.0.0 --port 8001
