import io

import streamlit as st
from PIL import Image

from pixel_segmenter import apply_double_coding


st.set_page_config(page_title="BeyondColor", page_icon="🎨", layout="wide")
st.title("BeyondColor - тест разметки")
st.caption("Загрузите один или несколько графиков. Слева - оригинал, справа - размеченная версия, гуд лак")

opacity = st.slider("Прозрачность узора", min_value=50, max_value=100, value=100)

files = st.file_uploader(
    "Перетащите картинки сюда",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
)

if not files:
    st.info("Пока пусто. Киньте картинку графика выше")

for f in files:
    st.divider()
    st.subheader(f.name)

    try:
        img = Image.open(f).convert("RGB")
    except Exception:
        st.error(f"Не удалось открыть {f.name} - файл повреждён или это не картинка")
        continue

    with st.spinner("Обрабатываю…"):
        try:
            out = apply_double_coding(img, opacity=opacity)
        except Exception as e:
            st.error(f"Ошибка при обработке {f.name}: {e}")
            continue

    left, right = st.columns(2)
    left.image(img, caption="было", use_container_width=True)
    right.image(out, caption="стало", use_container_width=True)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    st.download_button(
        label=f"Скачать результат ({f.name})",
        data=buf.getvalue(),
        file_name=f"beyondcolor_{f.name.rsplit('.', 1)[0]}.png",
        mime="image/png",
        key=f.name,
    )