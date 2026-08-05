import base64
import io

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from image_provenance import PROCESSING_MARKER, PROCESSING_VERSION
from pixel_segmenter import apply_double_coding

OPACITY_MIN = 50
OPACITY_MAX = 100
PREVIEW_WIDTH = 900


st.set_page_config(page_title="BeyondColor", page_icon="🎨", layout="wide")
st.title("BeyondColor - тест разметки")
st.caption("Загрузите один или несколько графиков. Слева - оригинал, справа - размеченная версия, гуд лак")


@st.cache_data(show_spinner=False, max_entries=8)
def _prepare(raw):
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        already = source.info.get(PROCESSING_MARKER) == PROCESSING_VERSION
        img = source.convert("RGB")
    if already:
        return True, None, None, None
    base = np.asarray(img, dtype=np.uint8)
    low = np.asarray(apply_double_coding(img, opacity=OPACITY_MIN), dtype=np.uint8)
    high = np.asarray(apply_double_coding(img, opacity=OPACITY_MAX), dtype=np.uint8)
    return False, base, low, high


def _to_url(arr, width):
    img = Image.fromarray(arr)
    if img.width > width:
        img = img.resize((width, max(1, round(img.height * width / img.width))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), img.width, img.height


TEMPLATE = """
<style>
 .bcwrap{font-family:"Source Sans Pro",sans-serif;color:#fafafa}
 .bcrow{display:flex;gap:16px;align-items:flex-start}
 .bccol{flex:1 1 0;min-width:0}
 .bccap{font-size:14px;color:#a3a8b8;margin-top:6px}
 .bcslide{width:100%;margin:0 0 14px 0}
 .bclab{font-size:14px;color:#fafafa;margin-bottom:2px}
 .bcval{color:#ff4b4b;font-weight:600}
 .bcbtn{display:inline-block;margin-top:12px;padding:7px 16px;border:1px solid #4a4f5c;
        border-radius:8px;color:#fafafa;text-decoration:none;font-size:14px;background:#262730}
 .bcbtn:hover{border-color:#ff4b4b;color:#ff4b4b}
 img,canvas{width:100%;max-width:__DW__px;display:block;border-radius:4px}
 .bccol{max-width:__DW__px}
</style>
<div class="bcwrap">
  <div class="bclab">Прозрачность узора: <span class="bcval" id="VAL_ID">100</span></div>
  <input class="bcslide" type="range" min="__MIN__" max="__MAX__" value="__MAX__" id="RNG_ID">
  <div class="bcrow">
    <div class="bccol">
      <img src="__BASE__">
      <div class="bccap">было</div>
    </div>
    <div class="bccol">
      <canvas id="CNV_ID" width="__W__" height="__H__"></canvas>
      <div class="bccap">стало</div>
    </div>
  </div>
  <a class="bcbtn" id="DL_ID" download="__FNAME__">Скачать то, что вижу</a>
</div>
<script>
(function(){
 function fit(){
   var h=document.body.scrollHeight;
   window.parent.postMessage({type:"streamlit:setFrameHeight",height:h+12},"*");
 }
 window.addEventListener("load",fit);
 window.addEventListener("resize",fit);
 setInterval(fit,700);
 var rng=document.getElementById("RNG_ID"),val=document.getElementById("VAL_ID"),
     cnv=document.getElementById("CNV_ID"),dl=document.getElementById("DL_ID"),
     ctx=cnv.getContext("2d"),lo=new Image(),hi=new Image(),ready=0;
 function draw(){
   if(ready<2)return;
   var t=(rng.value-__MIN__)/(__MAX__-__MIN__);
   ctx.clearRect(0,0,cnv.width,cnv.height);
   ctx.globalAlpha=1;ctx.drawImage(lo,0,0,cnv.width,cnv.height);
   ctx.globalAlpha=t;ctx.drawImage(hi,0,0,cnv.width,cnv.height);
   ctx.globalAlpha=1;
   val.textContent=rng.value;
 }
 function done(){ready++;draw();fit();}
 lo.onload=done;hi.onload=done;
 lo.src="__LOW__";hi.src="__HIGH__";
 rng.addEventListener("input",draw);
 dl.addEventListener("click",function(){dl.href=cnv.toDataURL("image/png");});
})();
</script>
"""


def _viewer(base, low, high, fname, key):
    b_url, w, h = _to_url(base, PREVIEW_WIDTH)
    l_url, _, _ = _to_url(low, PREVIEW_WIDTH)
    h_url, _, _ = _to_url(high, PREVIEW_WIDTH)
    disp_w = min(int(round(w * 1.6)), PREVIEW_WIDTH)
    disp_h = int(round(h * disp_w / max(w, 1)))
    html = TEMPLATE
    for token, value in [
        ("VAL_ID", "v" + key), ("RNG_ID", "r" + key),
        ("CNV_ID", "c" + key), ("DL_ID", "d" + key),
        ("__MIN__", str(OPACITY_MIN)), ("__MAX__", str(OPACITY_MAX)),
        ("__W__", str(w)), ("__H__", str(h)), ("__DW__", str(disp_w)),
        ("__BASE__", b_url), ("__LOW__", l_url), ("__HIGH__", h_url),
        ("__FNAME__", fname),
    ]:
        html = html.replace(token, value)
    components.html(html, height=disp_h + 210)


files = st.file_uploader(
    "Перетащите картинки сюда",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
)

if not files:
    st.info("Пока пусто. Киньте картинку графика выше")

for i, f in enumerate(files):
    st.divider()
    st.subheader(f.name)

    try:
        with st.spinner("Обрабатываю…"):
            already, base, low, high = _prepare(f.getvalue())
    except Exception as e:
        st.error(f"Ошибка при обработке {f.name}: {e}")
        continue

    if already:
        st.error("Этот файл уже обработан. Загрузите исходное изображение, чтобы не наложить паттерны второй раз.")
        continue

    _viewer(base, low, high, "beyondcolor_" + f.name.rsplit(".", 1)[0] + ".png", str(i))

    buf = io.BytesIO()
    info = PngInfo()
    info.add_text(PROCESSING_MARKER, PROCESSING_VERSION)
    Image.fromarray(high).save(buf, format="PNG", pnginfo=info)
    st.download_button(
        label="Скачать в полном размере",
        data=buf.getvalue(),
        file_name="beyondcolor_" + f.name.rsplit(".", 1)[0] + ".png",
        mime="image/png",
        key=f.name,
    )