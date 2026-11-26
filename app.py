import os
import json
from datetime import datetime
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from core import *  # noqa: F401,F403

st.set_page_config(
    page_title="Agri-Smart AI Agent | Cao Huỳnh Trực",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# CSS TÙY CHỈNH GIAO DIỆN VIP PRO
# ==========================================
st.markdown(CSS_BLOCK, unsafe_allow_html=True)

# Đặt True nếu muốn hiện MSSV trên trang công khai (khuyến nghị để False).
SHOW_STUDENT_ID = False
STUDENT_ID = "058208004351"
AUTHOR = "Cao Huỳnh Trực"
UNIVERSITY = "Trường Đại học Giao thông vận tải Thành phố Hồ Chí Minh"
ID_TEXT = f" &nbsp;|&nbsp; MSSV: <b>{STUDENT_ID}</b>" if SHOW_STUDENT_ID else ""

default_key = ""
try:
    default_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    default_key = os.environ.get("GEMINI_API_KEY", "")

FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_flash_models(key: str) -> list[str]:
    """Lấy danh sách model Gemini Flash, ưu tiên bản ổn định. Lỗi key/mạng -> dùng danh sách mặc định."""
    try:
        c = genai.Client(api_key=key)
        found = []
        blocked = ["embedding", "imagen", "veo", "aqa", "tts", "audio", "robotics", "computer", "image"]
        for m in c.models.list():
            name = getattr(m, "name", "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "gemini" in name and "flash" in name and (not actions or "generateContent" in actions):
                if not any(b in name for b in blocked):
                    found.append(name)
        stable = [x for x in found if "preview" not in x and "exp" not in x]
        preview = [x for x in found if x not in stable]
        return (stable + preview) or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    available_models = fetch_flash_models(api_key) if api_key else FALLBACK_MODELS
    model_choice = st.selectbox("Mô hình AI ưu tiên", available_models, index=0)
    st.caption("Đã nạp khóa từ Secrets." if default_key else "Nhập API Key để dùng Gemini.")
    st.divider()
    st.markdown(f"**Tác giả:** {AUTHOR}  \n**Đơn vị:** {UNIVERSITY}")

# ==========================================
# BANNER ĐẦU TRANG
# ==========================================
st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-badge">AI AGENT • IOT TELEMETRY • COMPUTER VISION</div>
        <div class="hero-title">🌱 AGRI-SMART AI AGENT</div>
        <div class="hero-sub">Tác tử AI: đọc cảm biến, chẩn đoán bệnh qua ảnh lá, điều phối tưới và giải thích quyết định</div>
        <div class="author-pill">👨‍💻 Tác giả: <b>{AUTHOR}</b>{ID_TEXT}<br/>🎓 {UNIVERSITY}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# 2. GEMINI: CHỈ LÀM 2 VIỆC (đọc ảnh lá; viết lời giải thích). Số liệu do code cấp.
# ==========================================================================

class LeafDiagnosis(BaseModel):
    is_healthy: bool = Field(description="True nếu lá khỏe mạnh")
    disease_name: str = Field(description="Tên bệnh bằng tiếng Việt, hoặc 'Lá khỏe'")
    severity_percent: int = Field(description="Ước lượng % diện tích lá bị ảnh hưởng, 0-100")
    evidence: str = Field(description="Dấu hiệu nhìn thấy trên ảnh, 1-2 câu")


def candidate_models():
    return list(dict.fromkeys([model_choice] + available_models[:3]))


def diagnose_image(client, img):
    last = ""
    for m_name in candidate_models():
        try:
            res = client.models.generate_content(
                model=m_name,
                contents=[img, "Bạn là chuyên gia bảo vệ thực vật. Quan sát ảnh lá cây và chẩn đoán. Nếu ảnh không rõ hoặc không phải lá cây, "
                               "đặt is_healthy=false, disease_name='Không xác định (ảnh không rõ)', severity_percent=0."],
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   response_schema=LeafDiagnosis, temperature=0.1),
            )
            d = res.parsed
            if d is None:
                d = LeafDiagnosis(**json.loads(res.text))
            sev = max(0, min(100, int(d.severity_percent)))
            return {"healthy": bool(d.is_healthy), "disease_name": d.disease_name.strip() or "Không xác định",
                    "severity": 0 if d.is_healthy else sev, "evidence": d.evidence,
                    "source": f"Gemini thị giác ({m_name})"}, ""
        except Exception as e:
            last = str(e)[:160]
    return None, last


def narrate(client, r):
    facts = json.dumps({"inputs": r["inputs"], "disease_image": r["vision"], "irrigation": r["irr"],
                        "disease_risk": r["dis"], "economics_estimate": r["eco"], "warnings": r["warnings"]}, ensure_ascii=False)
    prompt = (
        "Bạn là Agri-Smart AI Agent. Dưới đây là KẾT QUẢ ĐÃ TÍNH SẴN bằng code (JSON). Hãy viết báo cáo tiếng Việt, ngắn gọn, "
        "3 phần: '1. Cảnh báo & Chẩn đoán', '2. Quyết định Tưới & Phác đồ', '3. Hiệu quả kinh tế (ước tính)'.\n"
        "QUY TẮC: chỉ dùng đúng các con số trong JSON, KHÔNG tự tính lại hay bịa số; nếu valve_action là STANDBY thì "
        "tuyệt đối không khuyên mở van; nêu rõ lý do (reason) và các warnings; phần kinh tế phải ghi là ước tính theo giả định. "
        "Không viết gói tin JSON.\n\nDỮ LIỆU:\n" + facts
    )
    last = ""
    for m_name in candidate_models():
        try:
            res = client.models.generate_content(model=m_name, contents=prompt,
                                                 config=types.GenerateContentConfig(temperature=0.2))
            if res.text:
                bt = chr(96) * 3
                return res.text + f"\n\n### 4. Gói tin JSON điều khiển IoT/PLC\n{bt}json\n{r['packet']}\n{bt}\n", m_name, ""
        except Exception as e:
            last = str(e)[:160]
    return None, "", last


# ==========================================
# 3. GIAO DIỆN NHẬP DỮ LIỆU
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-card"><b>📡 1. Dữ liệu Môi trường (IoT Sensors)</b></div>', unsafe_allow_html=True)
    crop_type = st.selectbox("Giống cây trồng", list(CROPS.keys()), index=1)
    area_ha = st.number_input("Diện tích canh tác (ha)", min_value=0.1, max_value=100.0, value=2.5, step=0.1)
    soil_moisture = st.slider("Độ ẩm đất cảm biến (%)", 10, 100, 44)
    temp_c = st.slider("Nhiệt độ không khí (°C)", 15, 45, 32)
    air_humidity = st.slider("Độ ẩm không khí (%)", 30, 100, 85)
    rain_forecast = st.number_input("Dự báo mưa 24h tới (mm)", min_value=0, max_value=200, value=0)

with col2:
    st.markdown('<div class="section-card"><b>🔬 2. Thị giác Máy tính (Chẩn đoán Lá)</b></div>', unsafe_allow_html=True)
    uploaded_img = st.file_uploader("Tải ảnh lá cây để AI quét bệnh", type=["jpg", "jpeg", "png"])
    leaf_image = None
    symptom_note, manual_severity = "", 40
    if uploaded_img:
        leaf_image = Image.open(uploaded_img).convert("RGB")
        st.image(leaf_image, caption="Ảnh lá đã tải lên", width=260)
        leaf_image.thumbnail((640, 640))
        st.info("📷 Bấm nút Kích hoạt bên dưới để Gemini đọc ảnh. Nếu không có API key, hệ thống chuyển sang nhập tay.")
    symptom_note = st.selectbox(
        "Triệu chứng quan sát thực địa (dùng khi không có ảnh hoặc ảnh không đọc được):",
        ["Đốm nâu / Cháy bìa lá (nghi nấm/vi khuẩn)", "Vàng lá / Héo rũ rễ", "Phấn trắng / Mốc sương", "Lá xanh khỏe bình thường"],
    )
    manual_severity = st.slider("Mức nghiêm trọng quan sát được (% diện tích lá)", 0, 100, 40,
                                disabled="khỏe" in symptom_note.lower())

for msg in validate_inputs(soil_moisture, temp_c, air_humidity, rain_forecast):
    st.warning(msg)


def manual_vision():
    healthy = "khỏe" in symptom_note.lower()
    return {"healthy": healthy, "disease_name": "Lá khỏe" if healthy else symptom_note,
            "severity": 0 if healthy else manual_severity, "evidence": "", "source": "nhập tay"}


inputs_now = {"crop": crop_type, "area": area_ha, "soil": soil_moisture, "temp": temp_c, "hum": air_humidity, "rain": rain_forecast}
preview = run_agent(inputs_now, st.session_state.get("last_vision") if leaf_image and st.session_state.get("last_vision") else manual_vision())

irr_p = preview["irr"]
eco_p = preview["eco"]
water_txt = str(irr_p["recommended_water_m3"]) + " m³"
min_txt = str(irr_p["duration_minutes"]) + " phút mở van"
gap_txt = str(round(soil_moisture - irr_p["target_moisture_percent"], 1)) + "% so với chuẩn"
gain_txt = str(eco_p["economic_gain_trieu_vnd"]) + " Tr"
tons_txt = "+" + str(eco_p["saved_yield_tons"]) + " tấn"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Độ ẩm đất", str(soil_moisture) + "%", gap_txt)
m2.metric("Khí hậu vườn", str(temp_c) + "°C", "Ẩm KK " + str(air_humidity) + "%")
m3.metric("Nước tưới đề xuất", water_txt, min_txt)
m4.metric("Bảo toàn kinh tế (ước tính)", gain_txt, tons_txt)

# ==========================================
# 4. KÍCH HOẠT AGENT
# ==========================================
if st.button("🚀 Kích hoạt AI Agent Phân tích & Điều phối", type="primary", use_container_width=True):
    with st.spinner("AI Agent đang nhận thức, gọi công cụ và lập kế hoạch..."):
        client = genai.Client(api_key=api_key) if api_key else None
        err = ""
        vision = None
        if leaf_image is not None and client:
            vision, err = diagnose_image(client, leaf_image)
        if vision is None:
            vision = manual_vision()
            if leaf_image is not None:
                err = (err or "Chưa có API key") + " | ảnh chưa được AI đọc, đang dùng triệu chứng nhập tay"
        st.session_state["last_vision"] = vision if leaf_image is not None else None
        result = run_agent(inputs_now, vision)
        report, used, err2 = (None, "", "")
        if client:
            report, used, err2 = narrate(client, result)
        if report:
            mode = f"Gemini ({used}) diễn giải, số liệu do code tính"
        else:
            report, mode = local_report(result), "Agent cục bộ (dự phòng)"
        st.session_state["result"] = result
        st.session_state["agent_report"] = report
        st.session_state["run_mode"] = mode
        st.session_state["api_note"] = (err or err2)[:200]

if "result" in st.session_state:
    res = st.session_state["result"]
    st.divider()
    st.subheader(f"🧠 Trung tâm Điều hành AI — {st.session_state.get('run_mode', '')}")
    if res["inputs"] != inputs_now:
        st.warning("Bạn đã đổi thông số sau lần phân tích gần nhất. Bấm Kích hoạt lại để cập nhật báo cáo.")
    if st.session_state.get("api_note"):
        st.caption(f"Trạng thái kết nối API: {st.session_state['api_note']}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Báo cáo", "🧭 Nhật ký suy luận", "📊 Biểu đồ", "🔌 Gói tin IoT / PLC", "📐 Giả định & công thức"])
    with tab1:
        st.markdown(st.session_state["agent_report"])
        if res["vision"]["source"].startswith("Gemini"):
            st.caption("Chẩn đoán ảnh do mô hình ngôn ngữ đa phương thức đưa ra, chỉ mang tính tham khảo; cần cán bộ BVTV xác nhận trước khi dùng thuốc.")
    with tab2:
        for step, text in res["trace"]:
            st.markdown(f"**{step}:** {text}")
    with tab3:
        st.bar_chart({"Giá trị đo": {"Độ ẩm đất hiện tại (%)": res["inputs"]["soil"],
                                      "Độ ẩm đất chuẩn (%)": res["irr"]["target_moisture_percent"],
                                      "Độ ẩm không khí (%)": res["inputs"]["hum"],
                                      "Nhiệt độ (°C)": res["inputs"]["temp"]}})
    with tab4:
        st.code(res["packet"], language="json")
    with tab5:
        st.markdown(
            f"""
**Nước tưới (m³)** = (Thiếu hụt ẩm % ÷ 100 × {AWC_MM_PER_M:g} mm/m × chiều sâu rễ × 10 × diện tích × hệ số nhiệt − mưa × 10 × {RAIN_EFFICIENCY:g} × diện tích) ÷ {IRRIGATION_EFFICIENCY:g}.
Không mở van nếu đất đã đạt mục tiêu, mưa ≥ {HEAVY_RAIN_MM:g} mm, hoặc lượng nước < {MIN_OPEN_M3:g} m³.

**Rủi ro bệnh** = mức nghiêm trọng × 1,3 (nếu ẩm không khí ≥ 80%), tối đa 98.

**Sản lượng tránh mất** = năng suất × diện tích × (rủi ro/100 × {MAX_LOSS_AT_RISK_100:g} × {DISEASE_PROTECTION:g} + thiệt hại do thiếu nước × {IRRIGATION_PROTECTION:g} nếu có tưới).

Các hằng số (năng suất, giá, nước hữu dụng, hệ số bảo vệ…) là **giả định minh họa** để demo, cần thay bằng số liệu thực của vườn/địa phương.
"""
        )
        st.table({"Cây": list(CROPS.keys()),
                  "Ẩm mục tiêu (%)": [c["target"] for c in CROPS.values()],
                  "Rễ (m)": [c["root_m"] for c in CROPS.values()],
                  "Năng suất (tấn/ha)": [c["yield_t_ha"] for c in CROPS.values()],
                  "Giá (triệu/tấn)": [c["price"] for c in CROPS.values()]})

# ==========================================
# 5. HỎI ĐÁP NGỮ CẢNH (bám số liệu đã tính)
# ==========================================
st.divider()
st.subheader("💬 Hỏi đáp Ngữ cảnh với Agri-Smart Agent")
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_q := st.chat_input("Hỏi thêm về cách pha thuốc, lịch tưới hoặc thay đổi diện tích..."):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)
    with st.chat_message("assistant"):
        base = st.session_state.get("result", preview)
        reply = None
        if api_key:
            client = genai.Client(api_key=api_key)
            ctx = json.dumps({"inputs": base["inputs"], "irrigation": base["irr"], "disease_risk": base["dis"],
                              "economics_estimate": base["eco"]}, ensure_ascii=False)
            for m_name in candidate_models():
                try:
                    out = client.models.generate_content(
                        model=m_name,
                        contents=(f"Bạn là Agri-Smart Agent. Dữ liệu đã tính bằng code (không tự tính lại, không bịa số, "
                                  f"nếu van STANDBY thì không khuyên mở van; nếu cần số mới hãy nói người dùng đổi thông số và bấm Kích hoạt lại):\n{ctx}\n\n"
                                  f"Câu hỏi: {user_q}"),
                        config=types.GenerateContentConfig(temperature=0.3),
                    )
                    reply = out.text
                    break
                except Exception:
                    continue
        if not reply:
            reply = (f"Theo số liệu đã tính: van **{base['irr']['valve_action']}**, nước **{base['irr']['recommended_water_m3']} m³**, "
                     f"rủi ro bệnh **{base['dis']['risk_level']}**. {base['irr']['reason']} (Chưa kết nối được Gemini để trả lời sâu hơn.)")
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

st.markdown(
    f"""
    <div class="footer-vip">
        <b>AGRI-SMART AI AGENT</b> — Hệ thống Điều phối Nông nghiệp Thông minh (sản phẩm học tập)<br/>
        Phát triển bởi: <b>{AUTHOR}</b>{ID_TEXT}<br/>{UNIVERSITY}
    </div>
    """,
    unsafe_allow_html=True,
)
