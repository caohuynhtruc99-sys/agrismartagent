import os
import json
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Agri-Smart AI Agent | Cao Huỳnh Trực",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# CSS TÙY CHỈNH GIAO DIỆN VIP PRO
# ==========================================
st.markdown(
    """
    <style>
    .hero-box {
        background: linear-gradient(135deg, #064e3b 0%, #0f766e 50%, #1e3a8a 100%);
        padding: 24px 28px;
        border-radius: 16px;
        border: 1px solid rgba(52, 211, 153, 0.35);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 18px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.8px;
        padding: 4px 12px;
        border-radius: 999px;
        border: 1px solid rgba(110, 231, 183, 0.4);
        margin-bottom: 10px;
    }
    .hero-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: 0.4px;
    }
    .hero-sub {
        color: #d1fae5;
        font-size: 14px;
        margin: 0 0 14px 0;
    }
    .author-pill {
        display: inline-block;
        background: rgba(15, 23, 42, 0.55);
        color: #f8fafc;
        font-size: 13px;
        line-height: 1.5;
        padding: 8px 14px;
        border-radius: 10px;
        border-left: 4px solid #34d399;
    }
    .section-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .footer-vip {
        margin-top: 32px;
        padding: 16px 20px;
        border-radius: 14px;
        background: linear-gradient(90deg, rgba(6, 78, 59, 0.45), rgba(30, 58, 138, 0.45));
        border: 1px solid rgba(52, 211, 153, 0.25);
        text-align: center;
        color: #e2e8f0;
        font-size: 13px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

default_key = ""
try:
    default_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    default_key = os.environ.get("GEMINI_API_KEY", "")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_flash_models(key: str) -> list[str]:
    """Lấy danh sách model Gemini Flash và ưu tiên bản ổn định tránh lỗi 503."""
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
    return (stable + preview) or ["gemini-2.5-flash", "gemini-2.0-flash"]

with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    available_models = fetch_flash_models(api_key) if api_key else ["gemini-2.5-flash", "gemini-2.0-flash"]
    model_choice = st.selectbox("Mô hình AI ưu tiên", available_models, index=0)
    st.caption("Đã nạp khóa từ Secrets." if default_key else "Nhập API Key để dùng Gemini.")
    st.divider()
    st.markdown("**Tác giả:** Cao Huỳnh Trực  \n**MSSV:** 058208004351  \n**Đơn vị:** Trường Đại học Giao thông vận tải Thành phố Hồ Chí Minh")

# ==========================================
# BANNER ĐẦU TRANG + THÔNG TIN TÁC GIẢ
# ==========================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-badge">AI AGENT • IOT TELEMETRY • COMPUTER VISION</div>
        <div class="hero-title">🌱 AGRI-SMART AI AGENT</div>
        <div class="hero-sub">Hệ thống Tác tử AI Giám sát Cảm biến, Chẩn đoán Bệnh qua Ảnh Lá & Điều phối Tưới Tiêu Tự động</div>
        <div class="author-pill">👨‍💻 Tác giả: <b>Cao Huỳnh Trực</b> &nbsp;|&nbsp; MSSV: <b>058208004351</b><br/>🎓 Trường Đại học Giao thông vận tải Thành phố Hồ Chí Minh</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 1. CÁC CÔNG CỤ TÍNH TOÁN (FUNCTION CALLING)
# ==========================================

def calculate_irrigation(
    crop_type: str,
    area_ha: float,
    soil_moisture: float,
    temp_c: float,
    rain_forecast_mm: float,
) -> dict:
    """Tính lưu lượng nước tưới (m3) và lệnh đóng/mở van dựa trên cảm biến đất và thời tiết."""
    targets = {"Lúa nước": 75.0, "Cà chua": 65.0, "Cà phê Robusta": 60.0, "Sầu riêng": 65.0}
    target = targets.get(crop_type, 65.0)
    deficit = max(0.0, target - soil_moisture)
    temp_factor = 1.15 if temp_c >= 32.0 else 1.0
    water_m3 = max(0.0, round((deficit * 4.5 * area_ha * temp_factor) - (rain_forecast_mm * 8.0 * area_ha), 1))
    return {
        "recommended_water_m3": water_m3,
        "valve_action": "OPEN_DRIP_VALVE" if water_m3 > 10.0 else "STANDBY",
        "duration_minutes": int(water_m3 / (1.8 * max(area_ha, 0.5))) if water_m3 > 0 else 0,
        "target_moisture_percent": target,
    }

def evaluate_disease_risk(
    disease_name: str,
    severity_percent: int,
    air_humidity: float,
    temp_c: float,
) -> dict:
    """Đánh giá rủi ro dịch hại và đề xuất phác đồ xử lý nông học."""
    if "khỏe" in disease_name.lower() or severity_percent <= 10:
        return {
            "risk_level": "THẤP",
            "risk_score": 10,
            "action_window_hours": 72,
            "protocol": "Cây sinh trưởng ổn định. Duy trì bón phân cân đối và theo dõi định kỳ.",
        }
    risk_score = min(98, int(severity_percent * (1.3 if air_humidity >= 80.0 else 1.0)))
    risk_level = "CAO" if risk_score >= 70 else "TRUNG BÌNH"
    window = 12 if risk_score >= 70 else 24
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "action_window_hours": window,
        "protocol": f"Ưu tiên xử lý {disease_name} trong {window}h tới; tỉa lá bệnh nặng, hạn chế tưới phun lên tán khi ẩm độ cao ({air_humidity}%).",
    }

def forecast_economics(
    crop_type: str,
    area_ha: float,
    risk_score: int,
    water_m3: float,
) -> dict:
    """Ước tính sản lượng bảo toàn (tấn) và giá trị kinh tế tăng thêm (triệu VNĐ)."""
    yields = {"Lúa nước": 6.8, "Cà chua": 35.0, "Cà phê Robusta": 3.2, "Sầu riêng": 14.0}
    prices = {"Lúa nước": 7.5, "Cà chua": 12.0, "Cà phê Robusta": 95.0, "Sầu riêng": 65.0}
    ratio = (risk_score / 100.0) * 0.25 + (0.04 if water_m3 > 0 else 0.0)
    saved_tons = round(yields.get(crop_type, 10.0) * area_ha * ratio, 2)
    return {
        "saved_yield_tons": saved_tons,
        "economic_gain_trieu_vnd": round(saved_tons * prices.get(crop_type, 10.0), 1),
    }

def compute_telemetry_snapshot(crop_type, area_ha, soil_moisture, temp_c, air_humidity, rain_forecast, symptom):
    severity = 5 if "khỏe" in symptom.lower() else 68
    irr = calculate_irrigation(crop_type, area_ha, soil_moisture, temp_c, rain_forecast)
    dis = evaluate_disease_risk(symptom, severity, air_humidity, temp_c)
    eco = forecast_economics(crop_type, area_ha, dis["risk_score"], irr["recommended_water_m3"])
    payload = json.dumps(
        {
            "author": "Cao Huỳnh Trực",
            "university": "Trường Đại học Giao thông vận tải Thành phố Hồ Chí Minh",
            "crop": crop_type,
            "area_ha": area_ha,
            "irrigation": irr,
            "disease": {"name": symptom, "risk": dis["risk_level"], "window_h": dis["action_window_hours"]},
            "economics": eco,
        },
        ensure_ascii=False,
        indent=2,
    )
    return irr, dis, eco, payload

def run_local_agent(crop_type, area_ha, soil_moisture, temp_c, air_humidity, rain_forecast, symptom, has_image):
    irr, dis, eco, payload = compute_telemetry_snapshot(
        crop_type, area_ha, soil_moisture, temp_c, air_humidity, rain_forecast, symptom
    )
    bt = chr(96) * 3
    img_note = "Đã nhận ảnh lá (API thị giác đang tạm bận, tính theo mức cảnh báo)." if has_image else "Phân tích theo thông số cảm biến & thực địa."
    return (
        f"### 1. Cảnh báo Chủ động & Chẩn đoán Hình ảnh\n"
        f"* **Đối tượng:** {crop_type} ({area_ha} ha) — {img_note}\n"
        f"* **Tình trạng ghi nhận:** {symptom}\n"
        f"* **Mức độ rủi ro:** **{dis['risk_level']}** ({dis['risk_score']}/100 điểm) | **Khung xử lý tối ưu:** trong **{dis['action_window_hours']} giờ** tới.\n\n"
        f"### 2. Quyết định Điều phối Tưới Tiêu & Phác đồ Xử lý\n"
        f"* **Lệnh van tưới:** `{irr['valve_action']}` — Lưu lượng đề xuất: **{irr['recommended_water_m3']} m³** (thời gian tưới dự kiến **{irr['duration_minutes']} phút** để đạt độ ẩm mục tiêu {irr['target_moisture_percent']}%).\n"
        f"* **Phác đồ nông học:** {dis['protocol']}\n\n"
        f"### 3. Dự báo Sản lượng Bảo toàn & Hiệu quả Kinh tế\n"
        f"* **Sản lượng bảo toàn ước tính:** **{eco['saved_yield_tons']} tấn**\n"
        f"* **Giá trị kinh tế tăng thêm:** **{eco['economic_gain_trieu_vnd']} triệu VNĐ**\n\n"
        f"### 4. Gói tin JSON Điều khiển Trạm IoT/PLC\n"
        f"{bt}json\n{payload}\n{bt}\n"
    )

# ==========================================
# 2. GIAO DIỆN NHẬP DỮ LIỆU CẢM BIẾN & ẢNH
# ==========================================

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-card"><b>📡 1. Dữ liệu Môi trường (IoT Sensors)</b></div>', unsafe_allow_html=True)
    crop_type = st.selectbox("Giống cây trồng", ["Cà chua", "Sầu riêng", "Cà phê Robusta", "Lúa nước"])
    area_ha = st.number_input("Diện tích canh tác (ha)", min_value=0.1, max_value=100.0, value=2.5, step=0.1)
    soil_moisture = st.slider("Độ ẩm đất cảm biến (%)", 10, 100, 44)
    temp_c = st.slider("Nhiệt độ không khí (°C)", 15, 45, 32)
    air_humidity = st.slider("Độ ẩm không khí (%)", 30, 100, 85)
    rain_forecast = st.number_input("Dự báo mưa 24h tới (mm)", min_value=0, max_value=200, value=0)

with col2:
    st.markdown('<div class="section-card"><b>🔬 2. Thị giác Máy tính (Chẩn đoán Lá)</b></div>', unsafe_allow_html=True)
    uploaded_img = st.file_uploader("Tải ảnh lá cây thật để AI quét bệnh", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        leaf_image = Image.open(uploaded_img).convert("RGB")
        st.image(leaf_image, caption="Ảnh mẫu lá đã tải lên", width=260)
        st.info("📷 Đã nhận ảnh lá. Bấm nút Kích hoạt ngay bên dưới để AI quét bệnh từ ảnh.")
        leaf_image.thumbnail((640, 640))
        symptom_note = "Nghi ngờ nhiễm bệnh qua ảnh mẫu thực địa"
    else:
        leaf_image = None
        symptom_note = st.selectbox(
            "Hoặc chọn triệu chứng quan sát thực địa:",
            ["Đốm nâu / Cháy bìa lá (nghi nấm/vi khuẩn)", "Vàng lá / Héo rũ rễ", "Phấn trắng / Mốc sương", "Lá xanh khỏe bình thường"],
        )

# ==========================================
# 3. THANH CHỈ SỐ KPI THỜI GIAN THỰC
# ==========================================
irr_now, dis_now, eco_now, json_now = compute_telemetry_snapshot(
    crop_type, area_ha, soil_moisture, temp_c, air_humidity, rain_forecast, symptom_note
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Độ ẩm đất", f"{soil_moisture}%", f"{round(soil_moisture - irr_now['target_moisture_percent'], 1)}% so với chuẩn")
m2.metric("Khí hậu vườn", f"{temp_c}°C", f"Ẩm KK {air_humidity}%")
m3.metric("Nước tưới đề xuất", f"{irr_now['recommended_water_m3']} m³", f"{irr_now['duration_minutes']} phút mở van")
m4.metric("Bảo toàn kinh tế", f"{eco_now['economic_gain_trieu_vnd']} Tr", f"+{eco_now['saved_yield_tons']} tấn")

# ==========================================
# 4. KÍCH HOẠT AI AGENT PHÂN TÍCH & ĐIỀU PHỐI
# ==========================================

if st.button("🚀 Kích hoạt AI Agent Phân tích & Điều phối", type="primary", use_container_width=True):
    with st.spinner("AI Agent đang quét ảnh lá, tính toán thủy lợi và lập kế hoạch..."):
        vision_txt = (
            "Hãy trực tiếp quan sát ảnh lá cây đính kèm để nhận diện chính xác tên bệnh (ví dụ: bệnh sương mai, đốm lá, thán thư... hoặc xác nhận lá khỏe) và ước lượng mức độ nghiêm trọng (%)."
            if leaf_image
            else f"Triệu chứng thực địa: {symptom_note} (ước lượng nghiêm trọng 70% nếu có bệnh, 5% nếu khỏe)."
        )
        prompt = (
            f"Bạn là Agri-Smart AI Agent điều hành nông trại.\n"
            f"Thông số: {crop_type}, {area_ha} ha, ẩm đất {soil_moisture}%, nhiệt độ {temp_c}°C, ẩm KK {air_humidity}%, mưa {rain_forecast} mm.\n"
            f"Thị giác: {vision_txt}\n"
            f"Hãy gọi đủ 3 công cụ calculate_irrigation, evaluate_disease_risk, forecast_economics và trình bày báo cáo 4 phần:\n"
            f"1. Cảnh báo Chủ động & Chẩn đoán Hình ảnh\n"
            f"2. Quyết định Điều phối Tưới Tiêu & Phác đồ Xử lý\n"
            f"3. Dự báo Sản lượng Bảo toàn & Hiệu quả Kinh tế\n"
            f"4. Gói tin JSON Điều khiển Trạm IoT/PLC (code block json)\n"
        )
        contents = [leaf_image, prompt] if leaf_image else prompt
        report_done = False
        last_err = ""

        if api_key:
            client = genai.Client(api_key=api_key)
            candidates = list(dict.fromkeys([model_choice] + available_models[:3]))
            for m_name in candidates:
                try:
                    res = client.models.generate_content(
                        model=m_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            tools=[calculate_irrigation, evaluate_disease_risk, forecast_economics],
                            temperature=0.2,
                        ),
                    )
                    st.session_state["agent_report"] = res.text
                    st.session_state["run_mode"] = f"Gemini AI ({m_name})"
                    st.session_state["api_note"] = ""
                    report_done = True
                    break
                except Exception as e:
                    last_err = str(e)
                    continue

        if not report_done:
            st.session_state["agent_report"] = run_local_agent(
                crop_type, area_ha, soil_moisture, temp_c, air_humidity, rain_forecast, symptom_note, bool(leaf_image)
            )
            st.session_state["run_mode"] = "Agent Cục bộ (Dự phòng)"
            st.session_state["api_note"] = last_err[:160]

        st.session_state["sensor_summary"] = (
            f"Cây {crop_type} ({area_ha}ha), ẩm đất {soil_moisture}%, nhiệt độ {temp_c}°C, ẩm KK {air_humidity}%, mưa {rain_forecast}mm."
        )

if "agent_report" in st.session_state:
    st.divider()
    st.subheader(f"🧠 Trung tâm Điều hành AI — {st.session_state.get('run_mode', '')}")
    if st.session_state.get("api_note"):
        st.caption(f"Trạng thái kết nối API: {st.session_state['api_note']}")

    tab1, tab2, tab3 = st.tabs(["📋 Báo cáo Phân tích AI", "📊 Biểu đồ Cảm biến", "🔌 Gói tin IoT / PLC"])
    with tab1:
        st.markdown(st.session_state["agent_report"])
    with tab2:
        st.write(f"**Đối chiếu thông số môi trường vườn {crop_type} ({area_ha} ha):**")
        chart_data = {
            "Giá trị đo": {
                "Độ ẩm đất hiện tại (%)": soil_moisture,
                "Độ ẩm đất chuẩn (%)": irr_now["target_moisture_percent"],
                "Độ ẩm không khí (%)": air_humidity,
                "Nhiệt độ (°C)": temp_c,
            }
        }
        st.bar_chart(chart_data)
    with tab3:
        st.code(json_now, language="json")

# ==========================================
# 5. HỎI ĐÁP NGỮ CẢNH VỚI AGENT
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
        ctx = st.session_state.get("sensor_summary", f"Cây {crop_type} ({area_ha}ha), ẩm đất {soil_moisture}%.")
        reply = None
        if api_key:
            client = genai.Client(api_key=api_key)
            for m_name in list(dict.fromkeys([model_choice] + available_models[:3])):
                try:
                    chat_res = client.models.generate_content(
                        model=m_name,
                        contents=f"Ngữ cảnh vườn: {ctx}\nBáo cáo hiện tại: {st.session_state.get('agent_report', '')}\nCâu hỏi: {user_q}",
                        config=types.GenerateContentConfig(
                            tools=[calculate_irrigation, evaluate_disease_risk, forecast_economics],
                            temperature=0.3,
                        ),
                    )
                    reply = chat_res.text
                    break
                except Exception:
                    continue
        if not reply:
            reply = f"Dựa trên thông số hiện tại ({ctx}), bạn nên ưu tiên thực hiện đúng lưu lượng tưới ({irr_now['recommended_water_m3']} m³) và khung thời gian xử lý dịch hại trong báo cáo phía trên."
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# ==========================================
# CHÂN TRANG BẢN QUYỀN TÁC GIẢ
# ==========================================
st.markdown(
    """
    <div class="footer-vip">
        <b>AGRI-SMART AI AGENT</b> — Hệ thống Điều phối Nông nghiệp Thông minh<br/>
        Phát triển bởi: <b>Cao Huỳnh Trực</b> &nbsp;•&nbsp; MSSV: <b>058208004351</b><br/>
        Trường Đại học Giao thông vận tải Thành phố Hồ Chí Minh
    </div>
    """,
    unsafe_allow_html=True,
    )
