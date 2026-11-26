
import os
import json
from datetime import datetime
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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
# 1. MÔ HÌNH TÍNH TOÁN (xác định, kiểm chứng được). MỌI CON SỐ ĐỀU TỪ ĐÂY,
#    Gemini KHÔNG được tự bịa số. Các hằng số là GIẢ ĐỊNH MINH HỌA, không phải đo thực.
# ==========================================================================

# crop: (độ ẩm mục tiêu %, chiều sâu rễ m, năng suất tấn/ha, giá triệu VNĐ/tấn)
CROPS = {
    "Lúa nước": {"target": 75.0, "root_m": 0.20, "yield_t_ha": 6.8, "price": 7.5},
    "Cà chua": {"target": 65.0, "root_m": 0.40, "yield_t_ha": 35.0, "price": 12.0},
    "Cà phê Robusta": {"target": 60.0, "root_m": 0.60, "yield_t_ha": 3.2, "price": 95.0},
    "Sầu riêng": {"target": 65.0, "root_m": 0.80, "yield_t_ha": 14.0, "price": 65.0},
}
AWC_MM_PER_M = 120.0          # nước hữu dụng của đất thịt, mm nước / m đất
IRRIGATION_EFFICIENCY = 0.90  # hiệu suất tưới nhỏ giọt
RAIN_EFFICIENCY = 0.80        # tỉ lệ mưa thấm hữu hiệu
HEAVY_RAIN_MM = 20.0          # mưa lớn: hoãn tưới
DRIP_FLOW_M3_H_PER_HA = 10.0  # lưu lượng hệ thống tưới / ha
MIN_OPEN_M3 = 10.0            # dưới ngưỡng này không mở van
MAX_LOSS_AT_RISK_100 = 0.30   # giả định: rủi ro 100/100 mà không xử lý thì mất 30% năng suất
DISEASE_PROTECTION = 0.70     # giả định: xử lý kịp thời cứu được 70% phần thiệt hại
DROUGHT_LOSS_PER_DEFICIT = 0.50  # giả định: thiếu nước 20% -> mất 10% năng suất
IRRIGATION_PROTECTION = 0.90


def calculate_irrigation(crop_type, area_ha, soil_moisture, temp_c, rain_forecast_mm):
    """Lượng nước tưới (m3) = thiếu hụt ẩm x nước hữu dụng x chiều sâu rễ x diện tích, hiệu chỉnh theo
    nhiệt độ, trừ mưa hữu hiệu, chia hiệu suất tưới. Không tưới nếu đất đã đủ ẩm hoặc mưa lớn."""
    p = CROPS.get(crop_type, CROPS["Cà chua"])
    target = p["target"]
    deficit = max(0.0, target - soil_moisture)
    heat = 1.25 if temp_c >= 38 else (1.15 if temp_c >= 32 else 1.0)
    need_m3 = deficit / 100.0 * AWC_MM_PER_M * p["root_m"] * 10.0 * area_ha * heat
    rain_m3 = rain_forecast_mm * 10.0 * RAIN_EFFICIENCY * area_ha  # 1 mm = 10 m3/ha
    net = max(0.0, (need_m3 - rain_m3) / IRRIGATION_EFFICIENCY)
    if deficit == 0:
        reason = f"Độ ẩm đất {soil_moisture:g}% đã đạt/vượt mục tiêu {target:g}% nên không cần tưới."
        net = 0.0
    elif rain_forecast_mm >= HEAVY_RAIN_MM:
        reason = f"Dự báo mưa {rain_forecast_mm:g} mm (>= {HEAVY_RAIN_MM:g} mm) nên hoãn tưới, theo dõi lại sau mưa."
        net = 0.0
    elif net < MIN_OPEN_M3:
        reason = f"Lượng nước cần sau khi trừ mưa chỉ {net:.1f} m3 (< {MIN_OPEN_M3:g} m3) nên chưa mở van."
        net = 0.0
    else:
        reason = f"Đất thiếu {deficit:g} điểm % so với mục tiêu {target:g}%, mưa dự báo bù được {min(rain_m3, need_m3):.1f} m3."
    water = round(net, 1)
    open_valve = water >= MIN_OPEN_M3
    return {
        "recommended_water_m3": water,
        "valve_action": "OPEN_DRIP_VALVE" if open_valve else "STANDBY",
        "duration_minutes": int(round(water / (DRIP_FLOW_M3_H_PER_HA * area_ha) * 60)) if open_valve else 0,
        "target_moisture_percent": target,
        "moisture_deficit_points": round(deficit, 1),
        "reason": reason,
    }


def evaluate_disease_risk(disease_name, severity_percent, air_humidity, is_healthy=False):
    """Điểm rủi ro = mức nghiêm trọng x 1,3 nếu ẩm không khí >= 80% (nấm bệnh phát triển mạnh), tối đa 98."""
    if is_healthy or severity_percent <= 10:
        return {"risk_level": "THẤP", "risk_score": min(10, int(severity_percent)), "action_window_hours": 72,
                "protocol": "Cây sinh trưởng ổn định. Duy trì bón phân cân đối và theo dõi định kỳ."}
    mult = 1.3 if air_humidity >= 80.0 else 1.0
    score = min(98, int(round(severity_percent * mult)))
    level = "CAO" if score >= 70 else "TRUNG BÌNH"
    window = 12 if score >= 70 else 24
    return {
        "risk_level": level, "risk_score": score, "action_window_hours": window,
        "protocol": (f"Ưu tiên xử lý '{disease_name}' trong {window}h tới: tỉa lá bệnh nặng, hạn chế tưới phun lên tán "
                     f"khi ẩm độ không khí cao ({air_humidity:g}%). Chọn thuốc theo hướng dẫn của cán bộ khuyến nông/BVTV."),
    }


def forecast_economics(crop_type, area_ha, risk_score, moisture_deficit_points, target, irrigated):
    """ƯỚC TÍNH: sản lượng tránh mất = năng suất x diện tích x (thiệt hại do bệnh x 70% + thiệt hại do thiếu nước x 90%)."""
    p = CROPS.get(crop_type, CROPS["Cà chua"])
    base = p["yield_t_ha"] * area_ha
    disease_loss = risk_score / 100.0 * MAX_LOSS_AT_RISK_100
    drought_loss = min(0.30, moisture_deficit_points / target * DROUGHT_LOSS_PER_DEFICIT) if target else 0.0
    saved = base * (disease_loss * DISEASE_PROTECTION + (drought_loss * IRRIGATION_PROTECTION if irrigated else 0.0))
    return {
        "baseline_yield_tons": round(base, 2),
        "saved_yield_tons": round(saved, 2),
        "economic_gain_trieu_vnd": round(saved * p["price"], 1),
        "note": "Ước tính theo giả định minh họa, không phải số liệu đo thực tế.",
    }


def validate_inputs(soil, temp, hum, rain):
    w = []
    if soil >= 95:
        w.append(f"Độ ẩm đất {soil}% gần bão hòa: không tưới, kiểm tra thoát nước (nguy cơ úng, thối rễ).")
    if soil >= 95 and rain == 0:
        w.append("Đất bão hòa nhưng không có mưa: nên kiểm tra lại cảm biến hoặc hệ thống tưới đang rò rỉ.")
    if soil <= 15:
        w.append(f"Độ ẩm đất {soil}% rất thấp: kiểm tra cảm biến trước khi hành động.")
    if temp >= 40:
        w.append(f"Nhiệt độ {temp}°C cực đoan: ưu tiên tưới sáng sớm/chiều mát, theo dõi cây bị sốc nhiệt.")
    if rain >= 50:
        w.append(f"Dự báo mưa {rain} mm rất lớn: chuẩn bị thoát nước, hoãn phun thuốc.")
    return w


def run_agent(inp, vision):
    """Vòng lặp tác tử: Nhận thức -> Gọi công cụ -> Quyết định -> Ghi vết. Trả về mọi số liệu + nhật ký."""
    trace = []
    warns = validate_inputs(inp["soil"], inp["temp"], inp["hum"], inp["rain"])
    trace.append(("Nhận thức", f"Cảm biến: ẩm đất {inp['soil']}%, {inp['temp']}°C, ẩm KK {inp['hum']}%, mưa {inp['rain']} mm; "
                  f"cây {inp['crop']} {inp['area']} ha."))
    trace.append(("Nhận thức", f"Chẩn đoán ({vision['source']}): {vision['disease_name']}, mức nghiêm trọng {vision['severity']}%."))
    for x in warns:
        trace.append(("Kiểm tra dữ liệu", x))
    irr = calculate_irrigation(inp["crop"], inp["area"], inp["soil"], inp["temp"], inp["rain"])
    trace.append(("Công cụ calculate_irrigation", f"-> {irr['valve_action']}, {irr['recommended_water_m3']} m3. {irr['reason']}"))
    dis = evaluate_disease_risk(vision["disease_name"], vision["severity"], inp["hum"], vision["healthy"])
    trace.append(("Công cụ evaluate_disease_risk", f"-> rủi ro {dis['risk_level']} ({dis['risk_score']}/100), xử lý trong {dis['action_window_hours']}h."))
    eco = forecast_economics(inp["crop"], inp["area"], dis["risk_score"], irr["moisture_deficit_points"],
                             irr["target_moisture_percent"], irr["valve_action"] == "OPEN_DRIP_VALVE")
    trace.append(("Công cụ forecast_economics", f"-> tránh mất {eco['saved_yield_tons']} tấn ≈ {eco['economic_gain_trieu_vnd']} triệu VNĐ (ước tính)."))
    packet = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "crop": inp["crop"], "area_ha": inp["area"],
        "command": {"valve_action": irr["valve_action"], "water_m3": irr["recommended_water_m3"],
                    "duration_minutes": irr["duration_minutes"], "target_moisture_percent": irr["target_moisture_percent"]},
        "disease": {"name": vision["disease_name"], "risk": dis["risk_level"], "risk_score": dis["risk_score"],
                    "action_window_h": dis["action_window_hours"]},
        "economics_estimate": {"saved_yield_tons": eco["saved_yield_tons"], "gain_trieu_vnd": eco["economic_gain_trieu_vnd"]},
        "warnings": warns,
    }
    return {"inputs": inp, "vision": vision, "irr": irr, "dis": dis, "eco": eco, "warnings": warns,
            "trace": trace, "packet": json.dumps(packet, ensure_ascii=False, indent=2)}


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


def local_report(r):
    irr, dis, eco, v, i = r["irr"], r["dis"], r["eco"], r["vision"], r["inputs"]
    bt = chr(96) * 3
    return (
        f"### 1. Cảnh báo Chủ động & Chẩn đoán\n"
        f"* **Đối tượng:** {i['crop']} ({i['area']} ha)\n* **Tình trạng:** {v['disease_name']} (nghiêm trọng ~{v['severity']}%)"
        + (f"\n* **Dấu hiệu trên ảnh:** {v['evidence']}" if v.get("evidence") else "") + "\n"
        f"* **Rủi ro:** **{dis['risk_level']}** ({dis['risk_score']}/100), xử lý trong **{dis['action_window_hours']} giờ**.\n\n"
        f"### 2. Quyết định Tưới & Phác đồ\n"
        f"* **Van:** `{irr['valve_action']}`, nước **{irr['recommended_water_m3']} m³**, mở **{irr['duration_minutes']} phút**.\n"
        f"* **Lý do:** {irr['reason']}\n* **Phác đồ:** {dis['protocol']}\n\n"
        f"### 3. Hiệu quả kinh tế (ƯỚC TÍNH)\n"
        f"* Sản lượng tránh mất: **{eco['saved_yield_tons']} tấn** (trên nền {eco['baseline_yield_tons']} tấn)\n"
        f"* Giá trị: **{eco['economic_gain_trieu_vnd']} triệu VNĐ**. {eco['note']}\n\n"
        f"### 4. Gói tin JSON điều khiển IoT/PLC\n{bt}json\n{r['packet']}\n{bt}\n"
    )


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

m1, m2, m3, m4 = st.columns(4)
m1.metric("Độ ẩm đất", f"{soil_moisture}%", f"{round(soil_moisture - preview['irr']['target_moisture_percent'], 1)}% so với chuẩn")
m2.metric("Khí hậu vườn", f"{temp_c}°C", f"Ẩm KK {air_humidity}%")
m3.metric("Nước tưới đề xuất", f"{preview['irr']['recommended_water_m3']} m³", f"{preview['irr']['durat
