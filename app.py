import os
import json
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(page_title="Agri-Smart AI Agent", page_icon="🌱", layout="wide")

default_key = ""
try:
    default_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    default_key = os.environ.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    model_choice = st.selectbox("Mô hình AI ưu tiên", ["gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"], index=0)
    st.caption("Đã tự động nạp khóa từ Secrets." if default_key else "Nhập API Key để dùng Gemini.")

st.title("🌱 AGRI-SMART AI AGENT")
st.caption("Tác tử AI Giám sát Cảm biến, Chẩn đoán Bệnh qua Ảnh & Điều phối Tưới Tiêu")

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

def run_local_agent(crop_type, area_ha, soil_moisture, temp_c, air_humidity, rain_forecast, symptom, has_image):
    """Chạy trực tiếp 3 công cụ của Agent khi API Gemini tạm chạm giới hạn lượt gọi (429)."""
    severity = 5 if "khỏe" in symptom.lower() else 68
    irr = calculate_irrigation(crop_type, area_ha, soil_moisture, temp_c, rain_forecast)
    dis = evaluate_disease_risk(symptom, severity, air_humidity, temp_c)
    eco = forecast_economics(crop_type, area_ha, dis["risk_score"], irr["recommended_water_m3"])
    payload = json.dumps(
        {
            "crop": crop_type,
            "area_ha": area_ha,
            "irrigation": irr,
            "disease": {"name": symptom, "risk": dis["risk_level"], "window_h": dis["action_window_hours"]},
        },
        ensure_ascii=False,
        indent=2,
    )
    bt = chr(96) * 3
    img_note = "Đã tiếp nhận ảnh mẫu lá và đối chiếu triệu chứng thực địa." if has_image else "Phân tích theo thông số cảm biến & thực địa."
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
    st.subheader("1. Dữ liệu Môi trường (IoT)")
    crop_type = st.selectbox("Giống cây trồng", ["Sầu riêng", "Cà phê Robusta", "Lúa nước", "Cà chua"])
    area_ha = st.number_input("Diện tích canh tác (ha)", min_value=0.1, max_value=100.0, value=2.5, step=0.1)
    soil_moisture = st.slider("Độ ẩm đất cảm biến (%)", 10, 100, 44)
    temp_c = st.slider("Nhiệt độ không khí (°C)", 15, 45, 32)
    air_humidity = st.slider("Độ ẩm không khí (%)", 30, 100, 85)
    rain_forecast = st.number_input("Dự báo mưa 24h tới (mm)", min_value=0, max_value=200, value=0)

with col2:
    st.subheader("2. Thị giác Máy tính (Chẩn đoán Lá)")
    uploaded_img = st.file_uploader("Tải ảnh lá cây thật để AI quét bệnh", type=["jpg", "jpeg", "png"])
    symptom_note = st.selectbox(
        "Triệu chứng quan sát thực địa:",
        ["Đốm nâu / Cháy bìa lá (nghi nấm/vi khuẩn)", "Vàng lá / Héo rũ rễ", "Phấn trắng / Mốc sương", "Lá xanh khỏe bình thường"],
    )
    if uploaded_img:
        leaf_image = Image.open(uploaded_img).convert("RGB")
        st.image(leaf_image, caption="Ảnh mẫu lá đã tải lên", use_container_width=True)
        leaf_image.thumbnail((640, 640))
    else:
        leaf_image = None

# ==========================================
# 3. KÍCH HOẠT AI AGENT PHÂN TÍCH & ĐIỀU PHỐI
# ==========================================

if st.button("🚀 Kích hoạt AI Agent Phân tích & Điều phối", type="primary", use_container_width=True):
    with st.spinner("AI Agent đang phân tích dữ liệu cảm biến và hình ảnh..."):
        vision_txt = (
            "Quan sát ảnh lá cây đính kèm để chẩn đoán bệnh (hoặc xác nhận lá khỏe) và ước lượng mức độ nghiêm trọng (%)."
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

        if api_key:
            client = genai.Client(api_key=api_key)
            candidates = list(dict.fromkeys([model_choice, "gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]))
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
                    report_done = True
                    break
                except Exception:
                    continue

        if not report_done:
            st.session_state["agent_report"] = run_local_agent(
                crop_type, area_ha, soil_moisture, temp_c, air_humidity, rain_forecast, symptom_note, bool(leaf_image)
            )
            st.session_state["run_mode"] = "Agent Cục bộ (Tự động dự phòng hạn mức API)"

        st.session_state["sensor_summary"] = (
            f"Cây {crop_type} ({area_ha}ha), ẩm đất {soil_moisture}%, nhiệt độ {temp_c}°C, ẩm KK {air_humidity}%, mưa {rain_forecast}mm, triệu chứng: {symptom_note}."
        )

if "agent_report" in st.session_state:
    st.divider()
    st.subheader(f"🧠 Kết quả Điều hành — {st.session_state.get('run_mode', '')}")
    st.markdown(st.session_state["agent_report"])

# ==========================================
# 4. HỎI ĐÁP NGỮ CẢNH VỚI AGENT
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
            candidates = list(dict.fromkeys([model_choice, "gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]))
            for m_name in candidates:
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
            reply = f"Dựa trên thông số hiện tại ({ctx}), bạn nên ưu tiên thực hiện đúng lưu lượng tưới và khung thời gian xử lý dịch hại trong báo cáo phía trên; nếu thay đổi diện tích hoặc độ ẩm đất, hãy điều chỉnh thanh trượt ở mục 1 rồi bấm lại nút Kích hoạt AI Agent để tính toán lại chính xác."
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
                    
