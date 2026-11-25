import os
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

@st.cache_data(ttl=3600, show_spinner=False)
def get_available_models(key: str) -> list[str]:
    """Tự động lấy danh sách các model Gemini đang hoạt động từ API Key."""
    c = genai.Client(api_key=key)
    models = []
    for m in c.models.list():
        name = getattr(m, "name", "").replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if "gemini" in name and (not actions or "generateContent" in actions):
            if not any(x in name for x in ["embedding", "imagen", "veo", "aqa", "tts", "audio", "robotics", "computer"]):
                models.append(name)
    flash_models = [x for x in models if "flash" in x and "lite" not in x]
    other_models = [x for x in models if x not in flash_models]
    return flash_models + other_models

with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    st.caption("Nếu đã cấu hình trong Streamlit Secrets, ô này sẽ tự động điền.")

st.title("🌱 AGRI-SMART AI AGENT")
st.caption("Tác tử AI Tự động Giám sát Cảm biến, Chẩn đoán Bệnh qua Ảnh & Điều phối Tưới Tiêu")

if not api_key:
    st.warning("Vui lòng nhập Gemini API Key ở thanh bên trái (chạm dấu >> góc trên bên trái) để kích hoạt AI Agent.")
    st.stop()

try:
    model_list = get_available_models(api_key)
except Exception as e:
    st.error(f"Không thể kết nối API Key. Vui lòng kiểm tra lại khóa: {e}")
    st.stop()

if not model_list:
    st.error("Không tìm thấy mô hình Gemini khả dụng cho API Key này.")
    st.stop()

with st.sidebar:
    selected_model = st.selectbox("Mô hình AI (tự động cập nhật)", model_list, index=0)

client = genai.Client(api_key=api_key)

# ==========================================
# 1. ĐỊNH NGHĨA CÁC TOOLS (FUNCTION CALLING)
# ==========================================

def calculate_irrigation(
    crop_type: str,
    area_ha: float,
    soil_moisture: float,
    temp_c: float,
    rain_forecast_mm: float,
) -> dict:
    """Tính toán lưu lượng nước tưới đề xuất (m3) và lệnh đóng/mở van dựa trên thông số cảm biến đất và thời tiết."""
    optimal_moisture = {
        "Lúa nước": 75.0,
        "Cà chua": 65.0,
        "Cà phê Robusta": 60.0,
        "Sầu riêng": 65.0,
    }
    target = optimal_moisture.get(crop_type, 65.0)
    deficit = max(0.0, target - soil_moisture)
    rain_comp = rain_forecast_mm * 8.0 * area_ha
    temp_factor = 1.15 if temp_c >= 32.0 else 1.0

    water_m3 = max(0.0, round((deficit * 4.5 * area_ha * temp_factor) - rain_comp, 1))
    valve_action = "OPEN_DRIP_VALVE" if water_m3 > 10.0 else "STANDBY"
    duration_min = int(water_m3 / (1.8 * max(area_ha, 0.5))) if water_m3 > 0 else 0

    return {
        "recommended_water_m3": water_m3,
        "valve_action": valve_action,
        "duration_minutes": duration_min,
        "target_moisture_percent": target,
    }

def evaluate_disease_risk(
    disease_name: str,
    severity_percent: int,
    air_humidity: float,
    temp_c: float,
) -> dict:
    """Đánh giá mức độ rủi ro lây lan dịch hại và đề xuất phác đồ xử lý nông học."""
    if "khỏe" in disease_name.lower() or severity_percent <= 10:
        return {
            "risk_level": "THẤP",
            "risk_score": 10,
            "action_window_hours": 72,
            "protocol": "Cây sinh trưởng ổn định. Duy trì bón phân cân đối và theo dõi định kỳ.",
        }

    humidity_factor = 1.3 if air_humidity >= 80.0 else 1.0
    risk_score = min(98, int(severity_percent * humidity_factor))
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
    """Ước tính sản lượng bảo toàn (tấn) và giá trị kinh tế tăng thêm (triệu VNĐ) nếu can thiệp kịp thời."""
    base_yield = {"Lúa nước": 6.8, "Cà chua": 35.0, "Cà phê Robusta": 3.2, "Sầu riêng": 14.0}
    price_trieu = {"Lúa nước": 7.5, "Cà chua": 12.0, "Cà phê Robusta": 95.0, "Sầu riêng": 65.0}

    total_yield = base_yield.get(crop_type, 10.0) * area_ha
    unit_price = price_trieu.get(crop_type, 10.0)
    prevented_ratio = (risk_score / 100.0) * 0.25 + (0.04 if water_m3 > 0 else 0.0)

    saved_tons = round(total_yield * prevented_ratio, 2)
    gain_vnd = round(saved_tons * unit_price, 1)
    return {"saved_yield_tons": saved_tons, "economic_gain_trieu_vnd": gain_vnd}

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
    uploaded_img = st.file_uploader("Tải ảnh lá cây thật để AI quét bệnh (không bắt buộc)", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        leaf_image = Image.open(uploaded_img)
        st.image(leaf_image, caption="Ảnh mẫu lá đã tải lên", use_container_width=True)
    else:
        leaf_image = None
        symptom_note = st.selectbox(
            "Hoặc chọn triệu chứng quan sát thực địa:",
            ["Đốm nâu / Cháy bìa lá (nghi nấm/vi khuẩn)", "Vàng lá / Héo rũ rễ", "Phấn trắng / Mốc sương", "Lá xanh khỏe bình thường"],
        )

# ==========================================
# 3. KÍCH HOẠT AI AGENT SUY LUẬN & GỌI HÀM
# ==========================================

if st.button("🚀 Kích hoạt AI Agent Phân tích & Điều phối", type="primary", use_container_width=True):
    with st.spinner(f"AI Agent ({selected_model}) đang quan sát dữ liệu, gọi công cụ tính toán và lập kế hoạch..."):
        vision_context = (
            "Hãy trực tiếp quan sát ảnh lá cây đính kèm để xác định tên bệnh và ước lượng mức độ nghiêm trọng (%)."
            if leaf_image
            else f"Triệu chứng thực địa ghi nhận: {symptom_note} (ước lượng mức độ nghiêm trọng khoảng 75% nếu có bệnh, 5% nếu khỏe)."
        )

        agent_prompt = f"""
        Bạn là Agri-Smart AI Agent điều hành nông trại thông minh.
        Thông số cảm biến hiện tại:
        - Giống cây: {crop_type} | Diện tích: {area_ha} ha
        - Độ ẩm đất: {soil_moisture}% | Nhiệt độ: {temp_c}°C | Độ ẩm không khí: {air_humidity}%
        - Dự báo mưa 24h tới: {rain_forecast} mm
        - Dữ liệu thị giác: {vision_context}

        Nhiệm vụ bắt buộc:
        1. Gọi đầy đủ 3 hàm công cụ `calculate_irrigation`, `evaluate_disease_risk`, và `forecast_economics` với tham số phù hợp nhất.
        2. Trình bày báo cáo rõ ràng gồm 4 phần:
           - **Cảnh báo chủ động & Chẩn đoán hình ảnh**
           - **Quyết định điều phối tưới tiêu & Phác đồ xử lý bệnh** (nêu rõ số liệu từ công cụ)
           - **Dự báo sản lượng bảo toàn & Hiệu quả kinh tế**
           - **Gói tin JSON điều khiển gửi trạm IoT/PLC** (định dạng code block json)
        """

        contents = [leaf_image, agent_prompt] if leaf_image else agent_prompt
        response = client.models.generate_content(
            model=selected_model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[calculate_irrigation, evaluate_disease_risk, forecast_economics],
                temperature=0.2,
            ),
        )
        st.session_state["agent_report"] = response.text
        st.session_state["sensor_summary"] = (
            f"Cây {crop_type} ({area_ha}ha), ẩm đất {soil_moisture}%, nhiệt độ {temp_c}°C, ẩm KK {air_humidity}%, mưa {rain_forecast}mm."
        )

if "agent_report" in st.session_state:
    st.divider()
    st.subheader("🧠 Kết quả Điều hành của AI Agent")
    st.markdown(st.session_state["agent_report"])

# ==========================================
# 4. KHUNG HỎI ĐÁP NGỮ CẢNH VỚI AGENT
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
        context_prefix = st.session_state.get("sensor_summary", f"Cây {crop_type}, diện tích {area_ha}ha, ẩm đất {soil_moisture}%.")
        chat_res = client.models.generate_content(
            model=selected_model,
            contents=f"Ngữ cảnh vườn hiện tại: {context_prefix}\nBáo cáo trước đó: {st.session_state.get('agent_report', 'Chưa chạy phân tích')}\nCâu hỏi của nông dân: {user_q}",
            config=types.GenerateContentConfig(
                tools=[calculate_irrigation, evaluate_disease_risk, forecast_economics],
                temperature=0.3,
            ),
        )
        st.markdown(chat_res.text)
        st.session_state.messages.append({"role": "assistant", "content": chat_res.text})
  
