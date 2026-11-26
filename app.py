import streamlit as st
from google import genai
from PIL import Image

# ==========================================
# 1. CẤU HÌNH TRANG & THÔNG TIN TÁC GIẢ
# ==========================================
st.set_page_config(
    page_title="Agri-Smart AI Agent",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Danh sách mô hình đã xác thực hoạt động ổn định
WORKING_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

# Cơ sở dữ liệu thông số chuẩn cho 4 loại cây trồng chủ lực
CROP_DATABASE = {
    "Lúa (Oryza sativa)": {
        "moisture_min": 70.0,
        "moisture_max": 85.0,
        "temp_min": 25.0,
        "temp_max": 32.0,
        "ph_range": "5.5 - 6.5",
        "kc": 1.15,
        "notes": "Cần duy trì mực nước nông giai đoạn đẻ nhánh và làm đòng; rút nước giữa vụ để hạn chế đổ ngã.",
    },
    "Cà chua (Solanum lycopersicum)": {
        "moisture_min": 60.0,
        "moisture_max": 75.0,
        "temp_min": 20.0,
        "temp_max": 28.0,
        "ph_range": "6.0 - 6.8",
        "kc": 1.05,
        "notes": "Tránh tưới phun lên lá vào chiều tối để ngừa bệnh sương mai (Phytophthora infestans) và đốm vòng.",
    },
    "Cà phê Robusta (Coffea canephora)": {
        "moisture_min": 55.0,
        "moisture_max": 70.0,
        "temp_min": 22.0,
        "temp_max": 29.0,
        "ph_range": "5.5 - 6.5",
        "kc": 0.95,
        "notes": "Cần giai đoạn khô hạn nhẹ (xiết nước) trước khi tưới kích hoa đồng loạt; giữ ẩm tốt giai đoạn nuôi trái.",
    },
    "Sầu riêng (Durio zibethinus)": {
        "moisture_min": 65.0,
        "moisture_max": 80.0,
        "temp_min": 24.0,
        "temp_max": 32.0,
        "ph_range": "5.5 - 6.5",
        "kc": 1.10,
        "notes": "Rễ mẫn cảm với ngập úng; thoát nước tốt vào mùa mưa để phòng nấm Phytophthora gây thối rễ, xì mủ.",
    },
}


# ==========================================
# 2. HÀM XỬ LÝ LOGIC NÔNG HỌC & GEMINI AI
# ==========================================
def get_api_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return st.session_state.get("manual_api_key", "").strip()


def call_gemini_agent(contents, api_key: str):
    """Gọi Gemini API với cơ chế tự động chuyển đổi qua các mô hình ổn định."""
    if not api_key:
        raise ValueError("Chưa cấu hình GEMINI_API_KEY trong Secrets hoặc thanh bên.")

    client = genai.Client(api_key=api_key)
    last_error = None

    for model_name in WORKING_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
            )
            if response and response.text:
                return response.text, model_name
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Tất cả mô hình dự phòng đều đang bận: {last_error}")


def calculate_irrigation(crop_name: str, soil_moisture: float, air_temp: float, air_humidity: float, area_m2: float) -> dict:
    """Tính toán nhu cầu nước tưới dựa trên độ ẩm đất, hệ số cây trồng Kc và bốc thoát hơi nước."""
    spec = CROP_DATABASE[crop_name]
    target_moisture = (spec["moisture_min"] + spec["moisture_max"]) / 2.0

    # Hệ số điều chỉnh theo nhiệt độ và ẩm độ không khí
    et_factor = 1.0
    if air_temp > spec["temp_max"]:
        et_factor += (air_temp - spec["temp_max"]) * 0.04
    if air_humidity < 60.0:
        et_factor += (60.0 - air_humidity) * 0.005

    if soil_moisture < spec["moisture_min"]:
        deficit_pct = target_moisture - soil_moisture
        # Ước tính ~0.45 lít/m2 cho mỗi 1% độ ẩm cần bù ở tầng rễ hoạt động
        liters_per_m2 = deficit_pct * 0.45 * spec["kc"] * et_factor
        total_liters = round(liters_per_m2 * area_m2, 1)
        status = "THIẾU NƯỚC - CẦN TƯỚI NGAY"
        severity = "warning"
    elif soil_moisture > spec["moisture_max"]:
        total_liters = 0.0
        status = "DƯ ẨM - NGỪNG TƯỚI & THOÁT NƯỚC"
        severity = "error"
    else:
        total_liters = 0.0
        status = "ĐỘ ẨM TỐI ƯU - DUY TRÌ"
        severity = "success"

    return {
        "status": status,
        "severity": severity,
        "target_moisture": target_moisture,
        "total_liters": total_liters,
        "liters_per_m2": round(total_liters / area_m2, 2) if area_m2 > 0 else 0.0,
        "et_factor": round(et_factor, 2),
        "spec": spec,
    }


def local_irrigation_fallback(crop_name: str, soil_moisture: float, air_temp: float, air_humidity: float, area_m2: float, calc: dict) -> str:
    """Khuyến nghị dự phòng cục bộ khi không kết nối được API."""
    spec = calc["spec"]
    lines = [
        f"### Báo cáo Chế độ Tưới tiêu ({crop_name})",
        f"- **Trạng thái hiện tại:** {calc['status']}",
        f"- **Độ ẩm đo được:** `{soil_moisture}%` (Ngưỡng chuẩn: `{spec['moisture_min']}% - {spec['moisture_max']}%`)",
        f"- **Lượng nước khuyến nghị:** `{calc['total_liters']:,.1f} lít` cho `{area_m2:,.0f} m²` (~`{calc['liters_per_m2']} lít/m²`)",
        f"- **Lưu ý kỹ thuật:** {spec['notes']}",
    ]
    if air_temp > spec["temp_max"]:
        lines.append(f"- **Cảnh báo nhiệt độ:** Nhiệt độ `{air_temp}°C` vượt ngưỡng tối ưu (`{spec['temp_max']}°C`), nên chia nhỏ lần tưới vào sáng sớm (6h-8h) và chiều mát (16h-17h30).")
    return "\n".join(lines)


# ==========================================
# 3. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==========================================
st.markdown(
    """
    <style>
    .header-box {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .header-box h2 {
        margin: 0 0 0.35rem 0;
        font-size: 1.55rem;
        color: #ffffff;
    }
    .header-box p {
        margin: 0;
        font-size: 0.9rem;
        color: #d8f3dc;
        line-height: 1.45;
    }
    .footer-box {
        margin-top: 2rem;
        padding-top: 0.85rem;
        border-top: 1px solid rgba(128, 128, 128, 0.25);
        text-align: center;
        font-size: 0.85rem;
        opacity: 0.85;
    }
    </style>
    <div class="header-box">
        <h2>🌱 Agri-Smart AI Agent</h2>
        <p>Hệ thống Trợ lý Nông nghiệp Thông minh — Tối ưu Tưới tiêu & Chẩn đoán Bệnh hại Cây trồng</p>
        <p style="margin-top: 6px; font-weight: 600;">👨‍🎓 Tác giả: Nhóm 7 | 🏫 Trường Đại học Giao thông Vận tải TP.HCM (UTH)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Thanh cấu hình bên trái
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    current_key = get_api_key()
    if current_key:
        st.success("✅ Đã kết nối GEMINI_API_KEY")
    else:
        st.warning("⚠️ Chưa tìm thấy API Key trong Secrets")

    manual_key = st.text_input(
        "Nhập / Ghi đè Gemini API Key (tùy chọn):",
        type="password",
        placeholder="AIzaSy...",
    )
    if manual_key:
        st.session_state["manual_api_key"] = manual_key

    st.divider()
    st.caption("**Chuỗi mô hình kích hoạt:**")
    for idx, m in enumerate(WORKING_MODELS, 1):
        st.caption(f"{idx}. `{m}`")

# Chọn đối tượng cây trồng chung cho toàn ứng dụng
selected_crop = st.selectbox(
    "🌿 Chọn đối tượng cây trồng canh tác:",
    options=list(CROP_DATABASE.keys()),
)
crop_info = CROP_DATABASE[selected_crop]

# 3 Tab chức năng chính
tab_irrigation, tab_vision, tab_chat = st.tabs(
    ["💧 Tưới tiêu Thông minh", "🔬 Chẩn đoán Bệnh qua Ảnh", "💬 Hỏi đáp Chuyên gia"]
)

# ------------------------------------------
# TAB 1: PHÂN TÍCH TƯỚI TIÊU & VI KHÍ HẬU
# ------------------------------------------
with tab_irrigation:
    st.subheader("Thông số Cảm biến Môi trường")

    col1, col2 = st.columns(2)
    with col1:
        soil_moisture = st.slider("Độ ẩm đất (%)", min_value=10.0, max_value=100.0, value=52.0, step=1.0)
        area_m2 = st.number_input("Diện tích canh tác (m²)", min_value=10.0, max_value=100000.0, value=500.0, step=50.0)
    with col2:
        air_temp = st.slider("Nhiệt độ không khí (°C)", min_value=10.0, max_value=45.0, value=31.0, step=0.5)
        air_humidity = st.slider("Độ ẩm không khí (%)", min_value=20.0, max_value=100.0, value=65.0, step=1.0)

    calc = calculate_irrigation(selected_crop, soil_moisture, air_temp, air_humidity, area_m2)

    m1, m2, m3 = st.columns(3)
    m1.metric("Độ ẩm đất", f"{soil_moisture}%", f"Chuẩn: {crop_info['moisture_min']}-{crop_info['moisture_max']}%")
    m2.metric("Lượng nước cần tưới", f"{calc['total_liters']:,.0f} L", f"{calc['liters_per_m2']} L/m²")
    m3.metric("Hệ số bốc thoát hơi", f"x{calc['et_factor']}", f"Kc = {crop_info['kc']}")

    if calc["severity"] == "warning":
        st.warning(f"**Trạng thái:** {calc['status']}")
    elif calc["severity"] == "error":
        st.error(f"**Trạng thái:** {calc['status']}")
    else:
        st.success(f"**Trạng thái:** {calc['status']}")

    if st.button("🚀 Phân tích & Lập lịch tưới bằng AI Agent", type="primary", use_container_width=True):
        prompt = f"""
        Bạn là kỹ sư nông học AI của hệ thống Agri-Smart AI Agent.
        Hãy phân tích dữ liệu thực địa sau và đưa ra phác đồ tưới tiêu - chăm sóc ngắn gọn, thực tế bằng tiếng Việt:
        - Cây trồng: {selected_crop}
        - Độ ẩm đất hiện tại: {soil_moisture}% (Ngưỡng chuẩn: {crop_info['moisture_min']}% - {crop_info['moisture_max']}%)
        - Nhiệt độ không khí: {air_temp}°C (Ngưỡng chuẩn: {crop_info['temp_min']}°C - {crop_info['temp_max']}°C)
        - Độ ẩm không khí: {air_humidity}%
        - Diện tích: {area_m2} m²
        - Lượng nước tính toán tự động: {calc['total_liters']} lít ({calc['liters_per_m2']} lít/m²)
        - Trạng thái: {calc['status']}

        Yêu cầu trình bày rõ ràng theo 3 mục:
        1. Đánh giá rủi ro sinh lý cây trồng hiện tại.
        2. Khuyến nghị khung giờ tưới và cách chia lượng nước tối ưu.
        3. Lưu ý phòng ngừa sâu bệnh theo điều kiện thời tiết này.
        """
        with st.spinner("Đang phân tích dữ liệu vi khí hậu..."):
            try:
                reply, used_model = call_gemini_agent(prompt, get_api_key())
                st.markdown(reply)
                st.caption(f"⚡ Phản hồi từ mô hình: `{used_model}`")
            except Exception as e:
                st.info("⚠️ Đang dùng bộ suy luận dự phòng cục bộ do API tạm gián đoạn.")
                st.markdown(local_irrigation_fallback(selected_crop, soil_moisture, air_temp, air_humidity, area_m2, calc))

# ------------------------------------------
# TAB 2: CHẨN ĐOÁN BỆNH HẠI QUA HÌNH ẢNH
# ------------------------------------------
with tab_vision:
    st.subheader("Nhận diện Bệnh hại & Sâu hại qua Ảnh lá")
    st.caption("Hệ thống tự động nhận diện loài cây thực tế trong ảnh, kể cả khi khác với cây đang chọn ở trên.")

    uploaded_file = st.file_uploader(
        "Tải lên ảnh chụp lá, thân hoặc trái bị bệnh (JPG, PNG):",
        type=["jpg", "jpeg", "png"],
    )
    symptom_note = st.text_input(
        "Mô tả thêm triệu chứng thực địa (không bắt buộc):",
        placeholder="Ví dụ: Đốm nâu lan nhanh sau mưa, mặt dưới lá có lớp phấn trắng...",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh mẫu vật tải lên", use_container_width=True)

        if st.button("🔍 Chẩn đoán Bệnh hại ngay", type="primary", use_container_width=True):
            vision_prompt = f"""
            Bạn là chuyên gia bệnh học thực vật (Plant Pathologist) của Agri-Smart AI Agent.
            Người dùng đang chọn danh mục cây trồng tham chiếu là '{selected_crop}', kèm ghi chú: '{symptom_note or "Không có"}'.

            Hãy quan sát kỹ hình ảnh được cung cấp và trả lời bằng tiếng Việt theo cấu trúc sau:
            1. **Nhận diện mẫu vật:** Xác định chính xác đây là lá/bộ phận của cây gì trong ảnh (nếu khác với '{selected_crop}', hãy nêu rõ tên cây thực tế trong ảnh).
            2. **Chẩn đoán bệnh hại / thiếu dinh dưỡng:** Tên bệnh (kèm tên khoa học của nấm/vi khuẩn/virus nếu có) và mức độ nghiêm trọng (Nhẹ / Trung bình / Nặng).
            3. **Dấu hiệu nhận biết đặc trưng:** Phân tích các vết bệnh quan sát được trên ảnh.
            4. **Phác đồ xử lý & Phòng trừ:**
               - Biện pháp canh tác / sinh học (tỉa bỏ lá bệnh, điều chỉnh độ ẩm, nấm đối kháng Trichoderma...).
               - Hoạt chất thuốc bảo vệ thực vật đặc trị khuyến nghị (nêu tên hoạt chất phổ biến tại Việt Nam và nguyên tắc 4 đúng).
            """
            with st.spinner("Đang quét tổn thương mô lá bằng Thị giác Máy tính..."):
                try:
                    diagnosis, used_model = call_gemini_agent([vision_prompt, image], get_api_key())
                    st.markdown(diagnosis)
                    st.caption(f"⚡ Chẩn đoán bởi mô hình: `{used_model}`")
                except Exception as e:
                    st.error(f"Không thể phân tích hình ảnh lúc này. Chi tiết lỗi: {e}")

# ------------------------------------------
# TAB 3: HỎI ĐÁP CHUYÊN GIA NÔNG HỌC
# ------------------------------------------
with tab_chat:
    st.subheader(f"Tư vấn Kỹ thuật Canh tác — {selected_crop}")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input(f"Đặt câu hỏi về phân bón, sâu bệnh, kỹ thuật trồng {selected_crop}...")
    if user_question:
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        chat_prompt = f"""
        Bạn là Trợ lý Nông nghiệp Thông minh Agri-Smart AI Agent (phát triển bởi sinh viên Nhóm 7- Trường ĐH Giao thông Vận tải TP.HCM).
        Bối cảnh canh tác hiện tại:
        - Cây trồng: {selected_crop}
        - Độ ẩm đất: {soil_moisture}%, Nhiệt độ: {air_temp}°C, Độ ẩm không khí: {air_humidity}%
        - Ghi chú kỹ thuật chuẩn: {crop_info['notes']}

        Câu hỏi của người dùng: {user_question}

        Hãy trả lời chính xác, thực tế, dễ áp dụng cho nông dân Việt Nam.
        """
        with st.chat_message("assistant"):
            with st.spinner("Đang tổng hợp khuyến nghị..."):
                try:
                    answer, used_model = call_gemini_agent(chat_prompt, get_api_key())
                    st.markdown(answer)
                    st.caption(f"⚡ Mô hình: `{used_model}`")
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    err_msg = f"Hệ thống AI tạm thời quá tải, vui lòng thử lại sau ít giây. ({e})"
                    st.error(err_msg)

# Chân trang
st.markdown(
    """
    <div class="footer-box">
        <strong>Agri-Smart AI Agent</strong> • Phát triển bởi <strong>Nhóm 7</strong> — Trường Đại học Giao thông Vận tải TP.HCM (UTH)
    </div>
    """,
    unsafe_allow_html=True,
        )
import streamlit as st
from google import genai

MODEL = "gemini-3.5-flash-lite"

st.set_page_config(page_title="Smart Agent", page_icon="🤖")
st.title("🤖 Smart Agent")

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            history = "\n".join(
                f"{m['role']}: {m['content']}" for m in st.session_state.messages
            )
            response = client.models.generate_content(model=MODEL, contents=history)
            answer = response.text
        except Exception as e:
            if "429" in str(e):
                answer = "⚠️ Hết hạn mức tạm thời, đợi 1 phút rồi thử lại."
            elif "503" in str(e):
                answer = "⚠️ Model đang quá tải, thử lại sau ít giây."
            else:
                answer = f"⚠️ Lỗi: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
