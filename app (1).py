import os
import json
import hashlib
import re
import time
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from core import *  # noqa: F401,F403

st.set_page_config(
    page_title="Agri-Smart AI Agent | Nhóm 7",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(CSS_BLOCK, unsafe_allow_html=True)

# ==========================================
# THÔNG TIN TÁC GIẢ (MSSV chỉ đọc từ Secrets, KHÔNG ghi trong code)
# Muốn hiện MSSV: thêm vào Secrets  STUDENT_ID = "..."  và  SHOW_STUDENT_ID = true
# ==========================================
AUTHOR = "Nhóm 7"
UNIVERSITY = "Trường Đại học Giao thông vận tải Thành phố Hồ Chí Minh"
try:
    _sid = str(st.secrets.get("STUDENT_ID", ""))
    _show = bool(st.secrets.get("SHOW_STUDENT_ID", False))
except Exception:
    _sid, _show = "", False
ID_TEXT = f" &nbsp;|&nbsp; MSSV: <b>{_sid}</b>" if (_show and _sid) else ""

# ==========================================
# GIỚI HẠN DÙNG CHUNG API KEY CỦA TÁC GIẢ (mỗi phiên truy cập)
# ==========================================
MAX_RUNS_PER_SESSION = 10
MAX_CHATS_PER_SESSION = 20
CHAT_HISTORY_TURNS = 8  # số tin nhắn gần nhất gửi kèm để Agent nhớ ngữ cảnh

default_key = ""
try:
    default_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    default_key = os.environ.get("GEMINI_API_KEY", "")

# Mô hình đã dò thấy dùng được (còn hạn mức); app ưu tiên các mô hình này nếu tài khoản có
PREFERRED_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
FALLBACK_MODELS = list(PREFERRED_MODELS)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_flash_models(key: str) -> list[str]:
    """Lấy danh sách model Gemini Flash, ưu tiên bản ổn định. Lỗi key/mạng -> dùng danh sách mặc định."""
    try:
        c = genai.Client(api_key=key)
        found = []
        blocked = ["embedding", "imagen", "veo", "aqa", "tts", "audio", "robotics", "computer", "image", "live", "native"]
        for m in c.models.list():
            name = getattr(m, "name", "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "gemini" in name and "flash" in name and (not actions or "generateContent" in actions):
                if not any(b in name for b in blocked):
                    found.append(name)
        def ver(n):  # số phiên bản trong tên, vd gemini-3.8-flash -> 3.8
            m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
            return float(m.group(1)) if m else 0.0

        def order(n):  # bản mới nhất trước, bản lite xếp sau
            return (-ver(n), "lite" in n, n)

        stable = sorted([x for x in found if "preview" not in x and "exp" not in x], key=order)
        preview = sorted([x for x in found if x not in stable], key=order)
        return (stable + preview) or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    available_models = fetch_flash_models(api_key) if api_key else FALLBACK_MODELS
    available_models = ([m for m in PREFERRED_MODELS if m in available_models]
                        + [m for m in available_models if m not in PREFERRED_MODELS])
    model_choice = st.selectbox("Mô hình AI ưu tiên", available_models, index=0)
    st.caption("Đã nạp khóa từ Secrets." if default_key else "Nhập API Key để dùng Gemini.")
    st.divider()
    st.markdown(f"**Nhóm thực hiện:** {AUTHOR}  \n**Đơn vị:** {UNIVERSITY}")

using_shared_key = bool(default_key) and api_key == default_key


def quota_left(counter: str, limit: int) -> bool:
    """Chỉ giới hạn khi đang dùng khóa chung của tác giả; người dùng nhập khóa riêng thì không giới hạn."""
    if not using_shared_key:
        return True
    return st.session_state.get(counter, 0) < limit


def quota_use(counter: str):
    if using_shared_key:
        st.session_state[counter] = st.session_state.get(counter, 0) + 1


# ==========================================
# BANNER ĐẦU TRANG
# ==========================================
st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-badge">AI AGENT • IOT TELEMETRY • COMPUTER VISION</div>
        <div class="hero-title">🌱 AGRI-SMART AI AGENT</div>
        <div class="hero-sub">Tác tử AI: đọc cảm biến, chẩn đoán bệnh qua ảnh lá, điều phối tưới và giải thích quyết định</div>
        <div class="author-pill">👨‍💻 Nhóm thực hiện: <b>{AUTHOR}</b>{ID_TEXT}<br/>🎓 {UNIVERSITY}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# GEMINI: CHỈ LÀM 2 VIỆC (đọc ảnh lá; viết lời giải thích). Số liệu do code cấp.
# ==========================================================================


class LeafDiagnosis(BaseModel):
    is_healthy: bool = Field(description="True nếu lá khỏe mạnh")
    disease_name: str = Field(description="Tên bệnh bằng tiếng Việt, hoặc 'Lá khỏe'")
    severity_percent: int = Field(description="Ước lượng % diện tích lá bị ảnh hưởng, 0-100")
    evidence: str = Field(description="Dấu hiệu nhìn thấy trên ảnh, 1-2 câu")


def mark_bad(model, msg):
    """Ghi nhớ mô hình lỗi để lần sau bỏ qua: 404 (ngừng cung cấp) 1 giờ, 429 (hết hạn mức) 2 phút."""
    d = st.session_state.setdefault("bad_models", {})
    if "404" in msg or "NOT_FOUND" in msg:
        d[model] = time.time() + 3600
    elif "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        d[model] = time.time() + 120


def candidate_models():
    now = time.time()
    bad = st.session_state.get("bad_models", {})
    cands = list(dict.fromkeys([model_choice] + available_models[:6]))
    ok = [m for m in cands if bad.get(m, 0) <= now]
    return ok or cands


def generate(client, **kw):
    """Gọi generate_content, tự thử lại 1 lần nếu mô hình đang quá tải (503 UNAVAILABLE)."""
    for attempt in range(2):
        try:
            return client.models.generate_content(**kw)
        except Exception as e:
            msg = str(e)
            if attempt == 0 and ("503" in msg or "UNAVAILABLE" in msg):
                time.sleep(2)
                continue
            mark_bad(kw.get("model"), msg)
            raise


def diagnose_image(client, img):
    last = ""
    for m_name in candidate_models():
        try:
            res = generate(client, 
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
        "Không nêu liều lượng/nồng độ thuốc BVTV cụ thể, chỉ nêu nhóm biện pháp và khuyên đọc nhãn thuốc, hỏi cán bộ BVTV. "
        "Không viết gói tin JSON.\n\nDỮ LIỆU:\n" + facts
    )
    last = ""
    for m_name in candidate_models():
        try:
            res = generate(client, model=m_name, contents=prompt,
                                                 config=types.GenerateContentConfig(temperature=0.2))
            if res.text:
                bt = chr(96) * 3
                return res.text + f"\n\n### 4. Gói tin JSON điều khiển IoT/PLC\n{bt}json\n{r['packet']}\n{bt}\n", m_name, ""
        except Exception as e:
            last = str(e)[:160]
    return None, "", last


# ==========================================
# GIAO DIỆN NHẬP DỮ LIỆU
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-card"><b>📡 1. Dữ liệu Môi trường (IoT Sensors - mô phỏng)</b></div>', unsafe_allow_html=True)
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
    img_sig = None  # chữ ký ảnh hiện tại, để biết kết quả chẩn đoán cũ còn khớp ảnh không
    symptom_note, manual_severity = "", 40
    if uploaded_img:
        raw = uploaded_img.getvalue()
        img_sig = hashlib.md5(raw).hexdigest()
        leaf_image = Image.open(uploaded_img).convert("RGB")
        st.image(leaf_image, caption="Ảnh lá đã tải lên", width=260)
        leaf_image.thumbnail((640, 640))
        st.info("📷 Bấm nút Kích hoạt bên dưới để Gemini đọc ảnh. Nếu không có API key, hệ thống chuyển sang nhập tay.")
        st.caption("🔒 Ảnh sẽ được gửi tới Google Gemini để phân tích. Không tải ảnh có người hoặc thông tin cá nhân.")
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


def current_vision_for_preview():
    """Chỉ dùng kết quả Gemini cũ nếu nó đúng của ảnh đang tải lên; ngược lại dùng nhập tay."""
    lv = st.session_state.get("last_vision")
    if leaf_image is not None and lv and lv.get("sig") == img_sig:
        return lv["vision"]
    return manual_vision()


inputs_now = {"crop": crop_type, "area": area_ha, "soil": soil_moisture, "temp": temp_c, "hum": air_humidity, "rain": rain_forecast}
preview = run_agent(inputs_now, current_vision_for_preview())

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
# KÍCH HOẠT AGENT
# ==========================================
if st.button("🚀 Kích hoạt AI Agent Phân tích & Điều phối", type="primary", use_container_width=True):
    with st.spinner("AI Agent đang nhận thức, gọi công cụ và lập kế hoạch..."):
        quota_note = ""
        if api_key and not quota_left("run_count", MAX_RUNS_PER_SESSION):
            client = None
            quota_note = (f"Đã hết {MAX_RUNS_PER_SESSION} lượt dùng chung khóa API trong phiên này. "
                          "Hãy nhập API Key riêng ở thanh bên để tiếp tục; hiện đang dùng Agent cục bộ.")
        else:
            client = genai.Client(api_key=api_key) if api_key else None
        err = ""
        vision = None
        if client:
            quota_use("run_count")
        if leaf_image is not None and client:
            vision, err = diagnose_image(client, leaf_image)
        if vision is None:
            vision = manual_vision()
            if leaf_image is not None:
                err = (err or quota_note or "Chưa có API key") + " | ảnh chưa được AI đọc, đang dùng triệu chứng nhập tay"
        st.session_state["last_vision"] = {"sig": img_sig, "vision": vision} if leaf_image is not None else None
        result = run_agent(inputs_now, vision)
        report, used, err2 = (None, "", "")
        if client:
            report, used, err2 = narrate(client, result)
        if report:
            mode = f"Gemini ({used}) diễn giải, số liệu do code tính"
        else:
            report, mode = local_report(result), "Agent cục bộ (dự phòng)"
        st.session_state["result"] = result
        st.session_state["result_sig"] = img_sig
        st.session_state["agent_report"] = report
        st.session_state["run_mode"] = mode
        st.session_state["api_note"] = (quota_note or err or err2)[:250]

if "result" in st.session_state:
    res = st.session_state["result"]
    st.divider()
    st.subheader(f"🧠 Trung tâm Điều hành AI — {st.session_state.get('run_mode', '')}")
    if res["inputs"] != inputs_now or st.session_state.get("result_sig") != img_sig:
        st.warning("Bạn đã đổi thông số hoặc ảnh sau lần phân tích gần nhất. Bấm Kích hoạt lại để cập nhật báo cáo.")
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
        st.caption("Gói tin mô phỏng. Hệ thống demo chưa kết nối van/PLC thật; cần người vận hành xác nhận trước khi thực thi.")
    with tab5:
        st.markdown(
            f"""
**Nước tưới (m³)** = (Thiếu hụt ẩm % ÷ 100 × {AWC_MM_PER_M:g} mm/m × chiều sâu rễ × 10 × diện tích × hệ số nhiệt − mưa × 10 × {RAIN_EFFICIENCY:g} × diện tích) ÷ {IRRIGATION_EFFICIENCY:g}.
Không mở van nếu đất đã đạt mục tiêu, mưa ≥ {HEAVY_RAIN_MM:g} mm, hoặc lượng nước < {MIN_OPEN_M3:g} m³.

**Hệ số nhiệt** = {HEAT_FACTOR_38:g} nếu nhiệt độ ≥ 38°C; {HEAT_FACTOR_32:g} nếu ≥ 32°C; còn lại 1,0.

**Rủi ro bệnh** = mức nghiêm trọng × {HUMID_RISK_MULT:g} (nếu ẩm không khí ≥ {HUMID_THRESHOLD:g}%), tối đa {MAX_RISK_SCORE}. Ảnh không đọc được thì rủi ro là *chưa xác định*, không coi là cây khỏe.

**Thiệt hại do thiếu nước** = min({DROUGHT_LOSS_CAP:g}, thiếu hụt ẩm ÷ ẩm mục tiêu × {DROUGHT_LOSS_PER_DEFICIT:g}).

**Sản lượng tránh mất** (kịch bản) = năng suất × diện tích × (rủi ro/100 × {MAX_LOSS_AT_RISK_100:g} × {DISEASE_PROTECTION:g} + thiệt hại do thiếu nước × {IRRIGATION_PROTECTION:g} nếu có tưới).

Các hằng số (năng suất, giá, nước hữu dụng, hệ số bảo vệ…) là **giả định minh họa** để demo, cần thay bằng số liệu thực của vườn/địa phương.
"""
        )
        st.table({"Cây": list(CROPS.keys()),
                  "Ẩm mục tiêu (%)": [c["target"] for c in CROPS.values()],
                  "Rễ (m)": [c["root_m"] for c in CROPS.values()],
                  "Năng suất (tấn/ha)": [c["yield_t_ha"] for c in CROPS.values()],
                  "Giá (triệu/tấn)": [c["price"] for c in CROPS.values()]})

# ==========================================
# HỎI ĐÁP NGỮ CẢNH (có nhớ hội thoại, bám số liệu đã tính)
# ==========================================
st.divider()
st.subheader("💬 Hỏi đáp Ngữ cảnh với Agri-Smart Agent")
st.caption("⚠️ Agent chỉ trả lời về nông nghiệp theo số liệu đã tính, không đưa liều lượng thuốc BVTV cụ thể. "
           "Mọi khuyến nghị chỉ mang tính tham khảo, cần cán bộ BVTV/kỹ sư nông nghiệp xác nhận.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

CHAT_RULES = (
    "Bạn là Agri-Smart Agent, trợ lý nông nghiệp. Trả lời tiếng Việt, ngắn gọn.\n"
    "QUY TẮC:\n"
    "1. Chỉ dùng các số liệu trong DỮ LIỆU bên dưới, không tự tính lại, không bịa số. Nếu cần số mới, "
    "yêu cầu người dùng đổi thông số và bấm Kích hoạt lại.\n"
    "2. Nếu valve_action là STANDBY thì không được khuyên mở van.\n"
    "3. Chỉ trả lời câu hỏi liên quan nông nghiệp, tưới tiêu, bệnh cây trồng, hệ thống này. Câu hỏi ngoài phạm vi: từ chối lịch sự.\n"
    "4. KHÔNG đưa liều lượng, nồng độ, công thức pha thuốc BVTV cụ thể. Chỉ nêu nhóm biện pháp chung, "
    "khuyên đọc kỹ nhãn thuốc và hỏi cán bộ BVTV/cửa hàng thuốc có chứng chỉ.\n"
    "5. Chẩn đoán từ ảnh chỉ là tham khảo, hãy nhắc cần chuyên gia xác nhận khi nói về bệnh.\n"
)


def build_history(messages):
    """Chuyển lịch sử chat sang định dạng Gemini; bắt đầu bằng lượt của người dùng."""
    recent = messages[-CHAT_HISTORY_TURNS:]
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]
    return [
        types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["content"])])
        for m in recent
    ]


if user_q := st.chat_input("Hỏi thêm về lịch tưới, biện pháp phòng bệnh hoặc thay đổi diện tích..."):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)
    with st.chat_message("assistant"):
        base = st.session_state.get("result", preview)
        reply = None
        if api_key and quota_left("chat_count", MAX_CHATS_PER_SESSION):
            client = genai.Client(api_key=api_key)
            ctx = json.dumps({"inputs": base["inputs"], "disease_image": base["vision"], "irrigation": base["irr"],
                              "disease_risk": base["dis"], "economics_estimate": base["eco"]}, ensure_ascii=False)
            history = build_history(st.session_state.messages)
            for m_name in candidate_models():
                try:
                    out = generate(client, 
                        model=m_name,
                        contents=history,
                        config=types.GenerateContentConfig(
                            system_instruction=CHAT_RULES + "\nDỮ LIỆU:\n" + ctx,
                            temperature=0.3),
                    )
                    if out.text:
                        reply = out.text
                        quota_use("chat_count")
                        break
                except Exception:
                    continue
        elif api_key:
            st.info(f"Đã hết {MAX_CHATS_PER_SESSION} lượt chat dùng chung khóa API trong phiên này. "
                    "Nhập API Key riêng ở thanh bên để tiếp tục.")
        if not reply:
            reply = (f"Theo số liệu đã tính: van **{base['irr']['valve_action']}**, nước **{base['irr']['recommended_water_m3']} m³**, "
                     f"rủi ro bệnh **{base['dis']['risk_level']}**. {base['irr']['reason']} (Chưa kết nối được Gemini để trả lời sâu hơn.)")
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

st.markdown(
    f"""
    <div class="footer-vip">
        <b>AGRI-SMART AI AGENT</b> — Hệ thống Điều phối Nông nghiệp Thông minh (sản phẩm học tập)<br/>
        Nhóm thực hiện: <b>{AUTHOR}</b>{ID_TEXT}<br/>{UNIVERSITY}
    </div>
    """,
    unsafe_allow_html=True,
)
