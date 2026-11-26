"""Trang kiểm thử Agri-Smart Agent.
Chạy: streamlit run test_agent.py   (hoặc đặt vào thư mục pages/ của repo)
Phần 1 chỉ dùng core.py (không cần API key). Phần 2 cần Gemini API key."""
import os
import json
import time
import streamlit as st
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from core import *  # noqa: F401,F403

st.set_page_config(page_title="Kiểm thử Agri-Smart", page_icon="🧪", layout="centered")
st.title("🧪 Kiểm thử Agri-Smart Agent")
st.caption("Trang nội bộ để tự kiểm tra trước khi nộp bài. Không hiển thị API key.")

BASE = {"crop": "Cà chua", "area": 2.5, "soil": 44, "temp": 32, "hum": 85, "rain": 0}


def vis(name, sev, healthy):
    return {"healthy": healthy, "disease_name": name, "severity": sev, "evidence": "", "source": "test"}


# ==========================================
# PHẦN 1: KIỂM THỬ LOGIC (core.py)
# ==========================================
def t_default():
    r = run_agent(BASE, vis("Đốm nâu", 40, False))
    ok = r["irr"]["valve_action"] == "OPEN_DRIP_VALVE" and r["irr"]["recommended_water_m3"] == 322.0
    return ok, f"{r['irr']['valve_action']}, {r['irr']['recommended_water_m3']} m3 (mong đợi 322.0)"


def t_soil_enough():
    r = run_agent({**BASE, "soil": 70}, vis("Lá khỏe", 0, True))
    ok = r["irr"]["valve_action"] == "STANDBY" and r["irr"]["recommended_water_m3"] == 0
    return ok, f"{r['irr']['valve_action']}: {r['irr']['reason'][:60]}"


def t_heavy_rain():
    r = run_agent({**BASE, "rain": 30}, vis("Đốm nâu", 40, False))
    ok = r["irr"]["valve_action"] == "STANDBY"
    return ok, f"{r['irr']['valve_action']}: {r['irr']['reason'][:60]}"


def t_small_rain_reduces_water():
    a = run_agent(BASE, vis("Lá khỏe", 0, True))["irr"]["recommended_water_m3"]
    b = run_agent({**BASE, "rain": 5}, vis("Lá khỏe", 0, True))["irr"]["recommended_water_m3"]
    return b < a, f"không mưa {a} m3 > mưa 5 mm {b} m3"


def t_unknown_image():
    r = run_agent(BASE, vis("Không xác định (ảnh không rõ)", 0, False))
    return r["dis"]["risk_level"] == "CHƯA XÁC ĐỊNH", f"rủi ro: {r['dis']['risk_level']}"


def t_healthy_leaf():
    r = run_agent(BASE, vis("Lá khỏe", 0, True))
    return r["dis"]["risk_level"] == "THẤP", f"rủi ro: {r['dis']['risk_level']} ({r['dis']['risk_score']})"


def t_mild_disease():
    r = run_agent(BASE, vis("Đốm nâu", 5, False))
    ok = r["dis"]["risk_level"] == "THẤP" and "dấu hiệu nhẹ" in r["dis"]["protocol"]
    return ok, f"rủi ro: {r['dis']['risk_level']}"


def t_high_risk_humid():
    r = run_agent(BASE, vis("Đốm nâu", 60, False))
    ok = r["dis"]["risk_level"] == "CAO" and r["dis"]["risk_score"] == 78
    return ok, f"{r['dis']['risk_level']} {r['dis']['risk_score']}/100 (mong đợi CAO 78)"


def t_risk_cap():
    r = run_agent(BASE, vis("Đốm nâu", 90, False))
    return r["dis"]["risk_score"] == MAX_RISK_SCORE, f"điểm: {r['dis']['risk_score']} (trần {MAX_RISK_SCORE})"


def t_no_humid_multiplier():
    r = run_agent({**BASE, "hum": 60}, vis("Đốm nâu", 60, False))
    return r["dis"]["risk_score"] == 60, f"điểm: {r['dis']['risk_score']} (mong đợi 60, không nhân 1,3)"


def t_packet():
    r = run_agent(BASE, vis("Đốm nâu", 40, False))
    try:
        p = json.loads(r["packet"])
        ok = (p.get("mode") == "SIMULATION" and p.get("requires_operator_confirmation") is True
              and str(p.get("timestamp", "")).endswith("+07:00"))
        return ok, f"mode={p.get('mode')}, confirm={p.get('requires_operator_confirmation')}, time={p.get('timestamp')}"
    except Exception as e:
        return False, f"JSON lỗi: {e}"


def t_economics_zero():
    r = run_agent({**BASE, "soil": 70}, vis("Lá khỏe", 0, True))
    return r["eco"]["saved_yield_tons"] == 0, f"tránh mất {r['eco']['saved_yield_tons']} tấn"


def t_all_crops():
    bad = []
    for crop in CROPS:
        try:
            r = run_agent({**BASE, "crop": crop}, vis("Đốm nâu", 40, False))
            if r["irr"]["recommended_water_m3"] < 0:
                bad.append(crop)
        except Exception as e:
            bad.append(f"{crop}: {e}")
    return not bad, "tất cả cây chạy được" if not bad else "; ".join(bad)


def t_validate():
    w = validate_inputs(97, 30, 70, 0)
    return len(w) == 2, f"{len(w)} cảnh báo (mong đợi 2: bão hòa + không mưa)"


def t_local_report():
    r = run_agent(BASE, vis("Không xác định (ảnh không rõ)", 0, False))
    txt = local_report(r)
    return "CHƯA XÁC ĐỊNH" in txt and "```json" in txt, "báo cáo cục bộ có cảnh báo ảnh không rõ và gói JSON"


TESTS = [
    ("Ca mặc định: mở van 322 m³", t_default),
    ("Đất đủ ẩm: không tưới", t_soil_enough),
    ("Mưa lớn 30 mm: hoãn tưới", t_heavy_rain),
    ("Mưa nhỏ làm giảm lượng nước", t_small_rain_reduces_water),
    ("Ảnh không rõ: rủi ro CHƯA XÁC ĐỊNH", t_unknown_image),
    ("Lá khỏe: rủi ro THẤP", t_healthy_leaf),
    ("Bệnh nhẹ 5%: THẤP, nhắc theo dõi", t_mild_disease),
    ("Bệnh 60% + ẩm cao: CAO 78", t_high_risk_humid),
    ("Trần điểm rủi ro 98", t_risk_cap),
    ("Ẩm thấp: không nhân hệ số 1,3", t_no_humid_multiplier),
    ("Gói tin JSON: SIMULATION, giờ VN", t_packet),
    ("Kinh tế = 0 khi không có gì để cứu", t_economics_zero),
    ("Mọi loại cây đều chạy được", t_all_crops),
    ("validate_inputs cảnh báo đúng", t_validate),
    ("local_report xử lý ảnh không rõ", t_local_report),
]

st.header("1. Kiểm thử logic (core.py)")
if st.button("▶️ Chạy kiểm thử logic", type="primary", use_container_width=True):
    rows, passed = [], 0
    for name, fn in TESTS:
        try:
            ok, detail = fn()
        except Exception as e:  # lỗi bất ngờ cũng tính là không đạt
            ok, detail = False, f"Lỗi: {e}"
        passed += int(ok)
        rows.append({"Ca kiểm thử": name, "Kết quả": "✅ ĐẠT" if ok else "❌ KHÔNG ĐẠT", "Chi tiết": detail})
    (st.success if passed == len(TESTS) else st.error)(f"Đạt {passed}/{len(TESTS)} ca kiểm thử.")
    st.dataframe(rows, use_container_width=True, hide_index=True)

# ==========================================
# PHẦN 2: KIỂM THỬ GEMINI
# ==========================================
st.header("2. Kiểm thử kết nối Gemini")

secret_key = ""
try:
    secret_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    secret_key = os.environ.get("GEMINI_API_KEY", "")

st.write("Khóa trong Secrets:", "✅ có" if secret_key else "❌ chưa có")
typed_key = st.text_input("Hoặc nhập khóa tạm để thử (không được lưu)", type="password")
api_key = typed_key or secret_key
model_name = st.text_input("Mô hình dùng để thử", value="gemini-3.8-flash")


class LeafDiagnosis(BaseModel):
    is_healthy: bool = Field(description="True nếu lá khỏe mạnh")
    disease_name: str = Field(description="Tên bệnh bằng tiếng Việt, hoặc 'Lá khỏe'")
    severity_percent: int = Field(description="Ước lượng % diện tích lá bị ảnh hưởng, 0-100")
    evidence: str = Field(description="Dấu hiệu nhìn thấy trên ảnh, 1-2 câu")


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
            raise


def make_test_leaf():
    """Ảnh giả lập: lá xanh có vài đốm nâu (chỉ để thử đường truyền, không đánh giá độ chính xác)."""
    img = Image.new("RGB", (400, 400), (235, 235, 235))
    d = ImageDraw.Draw(img)
    d.ellipse((60, 40, 340, 360), fill=(60, 140, 60))
    for x, y in [(150, 150), (220, 200), (180, 260), (240, 120)]:
        d.ellipse((x, y, x + 30, y + 30), fill=(120, 70, 30))
    return img


if st.button("▶️ Chạy kiểm thử Gemini", use_container_width=True):
    if not api_key:
        st.error("Chưa có API key. Hãy thêm vào Secrets hoặc nhập ở ô trên.")
    else:
        from google import genai
        from google.genai import types
        results = []

        try:
            client = genai.Client(api_key=api_key)
            t0 = time.time()
            names = [getattr(m, "name", "").replace("models/", "") for m in client.models.list()]
            flash = [n for n in names if "flash" in n and "gemini" in n]
            results.append(("Liệt kê mô hình", True, f"{len(names)} mô hình, {len(flash)} bản flash ({time.time() - t0:.1f}s)"))
            if model_name not in names:
                results.append(("Mô hình có tồn tại", False, f"'{model_name}' không có trong danh sách. Thử: {', '.join(flash[:4])}"))
        except Exception as e:
            client = None
            results.append(("Liệt kê mô hình / xác thực khóa", False, str(e)[:200]))

        if client:
            try:
                t0 = time.time()
                out = generate(client, model=model_name, contents="Trả lời đúng một từ: OK")
                results.append(("Gọi văn bản", bool(out.text), f"'{(out.text or '').strip()[:40]}' ({time.time() - t0:.1f}s)"))
            except Exception as e:
                results.append(("Gọi văn bản", False, str(e)[:200]))

            try:
                t0 = time.time()
                res = generate(client, 
                    model=model_name,
                    contents=[make_test_leaf(), "Quan sát ảnh lá cây và chẩn đoán bệnh."],
                    config=types.GenerateContentConfig(response_mime_type="application/json",
                                                       response_schema=LeafDiagnosis, temperature=0.1),
                )
                d = res.parsed or LeafDiagnosis(**json.loads(res.text))
                results.append(("Đọc ảnh + trả JSON theo schema", True,
                                f"{d.disease_name}, {d.severity_percent}% ({time.time() - t0:.1f}s)"))
            except Exception as e:
                results.append(("Đọc ảnh + trả JSON theo schema", False, str(e)[:200]))

        all_ok = all(r[1] for r in results)
        (st.success if all_ok else st.error)("Gemini hoạt động tốt." if all_ok else "Có mục chưa đạt, xem chi tiết bên dưới.")
        st.table([{"Mục": a, "Kết quả": "✅ ĐẠT" if b else "❌ KHÔNG ĐẠT", "Chi tiết": c} for a, b, c in results])
        st.caption("Lưu ý: ảnh thử là ảnh vẽ giả, chỉ kiểm tra đường truyền và định dạng JSON, không đánh giá độ chính xác chẩn đoán.")

st.subheader("Tìm mô hình Gemini dùng được")
st.caption("Thử gọi lần lượt các mô hình flash trong tài khoản của bạn, rồi cho biết mô hình nào trả lời được.")
if st.button("🔍 Dò mô hình dùng được", use_container_width=True):
    if not api_key:
        st.error("Chưa có API key.")
    else:
        from google import genai
        try:
            client = genai.Client(api_key=api_key)
            names = [getattr(m, "name", "").replace("models/", "") for m in client.models.list()]
            skip = ["image", "tts", "audio", "embedding", "live", "robotics", "computer", "imagen", "veo", "aqa", "native"]
            flash = [n for n in names if "gemini" in n and "flash" in n and not any(k in n for k in skip)]
            cands = list(dict.fromkeys(["gemini-flash-latest"] + flash))[:10]
            rows = []
            prog = st.progress(0.0)
            for i, n in enumerate(cands):
                try:
                    out = generate(client, model=n, contents="Trả lời đúng một từ: OK")
                    ok = bool(out.text)
                    rows.append({"Mô hình": n, "Kết quả": "✅ DÙNG ĐƯỢC" if ok else "❌ rỗng", "Ghi chú": (out.text or "").strip()[:30]})
                except Exception as e:
                    msg = str(e)
                    rows.append({"Mô hình": n, "Kết quả": "❌ LỖI", "Ghi chú": msg[:120]})
                prog.progress((i + 1) / len(cands))
            st.table(rows)
            good = [r["Mô hình"] for r in rows if r["Kết quả"].startswith("✅")]
            if good:
                st.success("Mô hình dùng được: " + ", ".join(good))
            else:
                st.error("Không mô hình nào trả lời được. Xem cột Ghi chú (có thể do hạn mức hoặc quyền của khóa).")
        except Exception as e:
            st.error(f"Không liệt kê được mô hình: {str(e)[:200]}")

st.divider()
st.markdown(
    "**Kiểm thử thủ công trên app chính:** (1) Kích hoạt khi chưa có ảnh; (2) tải ảnh lá thật; "
    "(3) tải ảnh không phải lá, rủi ro phải là *CHƯA XÁC ĐỊNH*; (4) hỏi chat hai câu nối tiếp để kiểm tra trí nhớ; "
    "(5) đổi ảnh khác mà chưa Kích hoạt, app phải hiện cảnh báo."
)
