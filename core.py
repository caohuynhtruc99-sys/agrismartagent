"""Lõi tính toán của Agri-Smart AI Agent (không phụ thuộc Streamlit/Gemini).
MỌI CON SỐ đều tính ở đây; các hằng số là GIẢ ĐỊNH MINH HỌA."""
import json
from datetime import datetime, timezone, timedelta

VN_TZ = timezone(timedelta(hours=7))  # Streamlit Cloud chạy giờ UTC, cần ép về giờ Việt Nam

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
LONG_IRRIGATION_MIN = 480     # tưới liên tục quá 8 giờ thì khuyên chia đợt
HEAT_FACTOR_38 = 1.25         # hệ số nhiệt khi >= 38°C
HEAT_FACTOR_32 = 1.15         # hệ số nhiệt khi >= 32°C
HUMID_THRESHOLD = 80.0        # ẩm không khí từ ngưỡng này thì nấm bệnh dễ phát triển
HUMID_RISK_MULT = 1.3         # giả định: nhân điểm rủi ro khi ẩm cao
MAX_RISK_SCORE = 98
MAX_LOSS_AT_RISK_100 = 0.30   # giả định: rủi ro 100/100 mà không xử lý thì mất 30% năng suất
DISEASE_PROTECTION = 0.70     # giả định: xử lý kịp thời cứu được 70% phần thiệt hại
DROUGHT_LOSS_PER_DEFICIT = 0.50  # giả định: thiếu nước 20% -> mất 10% năng suất
DROUGHT_LOSS_CAP = 0.30       # thiệt hại do thiếu nước tối đa 30%
IRRIGATION_PROTECTION = 0.90


def calculate_irrigation(crop_type, area_ha, soil_moisture, temp_c, rain_forecast_mm):
    """Lượng nước tưới (m3) = thiếu hụt ẩm x nước hữu dụng x chiều sâu rễ x diện tích, hiệu chỉnh theo
    nhiệt độ, trừ mưa hữu hiệu, chia hiệu suất tưới. Không tưới nếu đất đã đủ ẩm hoặc mưa lớn.
    Giả định: độ ẩm đất (%) được coi là tỉ lệ so với nước hữu dụng của đất (mô hình đơn giản hóa)."""
    p = CROPS.get(crop_type, CROPS["Cà chua"])
    target = p["target"]
    deficit = max(0.0, target - soil_moisture)
    heat = HEAT_FACTOR_38 if temp_c >= 38 else (HEAT_FACTOR_32 if temp_c >= 32 else 1.0)
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
    duration = int(round(water / (DRIP_FLOW_M3_H_PER_HA * area_ha) * 60)) if open_valve else 0
    if duration > LONG_IRRIGATION_MIN:
        reason += " Thời gian tưới dài, nên chia thành nhiều đợt để tránh úng cục bộ."
    return {
        "recommended_water_m3": water,
        "valve_action": "OPEN_DRIP_VALVE" if open_valve else "STANDBY",
        "duration_minutes": duration,
        "target_moisture_percent": target,
        "moisture_deficit_points": round(deficit, 1),
        "reason": reason,
    }


def evaluate_disease_risk(disease_name, severity_percent, air_humidity, is_healthy=False):
    """Điểm rủi ro = mức nghiêm trọng x 1,3 nếu ẩm không khí >= 80% (nấm bệnh phát triển mạnh), tối đa 98.
    Ảnh không đọc được thì trả về CHƯA XÁC ĐỊNH (không được coi là cây khỏe)."""
    name = (disease_name or "").lower()
    if "không xác định" in name:
        return {"risk_level": "CHƯA XÁC ĐỊNH", "risk_score": 0, "action_window_hours": 24,
                "protocol": ("Ảnh chưa đủ rõ hoặc không phải lá cây nên chưa thể đánh giá bệnh. Hãy chụp lại ảnh lá rõ nét, đủ sáng "
                             "(hoặc chọn triệu chứng thủ công), và nhờ cán bộ BVTV kiểm tra trực tiếp nếu cây có biểu hiện bất thường.")}
    if is_healthy:
        return {"risk_level": "THẤP", "risk_score": min(10, int(severity_percent)), "action_window_hours": 72,
                "protocol": "Cây sinh trưởng ổn định. Duy trì bón phân cân đối và theo dõi định kỳ."}
    if severity_percent <= 10:
        return {"risk_level": "THẤP", "risk_score": int(severity_percent), "action_window_hours": 72,
                "protocol": (f"Có dấu hiệu nhẹ của '{disease_name}' (không quá 10% diện tích lá). Theo dõi lại sau 48-72 giờ, "
                             "vệ sinh vườn, tỉa lá bệnh và nhờ cán bộ BVTV xác nhận nếu bệnh lan rộng.")}
    mult = HUMID_RISK_MULT if air_humidity >= HUMID_THRESHOLD else 1.0
    score = min(MAX_RISK_SCORE, int(round(severity_percent * mult)))
    level = "CAO" if score >= 70 else "TRUNG BÌNH"
    window = 12 if score >= 70 else 24
    return {
        "risk_level": level, "risk_score": score, "action_window_hours": window,
        "protocol": (f"Ưu tiên xử lý '{disease_name}' trong {window}h tới: tỉa lá bệnh nặng, hạn chế tưới phun lên tán "
                     f"khi ẩm độ không khí cao ({air_humidity:g}%). Chọn thuốc theo hướng dẫn của cán bộ khuyến nông/BVTV."),
    }


def forecast_economics(crop_type, area_ha, risk_score, moisture_deficit_points, target, irrigated):
    """ƯỚC TÍNH (kịch bản): sản lượng tránh mất = năng suất x diện tích x
    (thiệt hại do bệnh x 70% + thiệt hại do thiếu nước x 90% nếu có tưới)."""
    p = CROPS.get(crop_type, CROPS["Cà chua"])
    base = p["yield_t_ha"] * area_ha
    disease_loss = risk_score / 100.0 * MAX_LOSS_AT_RISK_100
    drought_loss = min(DROUGHT_LOSS_CAP, moisture_deficit_points / target * DROUGHT_LOSS_PER_DEFICIT) if target else 0.0
    saved = base * (disease_loss * DISEASE_PROTECTION + (drought_loss * IRRIGATION_PROTECTION if irrigated else 0.0))
    return {
        "baseline_yield_tons": round(base, 2),
        "saved_yield_tons": round(saved, 2),
        "economic_gain_trieu_vnd": round(saved * p["price"], 1),
        "note": ("Kịch bản ước tính theo giả định minh họa: coi mức bệnh trên ảnh lá đại diện cho toàn bộ diện tích và so sánh "
                 "với trường hợp không can thiệp. Không phải số liệu đo thực tế."),
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
    if hum >= 90:
        w.append(f"Ẩm không khí {hum}% rất cao: nấm bệnh dễ bùng phát, hạn chế tưới phun lên tán lá.")
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
        "schema_version": "1.0",
        "mode": "SIMULATION",
        "requires_operator_confirmation": True,
        "timestamp": datetime.now(VN_TZ).isoformat(timespec="seconds"),
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


def local_report(r):
    irr, dis, eco, v, i = r["irr"], r["dis"], r["eco"], r["vision"], r["inputs"]
    bt = chr(96) * 3
    if dis["risk_level"] == "CHƯA XÁC ĐỊNH":
        risk_line = "* **Rủi ro:** **CHƯA XÁC ĐỊNH**, cần chụp lại ảnh lá rõ nét hoặc chọn triệu chứng thủ công.\n\n"
    else:
        risk_line = f"* **Rủi ro:** **{dis['risk_level']}** ({dis['risk_score']}/100), xử lý trong **{dis['action_window_hours']} giờ**.\n\n"
    return (
        f"### 1. Cảnh báo Chủ động & Chẩn đoán\n"
        f"* **Đối tượng:** {i['crop']} ({i['area']} ha)\n* **Tình trạng:** {v['disease_name']} (nghiêm trọng ~{v['severity']}%)"
        + (f"\n* **Dấu hiệu trên ảnh:** {v['evidence']}" if v.get("evidence") else "") + "\n"
        + risk_line +
        f"### 2. Quyết định Tưới & Phác đồ\n"
        f"* **Van:** `{irr['valve_action']}`, nước **{irr['recommended_water_m3']} m³**, mở **{irr['duration_minutes']} phút**.\n"
        f"* **Lý do:** {irr['reason']}\n* **Phác đồ:** {dis['protocol']}\n\n"
        f"### 3. Hiệu quả kinh tế (ƯỚC TÍNH)\n"
        f"* Sản lượng tránh mất: **{eco['saved_yield_tons']} tấn** (trên nền {eco['baseline_yield_tons']} tấn)\n"
        f"* Giá trị: **{eco['economic_gain_trieu_vnd']} triệu VNĐ**. {eco['note']}\n\n"
        f"### 4. Gói tin JSON điều khiển IoT/PLC\n{bt}json\n{r['packet']}\n{bt}\n"
    )


CSS_BLOCK = """
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
    """
