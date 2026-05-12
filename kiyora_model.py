import json
import os
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


HISTORY_FILE = "predictions.json"

FEATURES = [
    "age",
    "skin_oily",
    "skin_dry",
    "skin_combo",
    "skin_sensitive",
    "makeup_score",
    "problem_count",
    "budget_score",
    "used_cleansing",
    "safety_priority",
    "efficacy_priority",
    "comfort_priority"
]

SEGMENT_NAMES = {
    0: "Sensitive Loyal Potential",
    1: "Price Sensitive Switcher",
    2: "Performance Driven Buyer",
    3: "Not Loyal / New Customer"
}

SEGMENT_STRATEGY = {
    0: "เน้นความอ่อนโยน ปลอดภัย ไม่มีแอลกอฮอล์ น้ำหอม และพาราเบน พร้อมรีวิวจากผู้ใช้ผิวแพ้ง่าย",
    1: "ใช้โปรโมชัน ส่วนลด แพ็กคู่ หรือสินค้าทดลอง เพื่อดึงให้กลับมาซื้อซ้ำ",
    2: "เน้นประสิทธิภาพการทำความสะอาด ลดสิว ลดความมัน และผลลัพธ์หลังใช้จริง",
    3: "กลุ่มที่ยังไม่เคยใช้หรือยังไม่ผูกพันกับแบรนด์ ควรสร้างการรับรู้ ทดลองใช้ และรีวิว Before/After"
}


def build_training_data():
    rng = np.random.default_rng(42)
    data = []

    for _ in range(60):
        data.append([
            rng.integers(18, 31), rng.integers(0, 2), rng.integers(0, 2),
            rng.integers(0, 2), 1, rng.integers(1, 3),
            rng.integers(2, 5), rng.integers(2, 4), 1,
            rng.uniform(4.2, 5.0), rng.uniform(3.3, 4.4), rng.uniform(3.8, 5.0)
        ])

    for _ in range(55):
        data.append([
            rng.integers(18, 28), rng.integers(0, 2), rng.integers(0, 2),
            rng.integers(0, 2), rng.integers(0, 2), rng.integers(0, 2),
            rng.integers(1, 4), rng.integers(1, 3), rng.integers(0, 2),
            rng.uniform(2.5, 4.0), rng.uniform(2.8, 4.0), rng.uniform(2.5, 4.0)
        ])

    for _ in range(60):
        data.append([
            rng.integers(20, 36), 1, 0, rng.integers(0, 2),
            rng.integers(0, 2), rng.integers(2, 4),
            rng.integers(2, 6), rng.integers(3, 5), 1,
            rng.uniform(3.3, 4.5), rng.uniform(4.3, 5.0), rng.uniform(3.0, 4.3)
        ])

    for _ in range(45):
        data.append([
            rng.integers(16, 25), rng.integers(0, 2), rng.integers(0, 2),
            rng.integers(0, 2), rng.integers(0, 2), rng.integers(0, 2),
            rng.integers(0, 3), rng.integers(1, 4), 0,
            rng.uniform(2.5, 4.0), rng.uniform(2.5, 4.0), rng.uniform(2.5, 4.0)
        ])

    return np.array(data, dtype=float)


X_TRAIN = build_training_data()
SCALER = StandardScaler()
X_SCALED = SCALER.fit_transform(X_TRAIN)

KMEANS = KMeans(n_clusters=4, random_state=42, n_init=20)
KMEANS.fit(X_SCALED)


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def encode_skin_type(value):
    text = str(value)
    return {
        "skin_oily": 1 if "มัน" in text else 0,
        "skin_dry": 1 if "แห้ง" in text else 0,
        "skin_combo": 1 if "ผสม" in text else 0,
        "skin_sensitive": 1 if "แพ้" in text or "แพ้ง่าย" in text else 0,
    }


def encode_makeup(value):
    text = str(value)
    if "บ่อย" in text:
        return 3
    if "บางครั้ง" in text:
        return 2
    return 1


def encode_budget(value):
    text = str(value)
    if "ต่ำกว่า" in text:
        return 1
    if "300" in text:
        return 2
    if "500" in text:
        return 3
    if "1000" in text or "1,000" in text:
        return 4
    if "มากกว่า" in text or "2,000" in text:
        return 5
    return 3


def encode_used(value):
    text = str(value).strip()
    if text == "ไม่เคย":
        return 0
    return 1


def calculate_priorities(skin_type, problems, makeup, budget):
    problems_text = " ".join(problems)

    safety = 3.0
    efficacy = 3.0
    comfort = 3.0

    if "แพ้ง่าย" in skin_type or "ผิวแพ้ง่าย" in problems_text:
        safety += 1.4
        comfort += 0.8

    if "สิว" in problems_text:
        safety += 0.8
        efficacy += 0.8

    if "รูขุมขน" in problems_text or "มัน" in problems_text:
        efficacy += 0.9

    if "แห้ง" in problems_text or "ลอก" in problems_text:
        comfort += 1.0

    if makeup == 3:
        efficacy += 1.0

    if budget <= 2:
        safety -= 0.2
        efficacy -= 0.2

    return (
        min(max(safety, 1), 5),
        min(max(efficacy, 1), 5),
        min(max(comfort, 1), 5)
    )


def make_feature_vector(data):
    skin_type = data.get("skin_type", "")
    age = safe_int(data.get("age"), 21)
    makeup = encode_makeup(data.get("makeup_frequency", "บางครั้ง"))
    problems = data.get("skin_problems", [])
    budget = encode_budget(data.get("budget", "300–500"))
    used = encode_used(data.get("used_cleansing", "เคย"))

    skin = encode_skin_type(skin_type)
    safety, efficacy, comfort = calculate_priorities(skin_type, problems, makeup, budget)

    row = {
        "age": age,
        "skin_oily": skin["skin_oily"],
        "skin_dry": skin["skin_dry"],
        "skin_combo": skin["skin_combo"],
        "skin_sensitive": skin["skin_sensitive"],
        "makeup_score": makeup,
        "problem_count": len(problems),
        "budget_score": budget,
        "used_cleansing": used,
        "safety_priority": safety,
        "efficacy_priority": efficacy,
        "comfort_priority": comfort,
    }

    return np.array([[row[f] for f in FEATURES]], dtype=float), row


def score_customer(feature_row, cluster_id):
    age = feature_row["age"]
    budget = feature_row["budget_score"]
    used = feature_row["used_cleansing"]
    safety = feature_row["safety_priority"]
    efficacy = feature_row["efficacy_priority"]
    comfort = feature_row["comfort_priority"]
    problem_count = feature_row["problem_count"]
    makeup = feature_row["makeup_score"]

    loyalty = 40

    if used == 1:
        loyalty += 15
    else:
        loyalty -= 45

    loyalty += budget * 4
    loyalty += safety * 5
    loyalty += comfort * 3
    loyalty += efficacy * 2

    if problem_count >= 3:
        loyalty += 4

    if makeup >= 3:
        loyalty += 5

    if age < 18:
        loyalty -= 8

    if cluster_id == 1:
        loyalty -= 12

    loyalty = int(min(max(loyalty, 5), 95))

    not_loyal_risk = 100 - loyalty

    if used == 0:
        not_loyal_risk = max(not_loyal_risk, 80)
        loyalty = min(loyalty, 20)

    not_loyal_risk = int(min(max(not_loyal_risk, 0), 100))

    repeat_probability = int(min(max(loyalty + 5, 5), 97))
    purchase_intent = int(min(max((efficacy * 12) + (safety * 8) + (used * 15), 10), 96))

    if used == 0:
        repeat_probability = min(repeat_probability, 25)
        purchase_intent = min(purchase_intent, 45)

    return loyalty, not_loyal_risk, repeat_probability, purchase_intent


def get_level(percent):
    if percent >= 75:
        return "HIGH"
    if percent >= 45:
        return "MEDIUM"
    return "LOW"


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(item):
    history = load_history()
    history.append(item)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def predict_customer(data):
    X, feature_row = make_feature_vector(data)
    X_scaled = SCALER.transform(X)

    used = feature_row["used_cleansing"]

    cluster_id = int(KMEANS.predict(X_scaled)[0])

    if used == 0:
        cluster_id = 3

    distances = np.linalg.norm(X_scaled - KMEANS.cluster_centers_, axis=1)

    confidence = 1 / (1 + float(np.min(distances)))
    confidence_percent = int(min(max(confidence * 100, 45), 95))

    loyalty, not_loyal_risk, repeat_probability, purchase_intent = score_customer(feature_row, cluster_id)

    segment_name = SEGMENT_NAMES.get(cluster_id, "Customer Segment")
    strategy = SEGMENT_STRATEGY.get(cluster_id, "ใช้กลยุทธ์การตลาดเฉพาะกลุ่ม")

    if used == 0:
        risk_level = "HIGH"
        analysis = "ลูกค้าคนนี้ยังไม่เคยใช้ Cleansing Water จึงถูกจัดอยู่ในกลุ่ม Not Loyal / New Customer มีความเสี่ยงสูงที่จะยังไม่ผูกพันกับแบรนด์ ควรใช้แคมเปญทดลองใช้ รีวิวจริง และโปรโมชันแรกซื้อ"
    elif not_loyal_risk >= 60:
        risk_level = "HIGH"
        analysis = "ลูกค้ามีความเสี่ยงที่จะไม่ภักดีต่อแบรนด์ ควรใช้โปรโมชันและรีวิวจากผู้ใช้จริงเพื่อกระตุ้นการตัดสินใจ"
    elif not_loyal_risk >= 35:
        risk_level = "MEDIUM"
        analysis = "ลูกค้ามีความเสี่ยงระดับกลาง ควรสื่อสารจุดเด่นของสินค้าให้ตรงกับปัญหาผิว"
    else:
        risk_level = "LOW"
        analysis = "ลูกค้ามีแนวโน้มภักดีต่อแบรนด์และมีโอกาสกลับมาซื้อซ้ำค่อนข้างสูง"

    result = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": data,
        "features": feature_row,
        "cluster": cluster_id,
        "segment_name": segment_name,
        "not_loyal_risk": not_loyal_risk,
        "loyalty_percent": loyalty,
        "repeat_probability": repeat_probability,
        "purchase_intent": purchase_intent,
        "confidence": confidence_percent,
        "risk_level": risk_level,
        "loyalty_level": get_level(loyalty),
        "analysis": analysis,
        "marketing_strategy": strategy,
        "distances": {
            f"cluster_{i}": round(float(d), 4)
            for i, d in enumerate(distances)
        }
    }

    save_history(result)
    return result


def get_prediction_history():
    return load_history()


def get_dashboard_summary():
    history = load_history()

    if not history:
        return {
            "total_customers": 0,
            "avg_loyalty": 0,
            "avg_not_loyal_risk": 0,
            "avg_repeat_probability": 0,
            "avg_purchase_intent": 0,
            "segments": [],
            "monthly": [],
            "latest": []
        }

    total = len(history)

    avg_loyalty = round(sum(x["loyalty_percent"] for x in history) / total, 2)
    avg_risk = round(sum(x["not_loyal_risk"] for x in history) / total, 2)
    avg_repeat = round(sum(x["repeat_probability"] for x in history) / total, 2)
    avg_purchase = round(sum(x["purchase_intent"] for x in history) / total, 2)

    segment_count = {}
    for item in history:
        name = item["segment_name"]
        segment_count[name] = segment_count.get(name, 0) + 1

    segments = [
        {
            "name": name,
            "count": count,
            "percent": round((count / total) * 100, 2)
        }
        for name, count in segment_count.items()
    ]

    monthly_map = {}
    for item in history:
        date_key = item["created_at"][:10]

        if date_key not in monthly_map:
            monthly_map[date_key] = {
                "date": date_key,
                "count": 0,
                "loyalty_sum": 0,
                "risk_sum": 0,
                "repeat_sum": 0
            }

        monthly_map[date_key]["count"] += 1
        monthly_map[date_key]["loyalty_sum"] += item["loyalty_percent"]
        monthly_map[date_key]["risk_sum"] += item["not_loyal_risk"]
        monthly_map[date_key]["repeat_sum"] += item["repeat_probability"]

    monthly = []
    for row in monthly_map.values():
        count = row["count"]
        monthly.append({
            "date": row["date"],
            "count": count,
            "avg_loyalty": round(row["loyalty_sum"] / count, 2),
            "avg_risk": round(row["risk_sum"] / count, 2),
            "avg_repeat": round(row["repeat_sum"] / count, 2),
        })

    monthly = sorted(monthly, key=lambda x: x["date"])

    return {
        "total_customers": total,
        "avg_loyalty": avg_loyalty,
        "avg_not_loyal_risk": avg_risk,
        "avg_repeat_probability": avg_repeat,
        "avg_purchase_intent": avg_purchase,
        "segments": segments,
        "monthly": monthly,
        "latest": history[-10:][::-1]
    }