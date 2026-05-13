"""
Runtime backend for Kiyora AI Prediction.

This file is designed for Flask usage. It does NOT retrain the notebook/script
pipeline on every import. It exposes the 3 functions expected by app.py:

- predict_customer(data)
- get_dashboard_summary()
- get_prediction_history()

The original clustering script used 31 engineered features for K-Means. This
runtime module mirrors those feature names and maps the web form payload into a
compact customer-segmentation result that the existing HTML dashboard can read.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

PREDICTIONS_FILE = "predictions.json"

FEATURES = [
    "is_female", "age_ord", "income_ord", "is_bangkok",
    "skin_oily", "skin_dry", "skin_combo", "skin_sensitive",
    "acne_severity", "n_skin_concerns",
    "use_cw", "self_directed", "doctor_influenced",
    "friend_influenced", "influencer_influenced", "n_brands_used",
    "score_deep_cleansing", "score_acne_friendly", "score_sensitive_friendly",
    "score_no_allergen", "score_hypoallergenic", "score_moisturized",
    "score_low_friction", "score_nourishment", "score_eye_friendly",
    "score_oil_control",
    "safety_priority", "efficacy_priority", "comfort_priority",
    "price_sensitive", "performance_driven",
]

SEGMENTS = {
    0: {
        "name": "Sensitive Loyal Potential",
        "analysis": "ลูกค้ามีแนวโน้มภักดีต่อแบรนด์ เหมาะกับการสื่อสารเรื่องความอ่อนโยน ปลอดภัย และเหมาะกับผิวแพ้ง่าย",
        "marketing_strategy": "เน้นรีวิวจากผู้ใช้ผิวแพ้ง่าย จุดขาย 0% Alcohol / Fragrance / Paraben และแคมเปญซื้อซ้ำ",
    },
    1: {
        "name": "Performance Seeker",
        "analysis": "ลูกค้าให้ความสำคัญกับประสิทธิภาพการทำความสะอาดและผลลัพธ์ เหมาะกับการสื่อสารแบบ Before/After",
        "marketing_strategy": "เน้น Deep Clean, ลดความมัน, ทำความสะอาดเมคอัพ และคอนเทนต์พิสูจน์ผลลัพธ์จริง",
    },
    2: {
        "name": "Price Sensitive Switcher",
        "analysis": "ลูกค้ามีโอกาสเปลี่ยนแบรนด์ตามราคาและโปรโมชัน จึงต้องใช้ข้อเสนอที่ชัดเจนเพื่อกระตุ้นการซื้อ",
        "marketing_strategy": "ใช้โปรโมชันทดลองใช้ Bundle Set ส่วนลดครั้งแรก และแคมเปญคุ้มค่ากว่าราคา",
    },
    3: {
        "name": "Not Loyal / New Customer",
        "analysis": "ลูกค้ายังไม่คุ้นเคยหรือยังไม่ผูกพันกับ Cleansing Water จึงมีความเสี่ยงไม่ Loyal สูง",
        "marketing_strategy": "สร้าง Awareness ด้วยรีวิวจริง Sampling Before/After และโปรโมชันแรกซื้อเพื่อให้ทดลองใช้",
    },
}

# Prototype vectors built from the same 31-feature structure as the original
# clustering script. They are used as stable runtime centroids when the training
# dataset / serialized sklearn model is not bundled with the web app.
PROTOTYPES: Dict[int, Dict[str, float]] = {
    0: {
        "is_female": 1, "age_ord": 3, "income_ord": 4, "is_bangkok": 1,
        "skin_oily": 0, "skin_dry": 0, "skin_combo": 0, "skin_sensitive": 1,
        "acne_severity": 2, "n_skin_concerns": 3, "use_cw": 1,
        "self_directed": 1, "doctor_influenced": 1, "friend_influenced": 0,
        "influencer_influenced": 0, "n_brands_used": 2,
        "score_deep_cleansing": 4, "score_acne_friendly": 5,
        "score_sensitive_friendly": 5, "score_no_allergen": 5,
        "score_hypoallergenic": 5, "score_moisturized": 4,
        "score_low_friction": 4, "score_nourishment": 4,
        "score_eye_friendly": 4, "score_oil_control": 3,
        "safety_priority": 5, "efficacy_priority": 3.75,
        "comfort_priority": 4, "price_sensitive": 0, "performance_driven": 1,
    },
    1: {
        "is_female": 1, "age_ord": 3, "income_ord": 4, "is_bangkok": 0,
        "skin_oily": 1, "skin_dry": 0, "skin_combo": 1, "skin_sensitive": 0,
        "acne_severity": 3, "n_skin_concerns": 3, "use_cw": 1,
        "self_directed": 1, "doctor_influenced": 0, "friend_influenced": 0,
        "influencer_influenced": 1, "n_brands_used": 3,
        "score_deep_cleansing": 5, "score_acne_friendly": 4,
        "score_sensitive_friendly": 3, "score_no_allergen": 3,
        "score_hypoallergenic": 3, "score_moisturized": 3,
        "score_low_friction": 3, "score_nourishment": 3,
        "score_eye_friendly": 3, "score_oil_control": 5,
        "safety_priority": 3.25, "efficacy_priority": 4,
        "comfort_priority": 3, "price_sensitive": 0, "performance_driven": 1,
    },
    2: {
        "is_female": 1, "age_ord": 2, "income_ord": 2, "is_bangkok": 0,
        "skin_oily": 0, "skin_dry": 0, "skin_combo": 1, "skin_sensitive": 0,
        "acne_severity": 1, "n_skin_concerns": 2, "use_cw": 1,
        "self_directed": 0, "doctor_influenced": 0, "friend_influenced": 1,
        "influencer_influenced": 1, "n_brands_used": 3,
        "score_deep_cleansing": 3, "score_acne_friendly": 3,
        "score_sensitive_friendly": 3, "score_no_allergen": 3,
        "score_hypoallergenic": 3, "score_moisturized": 3,
        "score_low_friction": 3, "score_nourishment": 3,
        "score_eye_friendly": 3, "score_oil_control": 3,
        "safety_priority": 3, "efficacy_priority": 3,
        "comfort_priority": 3, "price_sensitive": 1, "performance_driven": 0,
    },
    3: {
        "is_female": 1, "age_ord": 2, "income_ord": 2, "is_bangkok": 0,
        "skin_oily": 0, "skin_dry": 0, "skin_combo": 0, "skin_sensitive": 0,
        "acne_severity": 0, "n_skin_concerns": 1, "use_cw": 0,
        "self_directed": 0, "doctor_influenced": 0, "friend_influenced": 1,
        "influencer_influenced": 1, "n_brands_used": 1,
        "score_deep_cleansing": 3, "score_acne_friendly": 3,
        "score_sensitive_friendly": 3, "score_no_allergen": 3,
        "score_hypoallergenic": 3, "score_moisturized": 3,
        "score_low_friction": 3, "score_nourishment": 3,
        "score_eye_friendly": 3, "score_oil_control": 3,
        "safety_priority": 3, "efficacy_priority": 3,
        "comfort_priority": 3, "price_sensitive": 1, "performance_driven": 0,
    },
}


def _first(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return default


def _age_to_ord(age_value: Any) -> int:
    try:
        age = int(age_value)
    except (TypeError, ValueError):
        age = 25
    if age < 18:
        return 1
    if age <= 22:
        return 2
    if age <= 28:
        return 3
    if age <= 34:
        return 4
    return 5


def _income_to_ord(budget_value: Any) -> int:
    text = str(budget_value or "").replace(",", "")
    if "ต่ำกว่า" in text or "300" in text and "500" not in text:
        return 1
    if "300" in text and "500" in text:
        return 2
    if "500" in text and "1000" in text:
        return 3
    if "1000" in text and "2000" in text:
        return 4
    if "มากกว่า" in text or "2000" in text:
        return 5
    return 3


def _makeup_score(value: Any) -> int:
    text = str(value or "")
    if "บ่อย" in text:
        return 3
    if "บางครั้ง" in text:
        return 2
    return 1


def _safe_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if value in (None, ""):
        return []
    return [str(value)]


def build_features(data: Dict[str, Any]) -> Dict[str, float]:
    """Map the HTML form payload into the 31 runtime features."""
    skin_type = str(_first(data, "skin_type", "skinType", default=""))
    problems = _safe_list(_first(data, "skin_problems", "skinProblems", default=[]))
    makeup_frequency = _first(data, "makeup_frequency", "makeupFrequency", default="บางครั้ง")
    budget = _first(data, "budget", default="500–1000")
    used = str(_first(data, "used_cleansing", "usedBefore", default="ไม่เคย"))
    age = _first(data, "age", default=25)

    n_problems = 0 if "ไม่มีปัญหาผิว" in problems else len(problems)
    acne_severity = 0
    if any("สิว" in p for p in problems):
        acne_severity = min(4, 1 + sum(1 for p in problems if "สิว" in p))

    skin_oily = int("ผิวมัน" in skin_type or any("ผิวมัน" in p for p in problems))
    skin_dry = int("ผิวแห้ง" in skin_type or any("แห้ง" in p or "ลอก" in p for p in problems))
    skin_combo = int("ผิวผสม" in skin_type)
    skin_sensitive = int("ผิวแพ้ง่าย" in skin_type or any("แพ้ง่าย" in p for p in problems))

    use_cw = int("เคย" in used or "ใช้" in used)
    makeup = _makeup_score(makeup_frequency)
    income_ord = _income_to_ord(budget)

    score_deep = 3 + int(makeup >= 2) + int(makeup >= 3)
    score_oil = 3 + skin_oily
    score_moist = 3 + skin_dry
    score_acne = 3 + int(acne_severity >= 2)
    score_sensitive = 3 + skin_sensitive
    score_no_allergen = 3 + skin_sensitive
    score_hypo = 3 + skin_sensitive
    score_low_friction = 3 + skin_sensitive
    score_eye = 3 + int(makeup >= 2)
    score_nourishment = 3 + skin_dry

    safety_priority = (score_acne + score_sensitive + score_no_allergen + score_hypo) / 4
    efficacy_priority = (score_deep + score_moist + score_nourishment + score_oil) / 4
    comfort_priority = (score_low_friction + score_eye) / 2

    price_sensitive = int(income_ord <= 2)
    performance_driven = int(score_deep >= 4 or acne_severity >= 2 or skin_oily == 1)

    features = {
        "is_female": 1,
        "age_ord": _age_to_ord(age),
        "income_ord": income_ord,
        "is_bangkok": 0,
        "skin_oily": skin_oily,
        "skin_dry": skin_dry,
        "skin_combo": skin_combo,
        "skin_sensitive": skin_sensitive,
        "acne_severity": acne_severity,
        "n_skin_concerns": n_problems,
        "use_cw": use_cw,
        "self_directed": 1,
        "doctor_influenced": int(skin_sensitive or acne_severity >= 3),
        "friend_influenced": 0,
        "influencer_influenced": int(makeup >= 2),
        "n_brands_used": 2 if use_cw else 1,
        "score_deep_cleansing": min(score_deep, 5),
        "score_acne_friendly": min(score_acne, 5),
        "score_sensitive_friendly": min(score_sensitive, 5),
        "score_no_allergen": min(score_no_allergen, 5),
        "score_hypoallergenic": min(score_hypo, 5),
        "score_moisturized": min(score_moist, 5),
        "score_low_friction": min(score_low_friction, 5),
        "score_nourishment": min(score_nourishment, 5),
        "score_eye_friendly": min(score_eye, 5),
        "score_oil_control": min(score_oil, 5),
        "safety_priority": round(safety_priority, 2),
        "efficacy_priority": round(efficacy_priority, 2),
        "comfort_priority": round(comfort_priority, 2),
        "price_sensitive": price_sensitive,
        "performance_driven": performance_driven,
    }
    return {feature: float(features.get(feature, 0)) for feature in FEATURES}


def _distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    # Normalized Euclidean distance to avoid 1-5 scores dominating too much.
    total = 0.0
    for key in FEATURES:
        denom = 5.0 if key in {"age_ord", "income_ord"} or key.startswith("score_") or key.endswith("priority") else 4.0
        total += ((a.get(key, 0.0) - b.get(key, 0.0)) / denom) ** 2
    return math.sqrt(total)


def predict_cluster(features: Dict[str, float]) -> Tuple[int, Dict[str, float]]:
    distances = {cid: _distance(features, proto) for cid, proto in PROTOTYPES.items()}
    cluster_id = min(distances, key=distances.get)
    return int(cluster_id), {f"cluster_{cid}": round(dist, 4) for cid, dist in distances.items()}


def _calculate_scores(features: Dict[str, float], cluster_id: int, distances: Dict[str, float]) -> Dict[str, int | str]:
    use_cw = features["use_cw"]
    safety = features["safety_priority"]
    efficacy = features["efficacy_priority"]
    comfort = features["comfort_priority"]
    price_sensitive = features["price_sensitive"]
    n_concerns = features["n_skin_concerns"]

    if cluster_id == 3:
        risk = 76 + int(price_sensitive * 8) + int(n_concerns <= 1) * 4
    elif cluster_id == 2:
        risk = 48 + int(price_sensitive * 18) - int(use_cw * 8)
    elif cluster_id == 1:
        risk = 28 + int(price_sensitive * 8) - int(efficacy * 2)
    else:
        risk = 22 - int((safety - 3) * 7) - int(use_cw * 6)

    risk = max(5, min(95, risk))
    loyalty = max(5, min(98, 100 - risk))
    repeat = max(5, min(98, loyalty + int(use_cw * 7) + int(comfort * 2) - int(price_sensitive * 5)))
    purchase = max(5, min(98, int((loyalty * 0.45) + (repeat * 0.35) + (efficacy * 6))))

    if risk >= 70:
        risk_level = "HIGH"
    elif risk >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    loyalty_level = "HIGH" if loyalty >= 70 else "MEDIUM" if loyalty >= 40 else "LOW"

    closest = min(distances.values()) if distances else 0
    confidence = max(45, min(95, int(95 - closest * 12)))

    return {
        "not_loyal_risk": int(risk),
        "loyalty_percent": int(loyalty),
        "repeat_probability": int(repeat),
        "purchase_intent": int(purchase),
        "confidence": int(confidence),
        "risk_level": risk_level,
        "loyalty_level": loyalty_level,
    }


def _load_history() -> List[Dict[str, Any]]:
    if not os.path.exists(PREDICTIONS_FILE):
        return []
    try:
        with open(PREDICTIONS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(history: List[Dict[str, Any]]) -> None:
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)


def predict_customer(data: Dict[str, Any]) -> Dict[str, Any]:
    """Main Flask prediction entry point."""
    data = data or {}
    features = build_features(data)
    cluster_id, distances = predict_cluster(features)
    segment = SEGMENTS[cluster_id]
    scores = _calculate_scores(features, cluster_id, distances)

    result: Dict[str, Any] = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": data,
        "features": features,
        "cluster": cluster_id,
        "segment_name": segment["name"],
        **scores,
        "analysis": segment["analysis"],
        "marketing_strategy": segment["marketing_strategy"],
        "distances": distances,
    }

    history = _load_history()
    history.append(result)
    _save_history(history)
    return result


def get_prediction_history() -> List[Dict[str, Any]]:
    return _load_history()


def _avg(items: List[Dict[str, Any]], key: str) -> int:
    values = [float(item.get(key, 0)) for item in items]
    if not values:
        return 0
    return int(round(sum(values) / len(values)))


def get_dashboard_summary() -> Dict[str, Any]:
    history = _load_history()
    total = len(history)

    segment_counts = Counter(item.get("segment_name", "Unknown") for item in history)
    segments = [
        {"name": name, "count": count}
        for name, count in segment_counts.most_common()
    ]

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in history:
        date_key = str(item.get("created_at", ""))[:10] or "ไม่ทราบวันที่"
        grouped[date_key].append(item)

    monthly = [
        {
            "date": date,
            "count": len(items),
            "avg_loyalty": _avg(items, "loyalty_percent"),
            "avg_risk": _avg(items, "not_loyal_risk"),
            "avg_repeat": _avg(items, "repeat_probability"),
        }
        for date, items in sorted(grouped.items())
    ]

    latest = list(reversed(history[-10:]))

    return {
        "total_customers": total,
        "avg_loyalty": _avg(history, "loyalty_percent"),
        "avg_not_loyal_risk": _avg(history, "not_loyal_risk"),
        "avg_repeat_probability": _avg(history, "repeat_probability"),
        "avg_purchase_intent": _avg(history, "purchase_intent"),
        "segments": segments,
        "monthly": monthly,
        "latest": latest,
    }
