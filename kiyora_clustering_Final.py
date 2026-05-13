"""
Kiyora Cleansing - Unsupervised Customer Segmentation
Dataset: แบบสอบถามผู้บริโภคสินค้าคลีนซิ่ง 129 คน

Objective: แบ่งกลุ่มผู้บริโภค (Customer Segmentation) เพื่อหา
Core Value Proposition ที่เหมาะสมสำหรับแต่ละกลุ่ม

Model: K-Means Clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

try:
    thai_fonts = [f for f in fm.findSystemFonts() if any(
        k in f.lower() for k in ["noto", "tahoma", "sarabun", "thsarabun"])]
    if thai_fonts:
        plt.rcParams["font.family"] = fm.FontProperties(fname=thai_fonts[0]).get_name()
except Exception:
    pass

plt.rcParams["figure.dpi"] = 120

# 1. LOAD DATA


print("STEP 1: LOADING DATA")
raw = pd.read_excel("Dataset_Cleansing.xlsx", sheet_name="1", header=None)
headers = raw.iloc[1].tolist()
df = raw.iloc[2:].copy()
df.columns = headers
df = df.reset_index(drop=True)

# Short column names สำหรับใช้ภายใน code
col_map = {
    headers[1]:  "gender",
    headers[2]:  "age",
    headers[3]:  "occupation",
    headers[4]:  "income",
    headers[5]:  "province",
    headers[6]:  "skin_type",
    headers[7]:  "skin_concerns",
    headers[8]:  "acne_severity",
    headers[9]:  "influence_source",
    headers[10]: "skincare_select_criteria",
    headers[11]: "cleansing_select_criteria",
    headers[12]: "use_cleansing_water",
    headers[13]: "cleansing_types_used",
    headers[14]: "cleansing_type_main",
    headers[15]: "cleansing_water_formula",
    headers[16]: "score_deep_cleansing",
    headers[17]: "score_acne_friendly",
    headers[18]: "score_sensitive_friendly",
    headers[19]: "score_no_allergen",
    headers[20]: "score_hypoallergenic",
    headers[21]: "score_moisturized",
    headers[22]: "score_low_friction",
    headers[23]: "score_nourishment",
    headers[24]: "score_eye_friendly",
    headers[25]: "score_oil_control",
    headers[26]: "switch_factors",
    headers[27]: "brands_used",
    headers[28]: "brand_most_used",
}
df = df.rename(columns=col_map)
df.info()

print(f" Loaded {len(df)} rows, {len(df.columns)} columns")
cols_to_check = [
    "age",
    "income",
    "province",
    "skin_type",
    "acne_severity",
    "influence_source",
    "switch_factors"
]

for col in cols_to_check:
    print("\n" + "="*60)
    print(f"COLUMN: {col}")
    print("="*60)

    vals = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    vc = vals.value_counts()

    print(f"\nUnique values: {len(vc)}\n")

    for k, v in vc.items():
        print(f"{k:<80} | count = {v}")



# 2. FEATURE ENGINEERING
print("STEP 2: FEATURE ENGINEERING")

feat = pd.DataFrame(index=df.index)

# 2A. Demographics (Ordinal Encoding)

# Gender
feat["is_female"] = (df["gender"].str.strip() == "หญิง").astype(int)

# Age → ordinal
age_order = age_order = {
    "ต่ำกว่า 18 ปี": 1,
    "18-22 ปี": 2,
    "23-28 ปี": 3,
    "29-34 ปี": 4,
    "35 ปี ขึ้นไป": 5
}
feat["age_ord"] = df["age"].map(age_order).fillna(3)

# Income → ordinal
income_order = income_order = {
    "ต่ำกว่า 10,000 บาท": 1,
    "10,001 - 14,999 บาท": 2,
    "15,000 - 19,999 บาท": 3,
    "20,000 - 24,999 บาท": 4,
    "25,000 - 29,999 บาท": 5,
    "30,000 - 34,999 บาท": 6,
    "35,000 - 39,999 บาท": 7,
    "40,000 บาท ขึ้นไป": 8
}
feat["income_ord"] = df["income"].map(income_order).fillna(3)

# Bangkok vs upcountry
feat["is_bangkok"] = df["province"].str.contains(
    "กทม|กรุงเทพ|bangkok",
    na=False,
    case=False
).astype(int)

# 2B. Skin Profile

# Skin type → one-hot key types
skin_map = [
    ("ผิวมัน", "skin_oily"),
    ("ผิวแห้ง", "skin_dry"),
    ("ผิวผสม", "skin_combo"),
    ("ผิวแพ้ง่าย", "skin_sensitive"),
    ("ผิวธรรมดา", "skin_normal"),
    ("ผิวขาดน้ำ", "skin_dehydrated"),
    ("ไม่แน่ใจ|ไม่ทราบ", "skin_unknown"),
]

for pattern, col in skin_map:
    feat[col] = df["skin_type"].str.contains(
        pattern,
        na=False,
        regex=True
    ).astype(int)

# Acne severity → ordinal
acne_order = acne_order = {
    "ไม่มีสิวเลย": 0,
    "นานๆทีเป็นสิว": 1,
    "สิวเล็กน้อย": 2,
    "สิวปานกลาง": 3,
    "สิวรุนแรง": 4
}
def map_acne(val):
    if pd.isna(val):
        return 0
    for k, v in acne_order.items():
        if k in str(val):
            return v
    return 0
feat["acne_severity"] = df["acne_severity"].apply(map_acne)

# Number of skin concerns (multi-select → count)
feat["n_skin_concerns"] = df["skin_concerns"].apply(
    lambda x: len(str(x).split(",")) if pd.notna(x) else 0
)

# 2C. Behavior

# Uses cleansing water?
feat["use_cw"] = (df["use_cleansing_water"].str.strip() == "ใช้").astype(int)

# Main cleansing type → label encode
le = LabelEncoder()
feat["cleansing_typ" \
"e_main"] = le.fit_transform(
    df["cleansing_type_main"].fillna("ไม่ระบุ").str.strip()
)

# Self-directed vs influenced decision
feat["self_directed"] = df["influence_source"].str.contains(
    "เลือกด้วยตนเอง", na=False).astype(int)
feat["doctor_influenced"] = df["influence_source"].str.contains(
    "ปรึกษาหมอ|dermatologist", na=False).astype(int)
feat["friend_influenced"] = df["influence_source"].str.contains(
    "เพื่อน|Friend", na=False).astype(int)
feat["influencer_influenced"] = df["influence_source"].str.contains(
    "บิวตี้บลอกเกอร์|รีวิว|influencer|review|ยูทูบ",
    na=False,
    case=False
).astype(int)

# Number of brands used
feat["n_brands_used"] = df["brands_used"].apply(
    lambda x: len(str(x).split(",")) if pd.notna(x) else 0
)

# Currently uses Kiyora?
feat["uses_kiyora"] = df["brands_used"].str.contains("Kiyora", na=False).astype(int)

# 2D. Purchase Decision Scores (Cols 16-25) 
score_cols = [
    "score_deep_cleansing", "score_acne_friendly", "score_sensitive_friendly",
    "score_no_allergen", "score_hypoallergenic", "score_moisturized",
    "score_low_friction", "score_nourishment", "score_eye_friendly",
    "score_oil_control"
]
for c in score_cols:
    feat[c] = pd.to_numeric(df[c], errors="coerce").fillna(3.0)

# Composite indices (business-meaningful aggregates)
feat["safety_priority"]    = feat[["score_acne_friendly", "score_sensitive_friendly",
                                    "score_no_allergen", "score_hypoallergenic"]].mean(axis=1)
feat["efficacy_priority"]  = feat[["score_deep_cleansing", "score_moisturized",
                                    "score_nourishment", "score_oil_control"]].mean(axis=1)
feat["comfort_priority"]   = feat[["score_low_friction", "score_eye_friendly"]].mean(axis=1)

# Switch trigger: price sensitivity
feat["price_sensitive"] = df["switch_factors"].str.contains(
    "ราคา|Cheaper|โปรโมชั่น|Promotion", na=False).astype(int)
feat["performance_driven"] = df["switch_factors"].str.contains(
    "ประสิทธิภาพ|ผล|effectiveness|result", na=False, case=False).astype(int)

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
    "price_sensitive", "performance_driven"
]

X = feat[FEATURES].copy()
print(f" Features created: {len(FEATURES)} features for {len(X)} respondents")

# 3. PREPROCESSING
print("STEP 3: PREPROCESSING")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f" Standardized {X_scaled.shape[1]} features (mean=0, std=1)")

# 4. OPTIMAL K (Elbow + Silhouette)
print("STEP 4: FINDING OPTIMAL K")

inertias, silhouettes = [], []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Finding Optimal Number of Clusters", fontsize=13, fontweight="bold")

axes[0].plot(K_range, inertias, "bo-")
axes[0].set_title("Elbow Method (Inertia)")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouettes, "rs-")
axes[1].set_title("Silhouette Score (Higher = Better)")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True, alpha=0.3)

best_k = K_range[np.argmax(silhouettes)]
axes[1].axvline(best_k, color="green", linestyle="--", label=f"Best k={best_k}")
axes[1].legend()

plt.tight_layout()
plt.savefig("elbow_silhouette.png", bbox_inches="tight")
plt.close()

for k, s in zip(K_range, silhouettes):
    marker = " ← BEST" if k == best_k else ""
    print(f"  k={k}  silhouette={s:.4f}{marker}")

print(f"\n Best k = {best_k} (silhouette = {max(silhouettes):.4f})")

# 5. FINAL K-MEANS MODEL
print("STEP 5: FITTING FINAL K-MEANS MODEL")

K = best_k
kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
df["cluster"] = kmeans.fit_predict(X_scaled)
feat["cluster"] = df["cluster"]

print(f" Fitted K-Means with k={K}")
print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index().to_string())

# 6. PCA VISUALIZATION
print("STEP 6: PCA VISUALIZATION")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

var_exp = pca.explained_variance_ratio_
print(f"✓ PCA: PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}%  "
      f"(total={sum(var_exp)*100:.1f}%)")

colors = plt.cm.Set1(np.linspace(0, 0.8, K))
fig, ax = plt.subplots(figsize=(9, 6))

for c in range(K):
    mask = df["cluster"] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[colors[c]], label=f"Cluster {c} (n={mask.sum()})",
               s=60, alpha=0.75, edgecolors="white", linewidths=0.4)

# Plot centroids in PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           c="black", marker="X", s=200, zorder=5, label="Centroids")

ax.set_title("K-Means Clusters (PCA Projection)", fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("cluster_pca.png", bbox_inches="tight")
plt.close()

# 7. CLUSTER PROFILING
print("STEP 7: CLUSTER PROFILING")

profile_cols = [
    "age_ord", "income_ord", "is_bangkok", "is_female",
    "skin_oily", "skin_dry", "skin_sensitive", "acne_severity",
    "use_cw", "self_directed", "doctor_influenced", "price_sensitive",
    "n_brands_used", "uses_kiyora",
    "safety_priority", "efficacy_priority", "comfort_priority",
    "score_deep_cleansing", "score_acne_friendly", "score_sensitive_friendly",
    "score_moisturized", "score_oil_control"
]

cluster_profile = feat.groupby("cluster")[profile_cols].mean()
print("\nCluster Mean Profile:")
print(cluster_profile.T.to_string())

# Heatmap
fig, ax = plt.subplots(figsize=(max(8, K*2), 10))
sns.heatmap(
    cluster_profile.T,
    annot=True, fmt=".2f", cmap="RdYlGn",
    linewidths=0.5, ax=ax, cbar_kws={"label": "Mean value"}
)
ax.set_title("Cluster Profile Heatmap", fontsize=13, fontweight="bold")
ax.set_xlabel("Cluster")
plt.tight_layout()
plt.savefig("cluster_heatmap.png", bbox_inches="tight")
plt.close()

# Radar Chart per cluster
radar_features = [
    "safety_priority", "efficacy_priority", "comfort_priority",
    "acne_severity", "price_sensitive", "doctor_influenced",
    "self_directed", "n_brands_used"
]
radar_labels = [
    "Safety\nPriority", "Efficacy\nPriority", "Comfort\nPriority",
    "Acne\nSeverity", "Price\nSensitive", "Doctor\nInfluenced",
    "Self\nDirected", "# Brands\nUsed"
]

angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, K, figsize=(5*K, 5), subplot_kw=dict(polar=True))
if K == 1:
    axes = [axes]

fig.suptitle("Radar Profile per Cluster", fontsize=14, fontweight="bold", y=1.02)

for c, ax in enumerate(axes):
    vals = cluster_profile.loc[c, radar_features].tolist()
    max_vals = feat[radar_features].max().tolist()
    vals_norm = [v / m if m > 0 else 0 for v, m in zip(vals, max_vals)]
    vals_norm += vals_norm[:1]

    ax.plot(angles, vals_norm, color=colors[c], linewidth=2)
    ax.fill(angles, vals_norm, color=colors[c], alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(f"Cluster {c}\n(n={int((feat['cluster']==c).sum())})",
                 fontsize=11, fontweight="bold", color=colors[c], pad=15)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cluster_radar.png", bbox_inches="tight")
plt.close()

# Brand distribution per cluster
brand_labels = ["Kiyora", "Bioderma", "Garnier", "Hada Labo", "Simple", "Other"]

fig, axes = plt.subplots(1, K, figsize=(5*K, 5))
if K == 1:
    axes = [axes]

fig.suptitle("Brand Used (Most Frequent) per Cluster", fontsize=13, fontweight="bold")
for c, ax in enumerate(axes):
    mask = df["cluster"] == c
    counts = df.loc[mask, "brand_most_used"].value_counts()
    counts.plot(kind="bar", ax=ax, color=colors[c], edgecolor="white")
    ax.set_title(f"Cluster {c}", fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("cluster_brands.png", bbox_inches="tight")
plt.close()

# 8. PREDICT (assign new respondent to cluster)
print("STEP 8: PREDICT - NEW RESPONDENT")

def predict_cluster(respondent: dict) -> int:
    """
    ส่ง dict ของ feature values เข้ามา (เฉพาะค่า numeric หลัง encode แล้ว)
    คืนค่า cluster id และ probabilities (distance-based)

    Parameters
    ----------
    respondent : dict
        Keys ต้องตรงกับ FEATURES list

    Returns
    -------
    cluster_id : int
    distances   : dict  {cluster_id: distance}
    """
    row = np.array([[respondent.get(f, 0) for f in FEATURES]], dtype=float)
    row_scaled = scaler.transform(row)
    cluster_id = int(kmeans.predict(row_scaled)[0])
    distances = {
        int(i): float(np.linalg.norm(row_scaled - c))
        for i, c in enumerate(kmeans.cluster_centers_)
    }
    return cluster_id, distances


# Example new respondent 
new_respondent = {
    "is_female": 1,
    "age_ord": 3,            # 23-28 ปี
    "income_ord": 5,         # 30,000-34,999
    "is_bangkok": 1,
    "skin_oily": 0,
    "skin_dry": 0,
    "skin_combo": 0,
    "skin_sensitive": 1,
    "acne_severity": 2,      # สิวน้อย
    "n_skin_concerns": 3,
    "use_cw": 1,
    "self_directed": 1,
    "doctor_influenced": 0,
    "friend_influenced": 1,
    "influencer_influenced": 0,
    "n_brands_used": 2,
    "score_deep_cleansing": 4,
    "score_acne_friendly": 5,
    "score_sensitive_friendly": 5,
    "score_no_allergen": 4,
    "score_hypoallergenic": 4,
    "score_moisturized": 3,
    "score_low_friction": 3,
    "score_nourishment": 3,
    "score_eye_friendly": 3,
    "score_oil_control": 4,
    "safety_priority": 4.5,
    "efficacy_priority": 3.5,
    "comfort_priority": 3.0,
    "price_sensitive": 0,
    "performance_driven": 1,
}

pred_cluster, distances = predict_cluster(new_respondent)
print(f"\nNew respondent → Cluster {pred_cluster}")
print("Distance to each cluster center:")
for cid, dist in sorted(distances.items()):
    marker = " ← assigned" if cid == pred_cluster else ""
    print(f"  Cluster {cid}: {dist:.4f}{marker}")


print("STEP 9: SEGMENT INSIGHTS")
# STEP 9: SEGMENT INSIGHTS (Business Interpretation)

# KIYORA CLEAN – CUSTOMER SEGMENT SUMMARY
#
# Clusters were derived from K-Means using 31 features
# covering demographics, skin profile, behavior,
# and purchase decision scores.
#
# โมเดลทำการแบ่งกลุ่มลูกค้าโดยใช้ข้อมูล 31 ตัวแปร
# เพื่อค้นหาลักษณะลูกค้าที่มีพฤติกรรมคล้ายกัน
# สำหรับใช้ในการวางกลยุทธ์ทางการตลาด

# Key Feature Groups
# ------------------

# Demographics :
#   gender, age, income, region
#   ข้อมูลพื้นฐานของลูกค้า เช่น เพศ อายุ รายได้ และภูมิภาค

# Skin Profile :
#   skin type, concerns, acne severity
#   ลักษณะผิว ปัญหาผิว และระดับความรุนแรงของสิว

# Behavior :
#   brand count, influence source
#   พฤติกรรมการเลือกแบรนด์และแหล่งอิทธิพลในการตัดสินใจซื้อ

# Purchase Score :
#   10 product attributes (1–5 scale)
#   คะแนนความสำคัญของคุณสมบัติผลิตภัณฑ์ (ระดับ 1–5)

# Composite KPI :
#   safety, efficacy, comfort priorities
#   ค่าดัชนีรวมที่สะท้อนความสำคัญด้าน
#   ความปลอดภัย ประสิทธิภาพ และความสบายผิว

# Segment Interpretation Guide
# ----------------------------

# High safety_priority
#   → ลูกค้าผิวแพ้ง่าย / กังวลเรื่องสิว
#   → ให้ความสำคัญกับความอ่อนโยนและความปลอดภัย

# High efficacy_priority
#   → ลูกค้าที่เน้นประสิทธิภาพการทำความสะอาด
#   → มักเป็นกลุ่มผิวมันหรือมีปัญหาสิว

# High price_sensitive
#   → ลูกค้าที่ตัดสินใจตามราคาและโปรโมชั่น
#   → มีแนวโน้มเปลี่ยนแบรนด์ได้ง่าย

# High doctor_influenced
#   → ลูกค้าที่เชื่อคำแนะนำจากแพทย์หรือผู้เชี่ยวชาญ
#   → ให้ความน่าเชื่อถือของแบรนด์เป็นหลัก

# NOTE:
# Use cluster_heatmap.png and cluster_radar.png
# to inspect exact characteristics of each segment
# in every model run.
#
# ใช้กราฟ heatmap และ radar chart
# เพื่อดูรายละเอียดของแต่ละ customer segment
# และเปรียบเทียบค่าของแต่ละกลุ่มลูกค้า



# 10. EXPORT RESULTS
print("STEP 10: EXPORT")

df_export = df.copy()
df_export["cluster"] = df["cluster"]
df_export.to_excel("kiyora_clustered_results.xlsx", index=False)

cluster_profile.to_excel("kiyora_cluster_profiles.xlsx")

print(" kiyora_clustered_results.xlsx  - raw data + cluster label")
print(" kiyora_cluster_profiles.xlsx  - mean feature per cluster")
print(" elbow_silhouette.png")
print(" cluster_pca.png")
print(" cluster_heatmap.png")
print(" cluster_radar.png")
print(" cluster_brands.png")
print("\n Pipeline complete!")

print("\n===== UNIQUE BRAND VALUES =====")

brands_raw = (
    df["brand_most_used"]
    .dropna()
    .astype(str)
    .str.strip()
)

# แสดง unique ทั้งหมด
unique_brands = sorted(brands_raw.unique())

print(f"\nTotal unique brands: {len(unique_brands)}\n")

for b in unique_brands:
    count = (brands_raw == b).sum()
    print(f"{b:<30} | count = {count}")

# ===============================
# BRAND RANKING: BEFORE vs AFTER UNSUPERVISED
# ===============================

print("STEP: BRAND RANKING BEFORE vs AFTER CLUSTERING")

# 1) Clean brand names
df["brand_most_used_clean"] = (
    df["brand_most_used"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["brand_most_used_clean"] = df["brand_most_used_clean"].replace({
    "hikari": "Hikari",
    "glow in skin": "Glow in Skin",
})

mask = ~df["brand_most_used_clean"].isin(["Hikari", "Glow in Skin"])
df.loc[mask, "brand_most_used_clean"] = (
    df.loc[mask, "brand_most_used_clean"]
    .str.title()
)

# 2) Group rare brands as Other
brand_counts = df["brand_most_used_clean"].value_counts()
rare_brands = brand_counts[brand_counts < 3].index

df["brand_grouped"] = df["brand_most_used_clean"].replace(
    rare_brands,
    "Other"
)

# ===============================
# BEFORE: Overall brand ranking
# ===============================

before_rank = (
    df["brand_grouped"]
    .value_counts()
    .reset_index()
)

before_rank.columns = ["brand", "count"]
before_rank["percent"] = before_rank["count"] / before_rank["count"].sum() * 100
before_rank["rank"] = before_rank["count"].rank(method="first", ascending=False).astype(int)

print("\nOverall Brand Ranking BEFORE clustering:")
print(before_rank.to_string(index=False))

plt.figure(figsize=(10, 5))
plt.bar(before_rank["brand"], before_rank["count"])
plt.title("Brand Ranking BEFORE Unsupervised Clustering", fontsize=13, fontweight="bold")
plt.xlabel("Brand")
plt.ylabel("Number of Respondents")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("brand_ranking_before.png", bbox_inches="tight")
plt.close()

# ===============================
# AFTER: Brand ranking by cluster
# ===============================

after_rank = (
    df.groupby(["cluster", "brand_grouped"])
    .size()
    .reset_index(name="count")
)

after_rank["percent_in_cluster"] = (
    after_rank["count"]
    / after_rank.groupby("cluster")["count"].transform("sum")
    * 100
)

after_rank["rank_in_cluster"] = (
    after_rank.groupby("cluster")["count"]
    .rank(method="first", ascending=False)
    .astype(int)
)

after_rank = after_rank.sort_values(["cluster", "rank_in_cluster"])

print("\nBrand Ranking AFTER clustering:")
print(after_rank.to_string(index=False))

# Plot top brands per cluster
top_n = 5
after_top = after_rank[after_rank["rank_in_cluster"] <= top_n]

fig, axes = plt.subplots(1, K, figsize=(5*K, 5))
if K == 1:
    axes = [axes]

fig.suptitle("Brand Used (Most Frequent) per Cluster",
             fontsize=13, fontweight="bold")

for c, ax in enumerate(axes):
    mask = df["cluster"] == c

    # TOP 6 ONLY
    counts = (
        df.loc[mask, "brand_most_used"]
        .value_counts()
        .head(6)
    )

    counts.plot(
        kind="bar",
        ax=ax,
        color=colors[c],
        edgecolor="white"
    )

    ax.set_title(f"Cluster {c}", fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("cluster_brands_top6.png", bbox_inches="tight")
plt.close()

# ===============================
# OPTIONAL: Heatmap brand share by cluster
# ===============================

brand_cluster_pct = pd.crosstab(
    df["brand_grouped"],
    df["cluster"],
    normalize="columns"
) * 100

plt.figure(figsize=(max(8, K*2), 8))
sns.heatmap(
    brand_cluster_pct,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    linewidths=0.5,
    cbar_kws={"label": "% within cluster"}
)
plt.title("Brand Share by Cluster (%)", fontsize=13, fontweight="bold")
plt.xlabel("Cluster")
plt.ylabel("Brand")
plt.tight_layout()
plt.savefig("brand_share_heatmap_by_cluster.png", bbox_inches="tight")
plt.close()

# ===============================
# EXPORT
# ===============================

before_rank.to_excel("brand_ranking_before.xlsx", index=False)
after_rank.to_excel("brand_ranking_after_by_cluster.xlsx", index=False)
brand_cluster_pct.to_excel("brand_share_heatmap_by_cluster.xlsx")

print("\nSaved files:")
print(" brand_ranking_before.png")
print(" brand_ranking_after_by_cluster.png")
print(" brand_share_heatmap_by_cluster.png")
print(" brand_ranking_before.xlsx")
print(" brand_ranking_after_by_cluster.xlsx")