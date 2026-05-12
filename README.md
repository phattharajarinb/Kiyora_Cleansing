# Kiyora Clean AI Customer Segmentation

โปรเจกต์นี้เชื่อมหน้าเว็บ HTML เดิมเข้ากับ Backend และโมเดล Unsupervised Learning สำหรับวิเคราะห์พฤติกรรมผู้บริโภค Kiyora Clean

## ไฟล์สำคัญ

- `app.py` — Backend API ด้วย Python HTTP Server
- `kiyora_model.py` — โมเดล K-Means Customer Segmentation ที่ปรับให้เข้ากับเว็บ
- `index.html` — หน้า AI Prediction เรียก `/api/predict`
- `dashboard.html` — Dashboard เรียก `/api/dashboard`
- `products.html` — หน้าสินค้า
- `kiyora_clustering_original.py` — ไฟล์โมเดลต้นฉบับที่ใช้อ้างอิงแนวคิด feature engineering / K-Means

## วิธีรัน

```bash
cd kiyora_ai_backend
python app.py
```

จากนั้นเปิดเว็บ:

```text
http://127.0.0.1:5000
```

## API

### POST `/api/predict`

รับข้อมูลจากหน้าเว็บ เช่น:

```json
{
  "skinType": "ผิวแพ้ง่าย",
  "age": 22,
  "makeupFrequency": "บ่อย",
  "skinProblems": ["ผิวแพ้ง่าย", "สิวอักเสบ"],
  "budget": "500–1000",
  "usedBefore": "เคย"
}
```

คืนค่า cluster, risk, loyalty, purchase intent, repeat probability และคำแนะนำเชิงกลยุทธ์

### GET `/api/dashboard`

คืนค่าสรุป segment ทั้งหมดและ insight สำหรับหน้า Dashboard

## แนวคิดโมเดล

โมเดลใช้แนวทางจากไฟล์ `kiyora_clustering_original.py`:

- Feature Engineering 31 ตัวแปร
- Standardization
- K-Means Clustering
- Segment Profiling
- Business Interpretation

เพื่อให้เว็บรันได้ทันทีโดยไม่ต้องติดตั้ง library เพิ่ม `kiyora_model.py` จึงใช้ KMeansLite และ StandardScalerLite ที่เขียนด้วย Python ล้วน หากมีไฟล์ Excel dataset จริง สามารถต่อยอดให้โหลดข้อมูลจริงเข้า pipeline เดิมได้
