# nguyentungson-project1-cranesfpt

# Project 1 - Team Nguyen Tung Son - World Stock Price Analysis & Prediction

**Data Scientist Course — DMP + EDA + ML + Streamlit Dashboard**

## 📁 Cấu trúc thư mục

```
nguyentungson-project1-cranesfpt/
├── World-Stock-Prices-Dataset.csv     # Dataset gốc
├── World_Stock_Prices_Analysis.ipynb  # Notebook DMP + EDA + ML
├── app.py                             # Streamlit dashboard
├── requirements.txt                   # Thư viện cần cài
└── README.md                          # File này
```
## Link Streamlit Online: https://nguyentungson-cranesfptk01.streamlit.app/
## 🚀 Cách chạy

### Bước 1: Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy Notebook (phần DMP + EDA + ML)

```bash
jupyter notebook World_Stock_Prices_Analysis.ipynb
```

Trong notebook, ấn **Run All** (hoặc Shift+Enter từng cell).
Sau khi chạy hết sẽ tạo ra file `World_Stock_Prices_Cleaned.csv`.

### Bước 3: Chạy Streamlit Dashboard

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở ở địa chỉ `http://localhost:8501`.

## 📊 Các bước thực hiện trong Project

| Bước | Nội dung | File thực hiện |
|------|----------|----------------|
| 1 | Load dataset & EDA | notebook |
| 2 | Xử lý missing values | notebook |
| 3 | Drop duplicates | notebook |
| 4 | Feature engineering (year, month, MA20, MA50, daily_return) | notebook |
| 5 | Correlation analysis | notebook |
| 6 | Feature scaling (StandardScaler) | notebook |
| 7 | Train Linear Regression | notebook |
| 8 | Streamlit dashboard interactive | app.py |

## 🔍 Thông tin dataset

- **310,122 dòng × 13 cột**
- **62 brand**: Apple, Amazon, Google, Nike, Adidas, Tesla...
- **7 quốc gia**: USA, Germany, Japan, Switzerland, Canada, Netherlands, France
- **23 ngành**: technology, fitness, footwear, e-commerce, automotive...
- **Thời gian**: 2000-01-03 đến 2025-07-03

## ⚠️ Các lưu ý quan trọng

1. **Cột `Capital Gains` thiếu ~99.99% giá trị** → ta drop cột này.
2. **Cột `Date` có timezone** (VD: `2025-07-03 00:00:00-04:00`) → phải dùng `pd.to_datetime(..., utc=True)`.
3. **Feature Engineering phải groupby theo Ticker** để tính moving average đúng cho từng cổ phiếu.
4. **Dataset lớn (310k dòng)** → khi vẽ scatter plot, sample 5000 dòng cho nhanh.

## 🎯 Kết quả mong đợi

- **R² Score ≈ 0.999** khi dùng Linear Regression với features open/high/low/volume.
- Lý do rất cao: Open, High, Low và Close trong cùng 1 ngày giao dịch có tương quan gần 1.0.

## 💡 Hướng mở rộng

- Dùng **RandomForest / XGBoost / LSTM** để dự đoán tốt hơn.
- Dự đoán giá ngày **hôm sau** dựa trên dữ liệu **hôm nay** (lag features).
- Thêm **filter theo Brand / Country / Industry** trên Streamlit.
- Tích hợp **real-time API** (yfinance) để lấy giá mới.
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![License](https://img.shields.io/badge/License-MIT-green)
