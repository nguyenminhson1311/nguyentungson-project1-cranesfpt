"""
PROJECT 15: WORLD STOCK PRICE ANALYSIS & PREDICTION
Streamlit Dashboard - app.py

Chạy lệnh: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="World Stock Price Prediction",
    layout="wide",
    page_icon="📈"
)

st.markdown(
    "<h1 style='text-align:center;color:#2E7D32;'>📈 World Stock Price Analysis & Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:#555;'>DMP + EDA + Linear Regression Model + Interactive Dashboard</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ============================================================
# SESSION STATE
# ============================================================
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV (tùy chọn)", type=["csv"])

if st.sidebar.button("📂 Load Dataset"):
    try:
        if uploaded_file is not None:
            st.session_state.raw_df = pd.read_csv(uploaded_file)
        else:
            st.session_state.raw_df = pd.read_csv("World-Stock-Prices-Dataset.csv")
        # Chuẩn hóa tên cột ngay lúc load
        st.session_state.raw_df.columns = (
            st.session_state.raw_df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        st.sidebar.success("Dataset Loaded Successfully!")
    except FileNotFoundError:
        st.sidebar.error("❌ Không tìm thấy file World-Stock-Prices-Dataset.csv. Hãy upload file hoặc đặt file cạnh app.py.")
    except Exception as e:
        st.sidebar.error(f"Lỗi: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Luồng: **Load → EDA → Preprocess → Train Model → Predict → Download**")

# ============================================================
# 1. DATASET OVERVIEW
# ============================================================
if st.session_state.raw_df is not None:
    df = st.session_state.raw_df.copy()

    st.subheader("🔹 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{df.shape[0]:,}")
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    c4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    st.subheader("📄 Dataset Preview (100 dòng đầu)")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    # Hiển thị missing values
    st.subheader("🔍 Missing Values per Column")
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / len(df) * 100).round(2)
    })
    st.dataframe(missing_df, use_container_width=True)

# ============================================================
# 2. EDA SECTION (BEFORE)
# ============================================================
if st.session_state.raw_df is not None:
    st.markdown("---")
    st.subheader("📊 Exploratory Data Analysis (Before Preprocessing)")

    # Chỉ vẽ histogram cho các cột số chính để chạy nhanh
    price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]

    col_select = st.selectbox("Chọn cột để xem phân phối:", price_cols, key="eda_hist")
    if col_select:
        fig = px.histogram(
            df,
            x=col_select,
            nbins=50,
            marginal="box",
            title=f"{col_select} Distribution (Before)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("📌 Correlation Heatmap (Before)")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap (Before)",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)

    # Time-series plot theo brand
    if "brand_name" in df.columns and "date" in df.columns and "close" in df.columns:
        st.subheader("📈 Stock Price Over Time by Brand")
        brands = sorted(df["brand_name"].dropna().unique().tolist())
        selected_brand = st.selectbox("Chọn Brand:", brands, key="brand_ts")
        if selected_brand:
            brand_df = df[df["brand_name"] == selected_brand].copy()
            brand_df["date"] = pd.to_datetime(brand_df["date"], utc=True, errors="coerce")
            brand_df = brand_df.dropna(subset=["date"]).sort_values("date")
            fig = px.line(
                brand_df,
                x="date",
                y="close",
                title=f"{selected_brand.title()} - Close Price Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 3. PREPROCESSING
# ============================================================
if st.session_state.raw_df is not None:
    st.markdown("---")
    st.subheader("🧹 Data Preprocessing")

    if st.button("▶️ Start Preprocessing"):
        df_prep = st.session_state.raw_df.copy()

        # 3.1 Drop column capital_gains nếu missing > 90%
        if "capital_gains" in df_prep.columns:
            miss_pct = df_prep["capital_gains"].isnull().sum() / len(df_prep) * 100
            if miss_pct > 90:
                df_prep = df_prep.drop(columns=["capital_gains"])

        # 3.2 Fill missing: số -> median, string -> mode
        for col in df_prep.columns:
            if df_prep[col].dtype in ["int64", "float64"]:
                df_prep[col].fillna(df_prep[col].median(), inplace=True)
            elif df_prep[col].dtype == "object":
                if df_prep[col].isnull().any():
                    df_prep[col].fillna(df_prep[col].mode()[0], inplace=True)

        # 3.3 Drop duplicates
        df_prep.drop_duplicates(inplace=True)

        # 3.4 Convert date (có timezone nên dùng utc=True)
        if "date" in df_prep.columns:
            df_prep["date"] = pd.to_datetime(df_prep["date"], utc=True, errors="coerce")
            df_prep.dropna(subset=["date"], inplace=True)
            df_prep["year"] = df_prep["date"].dt.year
            df_prep["month"] = df_prep["date"].dt.month
            df_prep["day"] = df_prep["date"].dt.day

        # 3.5 Feature Engineering: daily_return, ma_20, ma_50
        if "ticker" in df_prep.columns and "close" in df_prep.columns:
            df_prep = df_prep.sort_values(["ticker", "date"]).reset_index(drop=True)
            df_prep["daily_return"] = df_prep.groupby("ticker")["close"].pct_change().fillna(0)
            df_prep["ma_20"] = (
                df_prep.groupby("ticker")["close"]
                .transform(lambda x: x.rolling(20).mean())
                .fillna(df_prep["close"])
            )
            df_prep["ma_50"] = (
                df_prep.groupby("ticker")["close"]
                .transform(lambda x: x.rolling(50).mean())
                .fillna(df_prep["close"])
            )

        st.session_state.clean_df = df_prep
        st.success("✅ Preprocessing Completed Successfully!")

# ============================================================
# 4. AFTER PREPROCESSING VISUALIZATIONS
# ============================================================
if st.session_state.clean_df is not None:
    st.markdown("---")
    st.subheader("📊 Visualizations After Preprocessing")

    clean_df = st.session_state.clean_df.copy()
    price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in clean_df.columns]

    col_after = st.selectbox("Chọn cột để xem phân phối sau xử lý:", price_cols, key="eda_hist_after")
    if col_after:
        fig = px.histogram(
            clean_df,
            x=col_after,
            nbins=50,
            marginal="box",
            title=f"{col_after} Distribution (After)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap After
    st.subheader("📌 Correlation Heatmap (After)")
    num_cols_clean = clean_df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols_clean) > 1:
        corr_after = clean_df[num_cols_clean].corr()
        fig = px.imshow(
            corr_after,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap (After)",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 5. BEFORE vs AFTER COMPARISON
# ============================================================
if st.session_state.clean_df is not None:
    st.markdown("---")
    st.subheader("📊 Before vs After Comparison")

    comparison_df = pd.DataFrame({
        "Aspect": ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
        "Before": [
            df.shape[0],
            df.shape[1],
            int(df.isnull().sum().sum()),
            int(df.duplicated().sum())
        ],
        "After": [
            clean_df.shape[0],
            clean_df.shape[1],
            int(clean_df.isnull().sum().sum()),
            int(clean_df.duplicated().sum())
        ]
    })
    st.dataframe(comparison_df, use_container_width=True)

    # Distribution Comparison
    for col in ["open", "close"]:
        if col in df.columns and col in clean_df.columns:
            temp = pd.DataFrame({
                "Before": df[col].dropna().reset_index(drop=True),
                "After": clean_df[col].dropna().reset_index(drop=True)
            })
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=temp["Before"], name="Before", opacity=0.6))
            fig.add_trace(go.Histogram(x=temp["After"], name="After", opacity=0.6))
            fig.update_layout(
                barmode="overlay",
                title=f"{col} Before vs After Distribution",
                xaxis_title=col,
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 6. MACHINE LEARNING - TRAIN MODEL
# ============================================================
if st.session_state.clean_df is not None:
    st.markdown("---")
    st.subheader("🤖 Train Linear Regression Model")

    clean_df = st.session_state.clean_df.copy()

    potential_features = ["open", "high", "low", "volume", "ma_20", "ma_50", "daily_return"]
    available_features = [c for c in potential_features if c in clean_df.columns]
    target = "close"

    if target in clean_df.columns and len(available_features) >= 2:
        selected_features = st.multiselect(
            "Chọn features để train model:",
            available_features,
            default=available_features
        )

        if st.button("🚀 Train Model") and len(selected_features) >= 1:
            X = clean_df[selected_features].copy()
            y = clean_df[target].copy()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.feature_cols = selected_features

            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{r2:.4f}")
            m2.metric("MAE", f"{mae:.4f}")
            m3.metric("RMSE", f"{rmse:.4f}")

            # Actual vs Predicted
            sample_size = min(2000, len(y_test))
            sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
            fig = px.scatter(
                x=y_test.values[sample_idx],
                y=y_pred[sample_idx],
                labels={"x": "Actual Close", "y": "Predicted Close"},
                title="Actual vs Predicted Close Prices",
                opacity=0.5
            )
            fig.add_shape(
                type="line",
                x0=y_test.min(), y0=y_test.min(),
                x1=y_test.max(), y1=y_test.max(),
                line=dict(color="red", dash="dash")
            )
            st.plotly_chart(fig, use_container_width=True)

            # Coefficients
            coef_df = pd.DataFrame({
                "Feature": selected_features,
                "Coefficient": model.coef_
            }).sort_values("Coefficient", key=abs, ascending=False)
            st.subheader("📊 Model Coefficients")
            st.dataframe(coef_df, use_container_width=True)

# ============================================================
# 7. PREDICTION PANEL (User Input)
# ============================================================
if st.session_state.model is not None:
    st.markdown("---")
    st.subheader("🔮 Predict Close Price (Custom Input)")

    with st.form("prediction_form"):
        inputs = {}
        cols = st.columns(2)
        for i, feat in enumerate(st.session_state.feature_cols):
            col = cols[i % 2]
            default_val = float(st.session_state.clean_df[feat].median())
            inputs[feat] = col.number_input(
                f"{feat}",
                value=default_val,
                format="%.4f"
            )

        submitted = st.form_submit_button("🎯 Predict Close Price")
        if submitted:
            input_df = pd.DataFrame([inputs])[st.session_state.feature_cols]
            input_scaled = st.session_state.scaler.transform(input_df)
            prediction = st.session_state.model.predict(input_scaled)[0]
            st.success(f"💰 Predicted Close Price: **${prediction:.4f}**")

# ============================================================
# 8. DOWNLOAD CLEANED DATASET
# ============================================================
if st.session_state.clean_df is not None:
    st.markdown("---")
    st.subheader("💾 Download Cleaned Dataset")
    csv_data = st.session_state.clean_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV",
        csv_data,
        "World_Stock_Prices_Cleaned.csv",
        "text/csv"
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;'>Project 15 - World Stock Prices | Data Scientist Course</p>",
    unsafe_allow_html=True
)
