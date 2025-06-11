
import streamlit as st
import joblib
import re
import string
import pandas as pd
import altair as alt
import random
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import html

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = html.unescape(text)
    text = BeautifulSoup(text, "html5lib").get_text()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

def predict_multiple_reviews(reviews):
    cleaned = [clean_text(r) for r in reviews]
    vectors = vectorizer.transform(cleaned)
    preds = model.predict(vectors)
    probas = model.predict_proba(vectors)
    return preds, probas

def highlight_sentiment(val):
    color = "green" if val == "positive" else "red"
    return f"color: {color}; font-weight: bold"

st.set_page_config(page_title="🎬 IMDb Sentiment Analyzer", layout="centered")
st.title("🎬 Dự đoán cảm xúc review phim (IMDb)")

tab1, tab2 = st.tabs(["📝 Nhập văn bản", "📁 Tải file .txt"])

with tab1:
    st.markdown("Nhập **một hoặc nhiều câu**, mỗi câu trên **một dòng riêng**:")
    with st.expander("📌 Ví dụ mẫu"):
        st.code("This movie is terrible, I couldn’t even finish it.\nI absolutely loved this film, very emotional!")
    input_text = st.text_area("✍️ Nhập hoặc dán review tại đây:", height=200)
    if st.button("📊 Dự đoán cảm xúc", key="text_input"):
        reviews = [line.strip() for line in input_text.split("\n") if line.strip()]
        if not reviews:
            st.warning("⚠️ Vui lòng nhập ít nhất một dòng review.")
        else:
            preds, _ = predict_multiple_reviews(reviews)
            df_result = pd.DataFrame({"Review": reviews, "Dự đoán": preds})
            st.subheader("📋 Kết quả phân tích:")
            st.dataframe(df_result.style.applymap(highlight_sentiment, subset=["Dự đoán"]), use_container_width=True)

            st.subheader("📊 Thống kê tổng hợp:")
            sentiment_counts = df_result["Dự đoán"].value_counts().reset_index()
            sentiment_counts.columns = ["Cảm xúc", "Số lượng"]
            color_map = {"positive": "green", "negative": "red"}
            chart = alt.Chart(sentiment_counts).mark_bar().encode(
                x=alt.X("Cảm xúc", sort=["positive", "negative"]),
                y="Số lượng",
                color=alt.Color("Cảm xúc", scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())))
            ) + alt.Chart(sentiment_counts).mark_text(
                align='center', baseline='bottom', dy=-5
            ).encode(x="Cảm xúc", y="Số lượng", text="Số lượng")
            st.altair_chart(chart, use_container_width=True)
            st.subheader("📑 Bảng tổng hợp:")
            st.table(sentiment_counts)

with tab2:
    uploaded_file = st.file_uploader("📎 Tải file .txt chứa review", type=["txt"])
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        total_lines = len(lines)
        if total_lines == 0:
            st.warning("⚠️ File không chứa nội dung hợp lệ.")
        else:
            st.success(f"✅ Đã đọc {total_lines} dòng từ file.")
            sample_size = st.slider("📌 Số dòng muốn chọn ngẫu nhiên để phân tích:", min_value=10, max_value=min(2000, total_lines), value=500, step=50)
            if st.button("🔍 Phân tích cảm xúc"):
                sampled_lines = random.sample(lines, sample_size)
                preds, probas = [], []
                progress = st.progress(0)
                for i in range(0, sample_size, 100):
                    chunk = sampled_lines[i:i+100]
                    p, pr = predict_multiple_reviews(chunk)
                    preds.extend(p)
                    probas.extend(pr)
                    progress.progress(min(1.0, (i + 100) / sample_size))
                df_result = pd.DataFrame({"Review": sampled_lines, "Dự đoán": preds})
                st.subheader("📋 Kết quả phân tích:")
                st.dataframe(df_result.style.applymap(highlight_sentiment, subset=["Dự đoán"]), use_container_width=True)
                st.subheader("📊 Thống kê tổng hợp:")
                sentiment_counts = df_result["Dự đoán"].value_counts().reset_index()
                sentiment_counts.columns = ["Cảm xúc", "Số lượng"]
                color_map = {"positive": "green", "negative": "red"}
                chart = alt.Chart(sentiment_counts).mark_bar().encode(
                    x=alt.X("Cảm xúc", sort=["positive", "negative"]),
                    y="Số lượng",
                    color=alt.Color("Cảm xúc", scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())))
                ) + alt.Chart(sentiment_counts).mark_text(
                    align='center', baseline='bottom', dy=-5
                ).encode(x="Cảm xúc", y="Số lượng", text="Số lượng")
                st.altair_chart(chart, use_container_width=True)
                st.subheader("📑 Bảng tổng hợp:")
                st.table(sentiment_counts)

                # 🎯 Gợi ý xem phim (có kiểm tra biến)
                if not sentiment_counts.empty:
                    total_reviews = sentiment_counts["Số lượng"].sum()
                    if "positive" in sentiment_counts["Cảm xúc"].values:
                        positive_count = sentiment_counts[sentiment_counts["Cảm xúc"] == "positive"]["Số lượng"].values[0]
                    else:
                        positive_count = 0
                    percent_positive = (positive_count / total_reviews) * 100
                    st.subheader("🎯 Gợi ý xem phim:")
                    if percent_positive >= 70:
                        st.success(f"✅ {percent_positive:.1f}% đánh giá tích cực → **NÊN XEM!**")
                    elif percent_positive <= 30:
                        st.error(f"❌ Chỉ {percent_positive:.1f}% đánh giá tích cực → **KHÔNG NÊN XEM.**")
                    else:
                        st.warning(f"🤔 {percent_positive:.1f}% đánh giá tích cực → **Cân nhắc trước khi xem.**")
                else:
                    st.info("Không đủ dữ liệu để đưa ra gợi ý xem phim.")
