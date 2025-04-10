import streamlit as st
import joblib
import re
import string
import pandas as pd

# --- Load mô hình ---
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# --- Hàm làm sạch văn bản ---
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# --- Dự đoán nhiều dòng văn bản ---
def predict_multiple_reviews(reviews):
    cleaned = [clean_text(r) for r in reviews]
    vectors = vectorizer.transform(cleaned)
    preds = model.predict(vectors)
    probas = model.predict_proba(vectors)
    return preds, probas

# --- Tô màu kết quả ---
def highlight_sentiment(val):
    color = ""
    if val == "positive":
        color = "green"
    elif val == "negative":
        color = "red"
    elif val == "neutral":
        color = "gray"
    return f"color: {color}; font-weight: bold"

# --- Giao diện ---
st.set_page_config(page_title="🎬 IMDb Sentiment Analyzer", layout="centered")
st.title("🎬 Dự đoán cảm xúc review phim (IMDb)")

tab1, tab2 = st.tabs(["📝 Nhập văn bản", "📁 Tải file .txt"])

with tab1:
    st.markdown("Nhập **một hoặc nhiều câu**, mỗi câu trên **một dòng riêng**:")

    with st.expander("📌 Xem ví dụ mẫu"):
        st.code("Phim này quá dở, mình không thể xem nổi.\nTôi rất thích bộ phim này, thật cảm xúc!")

    input_text = st.text_area("✍️ Dán hoặc nhập review tại đây:", height=200)

    if st.button("📊 Dự đoán cảm xúc", key="text_input"):
        if not input_text.strip():
            st.warning("⚠️ Vui lòng nhập ít nhất một dòng review.")
        else:
            reviews = [line for line in input_text.split("\n") if line.strip()]
            preds, probas = predict_multiple_reviews(reviews)

            df_result = pd.DataFrame({
                "Review": reviews,
                "Dự đoán": preds
            })

            styled_df = df_result.style.applymap(highlight_sentiment, subset=["Dự đoán"])

            st.subheader("📋 Kết quả phân tích:")
            st.dataframe(styled_df, use_container_width=True)

            st.subheader("📊 Thống kê tổng hợp:")
            st.bar_chart(df_result["Dự đoán"].value_counts())

with tab2:
    uploaded_file = st.file_uploader("📎 Tải file .txt chứa review", type=["txt"])

    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        if len(lines) == 0:
            st.warning("⚠️ File không chứa nội dung hợp lệ.")
        else:
            preds, probas = predict_multiple_reviews(lines)
            df_result = pd.DataFrame({
                "Review": lines,
                "Dự đoán": preds
            })

            styled_df = df_result.style.applymap(highlight_sentiment, subset=["Dự đoán"])

            st.subheader("📋 Kết quả phân tích:")
            st.dataframe(styled_df, use_container_width=True)

            st.subheader("📊 Thống kê tổng hợp:")
            st.bar_chart(df_result["Dự đoán"].value_counts())
