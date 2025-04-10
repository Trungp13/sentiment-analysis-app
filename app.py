import streamlit as st
import joblib
import re
import string

# Load mô hình và vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Hàm tiền xử lý văn bản
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# Giao diện Streamlit
st.set_page_config(page_title="Sentiment Analysis IMDb", layout="centered")
st.title("🎬 Dự đoán cảm xúc review phim (IMDb)")

input_text = st.text_area("✍️ Nhập review phim tại đây:", height=150)

if st.button("Dự đoán cảm xúc"):
    if input_text.strip() == "":
        st.warning("Vui lòng nhập nội dung review.")
    else:
        cleaned = clean_text(input_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]
        prob_dict = dict(zip(model.classes_, proba))

        st.subheader("📌 Kết quả dự đoán:")
        st.markdown(f"**Cảm xúc:** `{prediction}`")

        st.subheader("📊 Xác suất:")
        st.bar_chart(prob_dict)
