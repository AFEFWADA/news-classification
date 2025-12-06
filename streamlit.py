import streamlit as st
import pickle
import joblib
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
import itertools
import os

# TensorFlow / Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# 1️⃣ Helper functions to safely load files
# ==========================
def safe_load_joblib(file_path):
    if not os.path.exists(file_path):
        st.error(f"❌ File not found: {file_path}. Make sure it exists in your app directory.")
        st.stop()
    return joblib.load(file_path)

def safe_load_pickle(file_path):
    if not os.path.exists(file_path):
        st.error(f"❌ File not found: {file_path}. Make sure it exists in your app directory.")
        st.stop()
    with open(file_path, "rb") as f:
        return pickle.load(f)

def safe_load_keras_model(file_path):
    if not os.path.exists(file_path):
        st.error(f"❌ Model file not found: {file_path}. Make sure it exists in your app directory.")
        st.stop()
    return load_model(file_path)

# ==========================
# 2️⃣ Load models safely
# ==========================
tfidf_vectorizer = safe_load_joblib("tfidf_vectorizer.pkl")
svc_model = safe_load_joblib("linear_svc_model.pkl")
bilstm_model = safe_load_keras_model("guardian_bilstm_model.h5")
tokenizer = safe_load_pickle("tokenizer.pkl")
le = safe_load_pickle("label_encoder.pkl")

st.success("✅ All models and files loaded successfully!")

MAX_LEN = 400  # must match training

# Saved metrics
ML_TEST_ACCURACY = 0.7365
ML_TEST_F1 = 0.7301
DL_TEST_ACCURACY = 0.7139
DL_TEST_F1 = 0.7033

# ==========================
# 3️⃣ Text preprocessing
# ==========================
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
important_words = {"uk", "us", "eu", "pm", "un", "ai", "tv", "gp"}

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    tokens = word_tokenize(text.lower())
    cleaned = []
    for t in tokens:
        if t.isdigit() or t in important_words:
            cleaned.append(t)
            continue
        if t in stop_words:
            continue
        if len(t) > 2:
            cleaned.append(lemmatizer.lemmatize(t))
    return " ".join(cleaned)

# ==========================
# 4️⃣ Prediction function
# ==========================
def predict_news(text):
    processed = clean_text(text)

    # ML Prediction
    tfidf_input = tfidf_vectorizer.transform([processed])
    ml_pred = svc_model.predict(tfidf_input)[0]

    # ML probabilities
    try:
        ml_decision = svc_model.decision_function(tfidf_input)
        if len(ml_decision.shape) == 1:  # binary
            ml_probs = 1 / (1 + np.exp(-ml_decision))
            ml_probs_full = np.array([1 - ml_probs[0], ml_probs[0]])
        else:
            ml_probs = np.exp(ml_decision) / np.sum(np.exp(ml_decision))
            ml_probs_full = ml_probs.flatten()
    except:
        ml_probs_full = np.zeros(len(le.classes_))

    # DL Prediction
    seq = pad_sequences(tokenizer.texts_to_sequences([processed]),
                        maxlen=MAX_LEN, padding="post")
    probs = bilstm_model.predict(seq)[0]
    dl_pred_idx = np.argmax(probs)
    dl_pred = le.inverse_transform([dl_pred_idx])[0]

    return {
        "processed_text": processed,
        "ml_prediction": ml_pred,
        "ml_test_accuracy": ML_TEST_ACCURACY,
        "ml_test_f1": ML_TEST_F1,
        "ml_probabilities": ml_probs_full,
        "dl_prediction": dl_pred,
        "dl_test_accuracy": DL_TEST_ACCURACY,
        "dl_test_f1": DL_TEST_F1,
        "dl_probabilities": probs
    }

# ==========================
# 5️⃣ Streamlit UI
# ==========================
st.title("📰 News Classification (ML + Deep Learning)")
st.write("Enter a news article and get ML & DL predictions with accuracy scores and confidence visualization.")

news_text = st.text_area("✏️ Paste your news article here:", height=200)

if st.button("🚀 Predict"):
    if not news_text.strip():
        st.warning("Please enter a news article!")
    else:
        with st.spinner("Analyzing..."):
            result = predict_news(news_text)

        dl_probs = result["dl_probabilities"]
        ml_probs = result["ml_probabilities"]
        categories = le.classes_
        base_colors = plt.cm.tab20.colors

        # Processed text
        st.subheader("🧹 Processed Text")
        st.code(result["processed_text"])

        # ML & DL Predictions
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 ML Prediction (LinearSVC)")
            st.success(result["ml_prediction"])
            st.write(f"**Accuracy:** {result['ml_test_accuracy']:.4f}")
            st.write(f"**F1-Score:** {result['ml_test_f1']:.4f}")
        with col2:
            st.subheader("🧠 DL Prediction (BiLSTM)")
            st.info(result["dl_prediction"])
            st.write(f"**Accuracy:** {result['dl_test_accuracy']:.4f}")
            st.write(f"**F1-Score:** {result['dl_test_f1']:.4f}")

        # DL probabilities with colors
        st.subheader("🔹 DL Probabilities with Labels")
        color_cycle = itertools.cycle(base_colors)
        for i in range(len(dl_probs)):
            if dl_probs[i] > 0:
                color = matplotlib.colors.rgb2hex(next(color_cycle))
                st.markdown(f"<span style='color:{color};'>● {categories[i]}: {dl_probs[i]*100:.1f}%</span>",
                            unsafe_allow_html=True)

        # Line chart
        st.subheader("📈 ML + DL Probabilities (Line Chart)")
        fig_line, ax_line = plt.subplots(figsize=(10, 4))
        non_zero_indices = [i for i, p in enumerate(dl_probs) if p > 0]
        non_zero_dl_probs = [dl_probs[i] for i in non_zero_indices]
        non_zero_ml_probs = [ml_probs[i] for i in non_zero_indices]
        non_zero_categories = [categories[i] for i in non_zero_indices]

        ax_line.plot(non_zero_indices, non_zero_dl_probs, marker='o', linestyle='-', color='blue', linewidth=2, label='DL')
        ax_line.plot(non_zero_indices, non_zero_ml_probs, marker='s', linestyle='--', color='red', linewidth=2, label='ML')

        ax_line.set_xticks(non_zero_indices)
        ax_line.set_xticklabels(non_zero_categories, rotation=45, ha='right')
        ax_line.set_ylim([0, 1])
        ax_line.set_ylabel("Probability")
        ax_line.set_title("ML + DL Prediction Probabilities Across Categories")
        ax_line.grid(True)
        ax_line.legend()

        st.pyplot(fig_line)
