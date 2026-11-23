import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Prediksi Kanker Payudara - Random Forest (TA-06)", layout="wide")

st.title("🔬 Sistem Prediksi Diagnosis Tumor Payudara (Random Forest)")
st.markdown("**Tugas Akhir 06 (TA-06) - Praktikum Pembelajaran Mesin** | Model: `RandomForestClassifier(n_estimators=50, max_depth=5)`")

data = load_breast_cancer()
X_df = pd.DataFrame(data.data, columns=data.feature_names)
y_sr = pd.Series(data.target)

@st.cache_resource
def get_model():
    try:
        return joblib.load("rf_breast_cancer_model.pkl")
    except Exception:
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X_df, y_sr)
        return clf

model = get_model()

st.sidebar.header("Input Parameter Klinis Utama")
st.sidebar.info("Sesuaikan nilai fitur utama di bawah ini (fitur lainnya menggunakan nilai median klinis).")

medians = X_df.median().to_dict()
top_features = [
    "worst concave points", "worst radius", "mean perimeter",
    "worst area", "mean concave points", "mean concavity",
    "worst perimeter", "mean radius", "worst concavity", "area error"
]

user_inputs = medians.copy()
for feat in top_features:
    min_val = float(X_df[feat].min())
    max_val = float(X_df[feat].max())
    med_val = float(medians[feat])
    user_inputs[feat] = st.sidebar.slider(feat, min_val, max_val, med_val)

input_df = pd.DataFrame([user_inputs])

col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("Data Input Pasien (10 Fitur Teratas)")
    st.dataframe(input_df[top_features].T.rename(columns={0: "Nilai Input"}), use_container_width=True)

with col2:
    st.subheader("Hasil Prediksi Model")
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    if pred == 1:
        st.success(f"### Diagnosis: BENIGN (Jinak)\n**Probabilitas Jinak:** {proba[1]*100:.2f}%")
    else:
        st.error(f"### Diagnosis: MALIGNANT (Ganas)\n**Probabilitas Ganas:** {proba[0]*100:.2f}%")

    st.bar_chart(pd.DataFrame({"Probabilitas": [proba[0], proba[1]]}, index=["Malignant (0)", "Benign (1)"]))
