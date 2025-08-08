import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load model dan preprocessing
@st.cache_resource
def load_artifacts():
    model = joblib.load('model/xgb_model.pkl')
    preprocessor = joblib.load('model/preprocessor.pkl')
    le = joblib.load('model/label_encoder.pkl')
    return model, preprocessor, le

model, preprocessor, le = load_artifacts()

# UI
st.title("Prediksi Engagement Pelanggan")
st.write("Aplikasi ini memprediksi apakah pelanggan akan berinteraksi dengan layanan makanan online.")

# Input form
with st.form("input_form"):
    age = st.slider("Usia", 18, 60, 25)
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
    marital_status = st.selectbox("Status Pernikahan", ["Single", "Married"])
    family_size = st.slider("Jumlah Anggota Keluarga", 1, 8, 3)

    occupation = st.selectbox("Pekerjaan", [
        "Healthcare", "Engineer", "Artist", "Doctor", "Lawyer",
        "Entertainment", "Executive", "Marketing", "Homemaker", "Other"
    ])

    monthly_income = st.number_input("Pendapatan Bulanan", min_value=0, step=500)

    education = st.selectbox("Kualifikasi Pendidikan", [
        "Graduate", "PhD", "Post-Graduate", "High School", "Basic", "Other"
    ])
    
    submitted = st.form_submit_button("Prediksi")
    
    if submitted:
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Marital Status': marital_status,
            'Family size': family_size,
            'Occupation': occupation,
            'Monthly Income': monthly_income,
            'Educational Qualifications': education
        }])

        # Preprocess → Predict → Decode
        X_processed = preprocessor.transform(input_data)
        y_pred = model.predict(X_processed)
        y_pred_original = le.inverse_transform(y_pred)  # 1/0 → Yes/No
        
        st.success(f"Hasil Prediksi: {y_pred_original[0]}")
