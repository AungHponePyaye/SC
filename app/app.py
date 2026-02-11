import os
import pandas as pd
import streamlit as st
import joblib


st.set_page_config(page_title="Telco Churn Predictor", page_icon=":bar_chart:", layout="wide")

st.title("Telco Customer Churn Prediction")
st.caption("MLDP project deployment - predicts whether a customer is likely to churn.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "outputs", "models", "telco_churn_model.pkl")


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Run the notebook first to generate "
        "`outputs/models/telco_churn_model.pkl`."
    )
    st.stop()

model = load_model(MODEL_PATH)

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1], index=0)
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protect = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=0.1,
    )
    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=1.0,
    )


if st.button("Predict Churn", type="primary"):
    input_df = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": lines,
                "InternetService": internet,
                "OnlineSecurity": online_sec,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protect,
                "TechSupport": tech_support,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
        ]
    )

    pred = model.predict(input_df)[0]
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(input_df)[0, 1])
    else:
        prob = None

    if pred == 1:
        st.error("Prediction: Customer is likely to churn.")
    else:
        st.success("Prediction: Customer is not likely to churn.")

    if prob is not None:
        st.metric("Churn Probability", f"{prob:.2%}")

