import os
import sys

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


st.set_page_config(page_title="Telco Churn Predictor", page_icon=":bar_chart:", layout="wide")

st.title("Telco Customer Churn Prediction")
st.caption("MLDP project deployment - predicts whether a customer is likely to churn.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "outputs", "models", "telco_churn_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


def add_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering function used inside the saved sklearn pipeline."""
    d = dataframe.copy()
    d["avg_charge_per_tenure"] = d["TotalCharges"] / (d["tenure"] + 1)
    d["is_new_customer"] = (d["tenure"] <= 12).astype(int)
    d["service_count"] = (
        (d["PhoneService"] == "Yes").astype(int)
        + (d["OnlineSecurity"] == "Yes").astype(int)
        + (d["OnlineBackup"] == "Yes").astype(int)
        + (d["DeviceProtection"] == "Yes").astype(int)
        + (d["TechSupport"] == "Yes").astype(int)
        + (d["StreamingTV"] == "Yes").astype(int)
        + (d["StreamingMovies"] == "Yes").astype(int)
    )
    return d


# Compatibility shim for models pickled from notebook scope:
# FunctionTransformer may reference '__main__.add_features'.
if "__main__" in sys.modules:
    setattr(sys.modules["__main__"], "add_features", add_features)


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_resource
def train_fallback_model(data_path: str):
    """Train a fallback model in app if pickle compatibility fails."""
    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn_bin"] = df["Churn"].map({"No": 0, "Yes": 1})

    df = df.drop(columns=["customerID"])
    X = df.drop(columns=["Churn", "Churn_bin"])
    y = df["Churn_bin"]

    numeric_cols = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "avg_charge_per_tenure",
        "is_new_customer",
        "service_count",
    ]
    categorical_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_features, validate=False)),
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=16,
                    min_samples_split=5,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    random_state=2025,
                ),
            ),
        ]
    )

    model.fit(X, y)
    return model


if not os.path.exists(MODEL_PATH):
    st.warning("Saved model file not found. Using in-app fallback trained model.")
    model = train_fallback_model(DATA_PATH)
else:
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

    try:
        pred = model.predict(input_df)[0]
        prob = float(model.predict_proba(input_df)[0, 1]) if hasattr(model, "predict_proba") else None
    except Exception:
        st.warning("Saved model is incompatible with deployment packages. Using fallback model.")
        model = train_fallback_model(DATA_PATH)
        pred = model.predict(input_df)[0]
        prob = float(model.predict_proba(input_df)[0, 1]) if hasattr(model, "predict_proba") else None

    if pred == 1:
        st.error("Prediction: Customer is likely to churn.")
    else:
        st.success("Prediction: Customer is not likely to churn.")

    if prob is not None:
        st.metric("Churn Probability", f"{prob:.2%}")
