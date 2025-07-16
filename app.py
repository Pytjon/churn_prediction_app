import streamlit as st
import joblib
from keras.models import load_model
from inference import (
    collect_user_input_or_file,
    predict_with_mlp,
    predict_with_lstm,
    predict_with_both
)

@st.cache_resource
def load_assets():
    preprocessor = joblib.load("preprocessing/preprocessor.pkl")
    mlp_model = load_model("models/mlp_model.keras")
    lstm_model = load_model("models/lstm_model.keras")
    return preprocessor, mlp_model, lstm_model

preprocessor, mlp_model, lstm_model = load_assets()

st.title("Customer Churn Prediction")

df, is_batch = collect_user_input_or_file()

model_option = st.radio("Choose model to use:", ["MLP", "LSTM", "Both (Average)"])

if st.button("Predict Churn"):
    if model_option == "MLP":
        probs, labels = predict_with_mlp(df, preprocessor, mlp_model)
    elif model_option == "LSTM":
        probs, labels = predict_with_lstm(df, preprocessor, lstm_model)
    else:
        probs, labels = predict_with_both(df, preprocessor, mlp_model, lstm_model)

    results = df.copy()
    results["Prediction"] = labels
    results["Probability"] = probs.round(4)

    st.subheader("Prediction Results")
    st.dataframe(results)

