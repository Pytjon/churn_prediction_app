import pandas as pd
import joblib
import streamlit as st


def collect_user_input():
    
    data = {
        "CreditScore": st.number_input("Credit Score", 300, 900, step=1),
        "Geography": st.selectbox("Geography", ["France", "Spain", "Germany"]),
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "Age": st.number_input("Age", 18, 100, step=1),
        "Tenure": st.number_input("Tenure", 0, 10, step=1),
        "Balance": st.number_input("Balance", 0.0),
        "NumOfProducts": st.number_input("Number of Products", 1, 4, step=1),
        "HasCrCard": st.selectbox("Has Credit Card", [0, 1]),
        "IsActiveMember": st.selectbox("Is Active Member", [0, 1]),
        "EstimatedSalary": st.number_input("Estimated Salary", 0.0),
    }
    return pd.DataFrame([data])


def collect_user_input_or_file():
    
    uploaded_file = st.file_uploader("Upload CSV file (optional)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        return df, True

    st.markdown("Or manually enter a single customer's details:")
    single_df = collect_user_input()
    return single_df, False


def predict_with_mlp(df, preprocessor, model):
    X = preprocessor.transform(df)
    probs = model.predict(X).flatten()
    labels = ["Likely to Churn" if p >= 0.5 else "Not Likely" for p in probs]
    return probs, labels


def predict_with_lstm(df, preprocessor, model):
    X = preprocessor.transform(df)
    X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
    probs = model.predict(X_lstm).flatten()
    labels = ["Likely to Churn" if p >= 0.5 else "Not Likely" for p in probs]()
    return probs, labels


def predict_with_both(df, preprocessor, mlp_model, lstm_model):
    prob_mlp, _ = predict_with_mlp(df, preprocessor, mlp_model)
    prob_lstm, _ = predict_with_lstm(df, preprocessor, lstm_model)
    avg_prob = (prob_mlp + prob_lstm) / 2
    label = "Likely to Churn" if avg_prob >= 0.5 else "Not Likely"
    return avg_prob, label
