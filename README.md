# 🧠 Customer Churn Prediction App

A Streamlit web app that uses trained **MLP** and **LSTM** models to predict customer churn based on banking data. Supports both manual input and CSV batch uploads.

![Streamlit](https://streamlit.io/images/brand/streamlit-mark-color.png)

---

## 🚀 Features

- 🔍 Predict churn using:
  - Multi-Layer Perceptron (MLP)
  - Long Short-Term Memory (LSTM)
  - Both (average prediction)
- 📥 Manual input or CSV upload
- 📊 Probability scores and classification labels
- 🖥️ CPU-friendly inference

---

## 🛠️ How to Run Locally

```bash
git clone https://github.com/your-username/churn_prediction_app.git
cd churn_prediction_app
pip install -r requirements.txt
streamlit run app.py
```

📂 File Structure
```
.
├── app.py
├── inference.py
├── requirements.txt
├── models/
│   ├── mlp_model.keras
│   └── lstm_model.keras
├── preprocessing/
│   └── preprocessor.pkl
└── README.md
```

📈 Input Features
```

The app expects the following customer features:

CreditScore

Geography — France, Spain, Germany

Gender — Male or Female

Age

Tenure

Balance

NumOfProducts

HasCrCard — 0 or 1

IsActiveMember — 0 or 1

EstimatedSalary
```

🧠 Model Info
```
MLP: Feedforward neural network trained on processed customer data

LSTM: Single-timestep recurrent model that accepts the same feature space

Preprocessing: Includes label encoding and feature scaling, saved in preprocessing/preprocessor.pkl
```