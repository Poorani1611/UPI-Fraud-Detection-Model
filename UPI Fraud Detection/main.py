# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FraudPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_columns_ = None  # will store columns after fit

    def fit(self, X):
        X_transformed = self._transform_features(X.copy())
        self.feature_columns_ = X_transformed.columns  # store columns
        return self

    def transform(self, X):
        X = self._transform_features(X.copy())

        # Only align columns if feature_columns_ is set (after fit)
        if self.feature_columns_ is not None:
            # add missing columns as 0
            for col in self.feature_columns_:
                if col not in X.columns:
                    X[col] = 0
            # keep only training columns
            X = X[self.feature_columns_]

        return X

    def _transform_features(self, X):
        # Drop unused columns
        X = X.drop(['isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1, errors='ignore')

        # Feature engineering
        X['hour'] = X['step'] % 24
        X['is_night'] = (X['hour'] < 6).astype(int)
        X['amount_ratio'] = X['amount'] / (X['oldbalanceOrg'] + 1)
        X['sender_balance_change'] = X['oldbalanceOrg'] - X['newbalanceOrig']
        X['receiver_balance_change'] = X['newbalanceDest'] - X['oldbalanceDest']
        X['orig_balance_zero'] = (X['oldbalanceOrg'] == 0).astype(int)
        X['dest_balance_zero'] = (X['oldbalanceDest'] == 0).astype(int)

        # One-hot encode type
        X = pd.get_dummies(X, columns=['type'], drop_first=True)
        if 'type_TRANSFER' not in X.columns:
            X['type_TRANSFER'] = 0

        return X



# Load trained model
# Load pipeline
with open("pipeline.pkl", "rb") as f:
    preprocessor, scaler, model = pickle.load(f)


st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details below to predict fraud.")

# Input fields
step = st.number_input("Step (time unit)", min_value=0, value=1)
amount = st.number_input("Amount", min_value=0.0, value=100.0)
oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, value=1000.0)
newbalanceOrig = st.number_input("New Balance Origin", min_value=-9999999999.0, value=900.0)
oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=100.0)
type_transfer = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"])

# Prepare raw input (no need to compute features manually!)
input_data = pd.DataFrame([{
    'step': step,
    'type': type_transfer,
    'amount': amount,
    'nameOrig': 'C123456789',
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'nameDest': 'M987654321',
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest,
    'isFlaggedFraud': 0
}])

# Predict
if st.button("Predict Fraud"):
    input_transformed = preprocessor.transform(input_data)
    input_scaled = scaler.transform(input_transformed)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"Fraud probability: {prob:.2%}")
    threshold = 0.2
    if prob >= threshold:
        st.error(f"🚨 Fraudulent Transaction Detected (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {prob:.2%})")
