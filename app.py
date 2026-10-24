import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection", page_icon="🔍")

@st.cache_resource
def load_model():
    return joblib.load("models/xgb_fraud.joblib")

model = load_model()

FEATURES = ["type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "errorOrig", "errorDest"]

st.title("Mobile Money Fraud Detection")
st.write("Enter transaction details to estimate the probability of fraud. Trained on the PaySim synthetic dataset.")

tx_type = st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)

col1, col2 = st.columns(2)
with col1:
    old_org = st.number_input("Sender balance before", min_value=0.0, value=1000.0)
    new_org = st.number_input("Sender balance after", min_value=0.0, value=0.0)
with col2:
    old_dest = st.number_input("Receiver balance before", min_value=0.0, value=0.0)
    new_dest = st.number_input("Receiver balance after", min_value=0.0, value=1000.0)

threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)

if st.button("Check transaction"):
    row = pd.DataFrame([{
        "type": 1 if tx_type == "TRANSFER" else 0,
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "errorOrig": old_org - amount - new_org,
        "errorDest": old_dest + amount - new_dest,
    }])[FEATURES]

    prob = float(model.predict_proba(row)[0, 1])
    st.metric("Fraud probability", f"{prob:.1%}")
    if prob >= threshold:
        st.error("Flagged as likely fraud")
    else:
        st.success("Looks legitimate")