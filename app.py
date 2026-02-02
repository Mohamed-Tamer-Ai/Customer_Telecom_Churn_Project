from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ======================================
# Load dataset (Dashboard + ML separated)
# ======================================
@st.cache_data
def load_data():
    df = pd.read_csv("./cleaned_dataset.csv")

    if 'customerid' in df.columns:
        df.drop(columns=['customerid'], inplace=True)

    # -------- Dashboard Data (RAW) --------
    df_dashboard = df.copy()

    # -------- ML Data --------
    df_ml = df.copy()
    df_ml['churn'] = df_ml['churn'].map({'Yes': 1, 'No': 0})

    categorical_cols = [
        'gender', 'partner', 'dependents', 'phoneservice', 'multiplelines',
        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
        'streamingtv', 'streamingmovies', 'paperlessbilling',
        'internetservice', 'contract', 'paymentmethod'
    ]

    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        encoders[col] = le

    scaler = StandardScaler()
    num_cols = ["tenure", "monthlycharges", "totalcharges"]
    df_ml[num_cols] = scaler.fit_transform(df_ml[num_cols])

    X = df_ml.drop('churn', axis=1)
    y = df_ml['churn']

    return df_dashboard, X, y, encoders, scaler


# ================================
# Load model
# ================================
@st.cache_resource
def load_model():
    return joblib.load("./logistic_regression_model.pkl")


# ================================
# Analysis Page
# ================================
def analysis_page(df):
    st.title("📊 Telecom Churn Analysis Dashboard")

    col1, col2 = st.columns(2)
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{(df['churn'] == 'Yes').mean()*100:.2f}%")

    st.divider()

    st.subheader("Payment Method Distribution")
    payment_counts = df['paymentmethod'].value_counts().reset_index()
    payment_counts.columns = ['Payment Method', 'Count']
    st.plotly_chart(
        px.bar(payment_counts, x='Payment Method', y='Count'),
        use_container_width=True
    )

    st.subheader("Contract Types")
    st.plotly_chart(
        px.pie(df, names='contract'),
        use_container_width=True
    )

    st.subheader("Churn Distribution")
    st.plotly_chart(
        px.pie(df, names='churn'),
        use_container_width=True
    )

    st.subheader("Contract vs Churn")
    st.plotly_chart(
        px.histogram(df, x='contract', color='churn', barmode='group'),
        use_container_width=True
    )

    st.subheader("Monthly Charges Distribution")
    st.plotly_chart(
        px.histogram(df, x='monthlycharges', nbins=30),
        use_container_width=True
    )

    cols = [
        'techsupport', 'streamingtv', 'streamingmovies',
        'paperlessbilling', 'paymentmethod'
    ]

    for col in cols:
        st.subheader(f"{col} vs Churn")
        st.plotly_chart(
            px.histogram(df, x=col, color='churn', barmode='group'),
            use_container_width=True
        )


# ================================
# Prediction Page
# ================================
def prediction_page():
    st.title("📉 Customer Churn Prediction")

    df, X, y, encoders, scaler = load_data()
    model = load_model()

    gender = st.selectbox("Gender", encoders['gender'].classes_)
    seniorcitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", encoders['partner'].classes_)
    dependents = st.selectbox("Dependents", encoders['dependents'].classes_)
    tenure = st.number_input("Tenure (Months)", 0, 100, 10)
    phoneservice = st.selectbox("Phone Service", encoders['phoneservice'].classes_)
    multiplelines = st.selectbox("Multiple Lines", encoders['multiplelines'].classes_)
    onlinesecurity = st.selectbox("Online Security", encoders['onlinesecurity'].classes_)
    onlinebackup = st.selectbox("Online Backup", encoders['onlinebackup'].classes_)
    deviceprotection = st.selectbox("Device Protection", encoders['deviceprotection'].classes_)
    techsupport = st.selectbox("Tech Support", encoders['techsupport'].classes_)
    streamingtv = st.selectbox("Streaming TV", encoders['streamingtv'].classes_)
    streamingmovies = st.selectbox("Streaming Movies", encoders['streamingmovies'].classes_)
    paperlessbilling = st.selectbox("Paperless Billing", encoders['paperlessbilling'].classes_)
    monthlycharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    totalcharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
    internetservice = st.selectbox("Internet Service", encoders['internetservice'].classes_)
    contract = st.selectbox("Contract", encoders['contract'].classes_)
    paymentmethod = st.selectbox("Payment Method", encoders['paymentmethod'].classes_)

    input_data = {
        'gender': encoders['gender'].transform([gender])[0],
        'seniorcitizen': 1 if seniorcitizen == "Yes" else 0,
        'partner': encoders['partner'].transform([partner])[0],
        'dependents': encoders['dependents'].transform([dependents])[0],
        'tenure': tenure,
        'phoneservice': encoders['phoneservice'].transform([phoneservice])[0],
        'multiplelines': encoders['multiplelines'].transform([multiplelines])[0],
        'onlinesecurity': encoders['onlinesecurity'].transform([onlinesecurity])[0],
        'onlinebackup': encoders['onlinebackup'].transform([onlinebackup])[0],
        'deviceprotection': encoders['deviceprotection'].transform([deviceprotection])[0],
        'techsupport': encoders['techsupport'].transform([techsupport])[0],
        'streamingtv': encoders['streamingtv'].transform([streamingtv])[0],
        'streamingmovies': encoders['streamingmovies'].transform([streamingmovies])[0],
        'paperlessbilling': encoders['paperlessbilling'].transform([paperlessbilling])[0],
        'monthlycharges': monthlycharges,
        'totalcharges': totalcharges,
        'internetservice': encoders['internetservice'].transform([internetservice])[0],
        'contract': encoders['contract'].transform([contract])[0],
        'paymentmethod': encoders['paymentmethod'].transform([paymentmethod])[0],
    }

    df_input = pd.DataFrame([input_data])
    df_input[['tenure','monthlycharges','totalcharges']] = scaler.transform(
        df_input[['tenure','monthlycharges','totalcharges']]
    )

    if st.button("Predict Churn"):
        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0][1]

        if pred == 1:
            st.error(f"❌ Customer is likely to churn (Probability: {proba:.2%})")
        else:
            st.success(f"✅ Customer is not likely to churn (Probability: {proba:.2%})")


# ================================
# Main
# ================================
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Analysis", "Prediction"])

    df_dashboard, _, _, _, _ = load_data()

    if page == "Analysis":
        analysis_page(df_dashboard)
    else:
        prediction_page()


if __name__ == "__main__":
    main()