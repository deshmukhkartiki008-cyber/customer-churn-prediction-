import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.markdown("""
<style>
.stButton>button {
    background-color: #2E8B57;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🛒 Customer Churn Prediction System")
st.write("AI-based system to predict whether a customer may leave the service.")

st.sidebar.title("About Project")
st.sidebar.info(
    "This Customer Churn Prediction app uses machine learning to analyze customer "
    "behavior and predict whether a customer is likely to churn or stay."
)

@st.cache_data
def load_data():
    return pd.read_csv("customer_churn_10000.csv")

df = load_data()

df = df.drop("customer_id", axis=1)

df["churn"] = np.where(
    (df["spending_score"] < 40) |
    (df["membership_years"] < 2) |
    (df["online_purchases"] < 20) |
    (df["store_visits"] < 5),
    1,
    0
)

le_gender = LabelEncoder()
le_category = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"])
df["preferred_category"] = le_category.fit_transform(df["preferred_category"])

X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

accuracy = accuracy_score(y_test, model.predict(X_test))

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
col3.metric("Features Used", len(X.columns))

st.divider()

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 25)
    gender = st.selectbox("Gender", le_gender.classes_)
    annual_income = st.number_input("Annual Income", min_value=0, value=40000)
    spending_score = st.slider("Spending Score", 0, 100, 50)

with col2:
    membership_years = st.slider("Membership Years", 0, 20, 2)
    preferred_category = st.selectbox("Preferred Category", le_category.classes_)
    online_purchases = st.number_input("Online Purchases", min_value=0, value=20)
    store_visits = st.number_input("Store Visits", min_value=0, value=5)

st.divider()

if st.button("Predict Customer Churn"):
    gender_encoded = le_gender.transform([gender])[0]
    category_encoded = le_category.transform([preferred_category])[0]

    input_data = pd.DataFrame([[
        age,
        gender_encoded,
        annual_income,
        spending_score,
        membership_years,
        category_encoded,
        online_purchases,
        store_visits
    ]], columns=X.columns)

    prediction = model.predict(input_data)[0]
    churn_probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if churn_probability >= 0.70:
        risk_level = "High Risk"
    elif churn_probability >= 0.40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    if prediction == 1:
        st.error(f"Customer is likely to churn")
        st.warning(f"Risk Level: {risk_level}")
        st.write("Suggestion: Offer discounts, loyalty rewards, personalized offers, or customer support.")
    else:
        st.success("Customer is likely to stay")
        st.info(f"Risk Level: {risk_level}")
        st.write("Suggestion: Maintain good service quality and continue personalized engagement.")

    st.metric("Churn Probability", f"{churn_probability * 100:.2f}%")

    report = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Annual Income": [annual_income],
        "Spending Score": [spending_score],
        "Membership Years": [membership_years],
        "Preferred Category": [preferred_category],
        "Online Purchases": [online_purchases],
        "Store Visits": [store_visits],
        "Prediction": ["Churn" if prediction == 1 else "Stay"],
        "Risk Level": [risk_level],
        "Churn Probability": [f"{churn_probability * 100:.2f}%"]
    })

    csv = report.to_csv(index=False)

    st.download_button(
        label="Download Prediction Report",
        data=csv,
        file_name="customer_churn_prediction_report.csv",
        mime="text/csv"
    )

st.divider()

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

st.caption("Developed using Streamlit and Machine Learning")
