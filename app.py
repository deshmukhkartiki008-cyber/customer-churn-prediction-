import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("🛒 Customer Churn Prediction System")
st.write("This app predicts whether a customer may leave the service based on shopping behavior.")

df = pd.read_csv("customer_churn_10000.csv")

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

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
col3.metric("Input Features", len(X.columns))

st.divider()

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=25)
    gender = st.selectbox("Gender", le_gender.classes_)
    annual_income = st.number_input("Annual Income", min_value=0, value=40000)
    spending_score = st.slider("Spending Score", 0, 100, 50)

with col2:
    membership_years = st.number_input("Membership Years", min_value=0, max_value=20, value=2)
    preferred_category = st.selectbox("Preferred Category", le_category.classes_)
    online_purchases = st.number_input("Online Purchases", min_value=0, value=20)
    store_visits = st.number_input("Store Visits", min_value=0, value=5)

if st.button("Predict Churn"):
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

    if prediction == 1:
        st.error("⚠️ Prediction: Customer is likely to churn")
        st.write("This customer may stop using the service. Offer discounts, loyalty points, or personalized support.")
    else:
        st.success("✅ Prediction: Customer is likely to stay")
        st.write("This customer seems satisfied and active.")

st.divider()

st.subheader("Dataset Preview")
st.dataframe(df.head())
