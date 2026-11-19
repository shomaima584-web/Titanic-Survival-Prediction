import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")


model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Enter the passenger details below and the machine learning model "
    "will predict whether the passenger would have survived the Titanic disaster."
)

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket Class", [1, 2, 3], help="1 = Upper, 2 = Middle, 3 = Lower")
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    embarked = st.selectbox(
        "Port of Embarkation",
        ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
    )

with col2:
    sibsp = st.number_input("Siblings / Spouses aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents / Children aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

if st.button("Predict", type="primary"):
    family_size = sibsp + parch + 1

    data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": 1 if sex == "Female" else 0,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "FamilySize": family_size,
        "IsAlone": 1 if family_size == 1 else 0,
        "Embarked_Q": 1 if "(Q)" in embarked else 0,
        "Embarked_S": 1 if "(S)" in embarked else 0,
    }])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.success(f"✅ Survived (survival probability: {probability:.1%})")
    else:
        st.error(f"❌ Did not survive (survival probability: {probability:.1%})")

st.caption("Built with Python, scikit-learn and Streamlit.")
