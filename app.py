import streamlit as st
import pandas as pd
import altair as alt
from utils.predict import predict, model_information, validate_input

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide",
)

# Header Section
st.title("🚢 Titanic Survival Prediction Dashboard")
st.markdown(
    "**Objective:** Predict whether a passenger would have survived the Titanic disaster based on demographic and ticketing data."
)

# Sidebar with Model Information
st.sidebar.header("📊 Model Configuration")
info = model_information()
st.sidebar.markdown(f"**Algorithm:** {info['algorithm']}")
st.sidebar.markdown(f"**Tuning Method:** {info['tuning_method']}")
st.sidebar.markdown(f"**Test Accuracy:** `{info['test_accuracy'] * 100:.2f}%`")
st.sidebar.markdown(f"**CV ROC-AUC:** `{info['cv_roc_auc']:.4f}`")

st.sidebar.subheader("Hyperparameters")
for param, val in info["best_hyperparameters"].items():
    st.sidebar.write(f"- **{param}:** `{val}`")

st.sidebar.subheader("Developer Info")
st.sidebar.write("Senior Machine Learning Engineer Placeholder")

# Main Page Inputs
st.subheader("📋 Enter Passenger Information")

# Arrange inputs in three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Passenger Details")
    sex_label = st.selectbox("Sex", options=["Male", "Female"], index=0)
    sex = 0 if sex_label == "Male" else 1
    
    age = st.slider("Age (in years)", min_value=0, max_value=100, value=25, step=1)

with col2:
    st.markdown("### Travel Details")
    pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3], index=2, format_func=lambda x: f"Class {x}")
    
    fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=512.0, value=15.0, step=1.0)
    
    embarked_label = st.selectbox(
        "Port of Embarkation", 
        options=["Cherbourg (0.0)", "Queenstown (1.0)", "Southampton (2.0)"],
        index=2
    )
    embarked = float(embarked_label.split("(")[1].replace(")", ""))

with col3:
    st.markdown("### Family Details")
    sibsp = st.number_input("Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parents / Children Aboard (Parch)", min_value=0, max_value=10, value=0, step=1)

# Run prediction
input_data = {
    "Age": age,
    "Fare": fare,
    "Sex": sex,
    "sibsp": sibsp,
    "Parch": parch,
    "Pclass": pclass,
    "Embarked": embarked
}

st.markdown("---")

if st.button("Predict Survival Status", type="primary"):
    # Validate
    is_valid, msg = validate_input(input_data)
    if not is_valid:
        st.error(f"❌ Input validation failed: {msg}")
    else:
        try:
            prediction, class_id, confidence, prob_dict = predict(input_data)
            
            # Show prediction card
            st.subheader("🔮 Prediction Outcome")
            
            if class_id == 1:
                st.success(f"🎉 **{prediction}**")
                st.markdown(
                    f"The model predicts with **{confidence * 100:.2f}%** confidence that this passenger **survived**."
                )
            else:
                st.error(f"💀 **{prediction}**")
                st.markdown(
                    f"The model predicts with **{confidence * 100:.2f}%** confidence that this passenger **did not survive**."
                )
            
            # Visualization
            st.markdown("### Probability Distribution")
            prob_df = pd.DataFrame({
                "Outcome": list(prob_dict.keys()),
                "Probability": list(prob_dict.values())
            })
            
            chart = alt.Chart(prob_df).mark_bar().encode(
                x='Probability:Q',
                y=alt.Y('Outcome:N', sort='-x'),
                color=alt.Color('Outcome:N', scale=alt.Scale(domain=['Survived', 'Did Not Survive'], range=['#2ca02c', '#d62728']))
            ).properties(height=150)
            
            st.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
