
import streamlit as st
import joblib
import numpy as np

# Set up page configurations
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📈",
    layout="centered"
)

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('model.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model 'model.sav': {e}")
    st.stop()

# App title and description
st.title("📈 Advertising Sales Predictor")
st.write("""
This app predicts product **Sales (in thousands of units)** based on the advertising budgets allocated to **TV**, **Radio**, and **Newspaper** channels.
""")

st.divider()

# Layout inputs
st.subheader("💵 Advertising Budget Inputs (in $1,000s)")
col1, col2, col3 = st.columns(3)

with col1:
    tv = st.number_input("TV Budget", min_value=0.0, max_value=500.0, value=20.0, step=1.0)

with col2:
    radio = st.number_input("Radio Budget", min_value=0.0, max_value=500.0, value=12.0, step=1.0)

with col3:
    newspaper = st.number_input("Newspaper Budget", min_value=0.0, max_value=500.0, value=5.0, step=1.0)

st.divider()

# Prediction Trigger
if st.button("🔮 Predict Sales", use_container_width=True):
    # Prepare inputs as 2D array (matching training format)
    features = np.array([[tv, radio, newspaper]])
    
    # Predict
    prediction = model.predict(features)[0]
    
    # Display Result
    st.success(f"### Estimated Sales: **{prediction:.2f}k** units")
