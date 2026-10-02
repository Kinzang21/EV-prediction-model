import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EV Purchase Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Remove excessive top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #164e63 55%,
            #0f766e 100%
        );

        padding: 2.5rem 3rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
    }

    .hero h1 {
        color: white;
        font-size: 2.7rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1.1rem;
        margin-bottom: 0;
    }

    /* Section titles */
    .section-title {
        font-size: 1.6rem;
        font-weight: 650;
        color: #0f172a;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Info card */
    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    }

    .info-card h4 {
        margin-top: 0;
        color: #0f172a;
    }

    .info-card p {
        color: #64748b;
        margin-bottom: 0;
    }

    /* Threshold badge */
    .threshold {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }

    /* Result card */
    .result-card {
        background: white;
        border-radius: 18px;
        padding: 1.8rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-label {
        color: #64748b;
        font-size: 0.95rem;
    }

    .result-value {
        color: #0f172a;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3.2rem;
        font-size: 1.05rem;
        font-weight: 650;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("ev_model.pkl")


model = load_model()


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
    <h1>🚗 Electric Vehicle Purchase Predictor</h1>
    <p>
    An interactive machine learning application for predicting whether a customer is likely to purchase an electric vehicle.
    </p>
    </div>
    """,
        unsafe_allow_html=True
)


# =========================================================
# MODEL INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Logistic Regression"
    )

with col2:
    st.metric(
        "Decision Threshold",
        "0.70"
    )

with col3:
    st.metric(
        "Validation ROC-AUC",
        "93.80%"
    )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# THRESHOLD INFORMATION
# =========================================================

st.markdown(
    """
<div class="threshold">
🎯 <strong>Prediction threshold: 70%</strong><br>
Customers with an estimated EV purchase probability of 70% or higher are classified as likely EV buyers.
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)


left, right = st.columns(2)


# =========================================================
# LEFT COLUMN
# =========================================================

with left:

    st.markdown(
        '<div class="info-card"><h4>Personal & Financial</h4>'
        '<p>Basic customer information</p></div>',
        unsafe_allow_html=True
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    annual_income = st.number_input(
        "Annual Income (USD)",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    city_type = st.selectbox(
        "City Type",
        [
            "Urban",
            "Suburban",
            "Rural"
        ]
    )

    current_car_type = st.selectbox(
        "Current Car Type",
        [
            "Petrol",
            "Diesel",
            "Hybrid",
            "Electric"
        ]
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with right:

    st.markdown(
        '<div class="info-card"><h4>EV & Mobility Factors</h4>'
        '<p>Transportation and EV-related information</p></div>',
        unsafe_allow_html=True
    )

    daily_commute = st.number_input(
        "Daily Commute Distance (km)",
        min_value=0.0,
        value=20.0,
        step=1.0
    )

    cars_owned = st.number_input(
        "Number of Cars Owned",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    charging_home = st.number_input(
        "Charging Stations Near Home",
        min_value=0,
        value=1,
        step=1
    )

    charging_work = st.number_input(
        "Charging Stations Near Work",
        min_value=0,
        value=1,
        step=1
    )

    environmental_concern = st.number_input(
        "Environmental Concern Level",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )


# =========================================================
# ADDITIONAL CATEGORICAL VARIABLES
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">⚡ EV Readiness</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    home_charging = st.selectbox(
        "Home Charging Possible",
        [
            "Yes",
            "No"
        ]
    )


with col2:

    subsidy = st.selectbox(
        "Subsidy Available",
        [
            "Yes",
            "No"
        ]
    )


with col3:

    range_anxiety = st.selectbox(
        "Range Anxiety Level",
        [
            "Low",
            "Medium",
            "High"
        ]
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔮 Predict EV Purchase",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_data = pd.DataFrame({

        "Gender": [gender],

        "City_Type": [city_type],

        "Current_Car_Type": [current_car_type],

        "Home_Charging_Possible": [home_charging],

        "Subsidy_Available": [subsidy],

        "Range_Anxiety_Level": [range_anxiety],

        "Age": [age],

        "Annual_Income_USD": [annual_income],

        "Daily_Commute_km": [daily_commute],

        "Number_of_Cars_Owned": [cars_owned],

        "Charging_Stations_Near_Home": [charging_home],

        "Charging_Stations_Near_Work": [charging_work],

        "Environmental_Concern_Level": [environmental_concern]
    })


    # Probability of Yes
    probability = model.predict_proba(
        input_data
    )[0, 1]


    # Selected threshold
    threshold = 0.70


    # Final prediction
    prediction = (
        "Yes"
        if probability >= threshold
        else "No"
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "EV Purchase Probability",
            f"{probability:.2%}"
        )


    with result_col2:

        st.metric(
            "Predicted Outcome",
            prediction
        )


    if prediction == "Yes":

        st.success(
            "🚗 The model predicts that this customer "
            "is likely to purchase an electric vehicle."
        )

    else:

        st.info(
            "The model predicts that this customer "
            "is unlikely to purchase an electric vehicle."
        )


    # Probability bar
    st.progress(
        float(probability)
    )


    if probability >= threshold:

        st.caption(
            f"Probability ({probability:.2%}) is above "
            f"the 70% decision threshold."
        )

    else:

        st.caption(
            f"Probability ({probability:.2%}) is below "
            f"the 70% decision threshold."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
Built with Python • Scikit-learn • Pandas • Streamlit<br>
EV Purchase Prediction — Machine Learning Project
</div>
""",
    unsafe_allow_html=True
)