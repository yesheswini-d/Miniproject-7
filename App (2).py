import streamlit as st
import pandas as pd
import pickle

from tensorflow.keras.models import load_model


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(

    page_title="Loan Approval Prediction",

    page_icon="🏦",

    layout="centered"
)


# ---------------------------------------------------
# Load Neural Network
# ---------------------------------------------------

@st.cache_resource
def load_ml_model():

    model = load_model(
        "loan_model.keras"
    )

    return model


# ---------------------------------------------------
# Load Preprocessing Objects
# ---------------------------------------------------

@st.cache_resource
def load_preprocessing():

    with open(
        "preprocessing.pkl",
        "rb"
    ) as file:

        preprocessing = pickle.load(file)

    return preprocessing


model = load_ml_model()

preprocessing = load_preprocessing()

scaler = preprocessing["scaler"]

label_encoders = preprocessing["label_encoders"]


# ---------------------------------------------------
# Application Title
# ---------------------------------------------------

st.title(
    "🏦 Loan Approval Prediction"
)

st.write(
    "Neural Network based loan approval prediction system"
)

st.divider()


# ---------------------------------------------------
# Input Fields
# ---------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=500
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=0,
        step=500
    )


with col2:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1,
        value=150,
        step=10
    )

    loan_term = st.selectbox(
        "Loan Term (Months)",
        [120, 180, 240, 300, 360, 480],

        index=4
    )

    credit_history = st.selectbox(

        "Credit History",

        [1, 0],

        format_func=lambda x:
        "Good (1)"
        if x == 1
        else "Bad / Unavailable (0)"
    )

    property_area = st.selectbox(

        "Property Area",

        ["Urban", "Semiurban", "Rural"]
    )


st.divider()


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button(
    "🔮 Predict Loan Approval"
):

    # Create input dataframe

    input_data = pd.DataFrame([{

        "Gender": gender,

        "Married": married,

        "Education": education,

        "Self_Employed": self_employed,

        "ApplicantIncome": applicant_income,

        "CoapplicantIncome": coapplicant_income,

        "LoanAmount": loan_amount,

        "Loan_Amount_Term": loan_term,

        "Credit_History": credit_history,

        "Property_Area": property_area

    }])


    # ------------------------------------------------
    # Encode categorical values
    # ------------------------------------------------

    categorical_columns = [

        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Property_Area"

    ]


    for column in categorical_columns:

        encoder = label_encoders[column]

        input_data[column] = encoder.transform(
            input_data[column]
        )


    # ------------------------------------------------
    # Scale input
    # ------------------------------------------------

    input_scaled = scaler.transform(
        input_data
    )


    # ------------------------------------------------
    # Neural Network Prediction
    # ------------------------------------------------

    probability = model.predict(
        input_scaled,
        verbose=0
    )[0][0]


    # ------------------------------------------------
    # Display Result
    # ------------------------------------------------

    if probability >= 0.5:

        confidence = probability * 100

        st.success(
            "✅ LOAN APPROVED"
        )

        st.info(
            f"Prediction Confidence: "
            f"{confidence:.2f}%"
        )

    else:

        confidence = (1 - probability) * 100

        st.error(
            "❌ LOAN NOT APPROVED"
        )

        st.info(
            f"Prediction Confidence: "
            f"{confidence:.2f}%"
        )


# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.caption(
    "Developed as a Machine Learning mini project "
    "using Artificial Neural Network and Streamlit."
)
