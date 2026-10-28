# 🏦 Loan Approval Prediction Using Neural Network

## Introduction

Loan Approval Prediction is a Machine Learning mini project
that uses an Artificial Neural Network to predict whether a
loan application is likely to be approved or not.

The project also uses Streamlit to provide an interactive
web interface.

## Objectives

- Predict loan approval using a Neural Network.
- Perform data preprocessing.
- Encode categorical data.
- Scale numerical features.
- Train an Artificial Neural Network.
- Build a web application using Streamlit.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- Streamlit
- Pickle

## Neural Network Architecture

Input Layer
     ↓
Dense Layer - 32 Neurons
     ↓
Dropout Layer
     ↓
Dense Layer - 16 Neurons
     ↓
Dense Layer - 8 Neurons
     ↓
Output Layer - 1 Neuron
     ↓
Sigmoid Activation

## Features

The model uses:

- Gender
- Married
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

## Output

The Neural Network produces a probability.

If probability >= 0.5:

Loan Approved

If probability < 0.5:

Loan Not Approved

## Project Structure

```text
Loan_Approval_Neural_Network/

├── train_model.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── loan_model.keras
└── preprocessing.pkl
