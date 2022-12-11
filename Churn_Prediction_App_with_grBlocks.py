# ----- Load base libraries and packages
import gradio as gr

import numpy as np
import pandas as pd
import re

import os
import pickle

import xgboost as xgb
from xgboost import XGBClassifier



# ----- Useful lists
expected_inputs = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
columns_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
categoricals = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]


# ----- Helper Functions
# Function to load ML toolkit
def load_ml_toolkit(file_path=r"gradio_src\Gradio_App_toolkit"):
    with open(file_path, "rb") as file:
        loaded_toolkit = pickle.load(file)
    return loaded_toolkit


# Importing the toolkit
loaded_toolkit = load_ml_toolkit(r"gradio_src\Gradio_App_toolkit")
encoder = loaded_toolkit["encoder"]
scaler = loaded_toolkit["scaler"]

# Import the model
model = XGBClassifier()
model.load_model(r"gradio_src\xgb_model.json")


# Function to process inputs and return prediction
def process_and_predict(*args, encoder=encoder, scaler=scaler, model=model):
    # Convert inputs into a DataFrame
    input_data = pd.DataFrame([args], columns=expected_inputs)

    # Encode the categorical columns
    encoded_categoricals = encoder.transform(input_data[categoricals])
    encoded_categoricals = pd.DataFrame(encoded_categoricals, columns=encoder.get_feature_names_out().tolist())
    df_processed = input_data.join(encoded_categoricals)
    df_processed.drop(columns=categoricals, inplace=True)

    # Scale the numeric columns
    df_processed[columns_to_scale] = scaler.transform(df_processed[columns_to_scale])

    # Restrict column name characters to alphanumerics
    df_processed.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x), inplace=True)

    # Making the prediction
    model_output = model.predict(df_processed)
    return {"Prediction: CHURN": float(model_output[0]), "Prediction: STAY": 1-float(model_output[0])}


# Define some variable limits and lists of options
max_tenure = 1.61803398875 * 72 # Maximum value from the training data
max_monthly_charges = 1.61803398875 * 200 # Maximum value from the training data
max_total_charges = 1.61803398875 * 8684.8 # Maximum value from the training data
yes_or_no = ["Yes", "No"] # For the columns with Yes and No as options
internet_service_choices = ["Yes", "No", "No internet service"] # For the services that depend on internet service usage



# ----- App Interface
with gr.Blocks() as turn_on_the_gradio:
    gr.Markdown("# Telecom Customer Churn Prediction")
    gr.Markdown("""This app uses a machine learning model to predict whether or not a customer will churn based on inputs made by you, the user. The (XGBoost) model was trained and built based on the Telecom Churn Dataset""")
    
    
    # Phase 1: Receiving Inputs
    gr.Markdown("**Demographic Data**")
    with gr.Row():
        gender = gr.Dropdown(label="Gender", choices=["Female", "Male"], value="Female")
        SeniorCitizen = gr.Radio(label="Senior Citizen", choices=yes_or_no, value="No")
        Partner = gr.Radio(label="Partner", choices=yes_or_no, value="No")
        Dependents = gr.Radio(label="Dependents", choices=yes_or_no, value="No")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Contract and Tenure Data**")
            Contract = gr.Dropdown(label="Contract", choices=["Month-to-month", "One year", "Two year"], value="Month-to-month")
            tenure = gr.Slider(label="Tenure (months)", minimum=1, step=1, interactive=True, value=1, maximum= max_tenure)
        with gr.Column():
            gr.Markdown("**Phone Service Usage**")
            PhoneService = gr.Radio(label="Phone Service", choices=yes_or_no, value="Yes")
            MultipleLines = gr.Dropdown(label="Multiple Lines", choices=["Yes", "No", "No phone service"], value="No")

    # Internet Service Usage
    gr.Markdown("**Internet Service Usage**")
    with gr.Row():
        InternetService = gr.Dropdown(label="Internet Service", choices=["DSL", "Fiber optic", "No"], value="Fiber optic")
        OnlineSecurity = gr.Dropdown(label="Online Security", choices=internet_service_choices, value="No")
        OnlineBackup = gr.Dropdown(label="Online Backup", choices=internet_service_choices, value="No")
        DeviceProtection = gr.Dropdown(label="Device Protection", choices=internet_service_choices, value="No")
        TechSupport = gr.Dropdown(label="Tech Support", choices=internet_service_choices, value="No")
        StreamingTV = gr.Dropdown(label="TV Streaming", choices=internet_service_choices, value="No")
        StreamingMovies = gr.Dropdown(label="Movie Streaming", choices=internet_service_choices, value="No")

    # Billing and Payment
    gr.Markdown("**Charges (USD), Billing and Payment**")
    with gr.Row():
        MonthlyCharges = gr.Slider(label="Monthly Charges", step=0.05, maximum=max_monthly_charges)
        TotalCharges = gr.Slider(label="Total Charges", step=0.05, maximum=max_total_charges)
        PaperlessBilling = gr.Radio(label="Paperless Billing", choices=yes_or_no, value="Yes")
        PaymentMethod = gr.Dropdown(label="Payment Method", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], value="Electronic check")

    # Output Prediction
    output = gr.Label("Awaiting Submission...")
    submit_button = gr.Button("Submit")
    
    submit_button.click(fn = process_and_predict,
                        outputs = output,
                        inputs = [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges])

turn_on_the_gradio.launch(favicon_path=r"gradio_src\app_thumbnail.png",
                          inbrowser= True,) #share=True)