# Importing the base libraries and packages
import pickle
import os
import pandas as pd
import gradio as gr
import re

import xgboost as xgb


# ----- Useful Lists
expected_inputs = ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract", "tenure", "MonthlyCharges", "TotalCharges", "PaymentMethod", "PhoneService",
                   "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]
categoricals = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
columns_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]



# ----- Helper Functions and Imports
# Function to load ML toolkit
def load_ml_toolkit(filepath="gradio_src\Gradio_App_toolkit"):
    with open(filepath, "rb") as file:
        loaded_toolkit = pickle.load(file)
    return loaded_toolkit


# Importing the toolkit
loaded_toolkit = load_ml_toolkit()
encoder = loaded_toolkit["encoder"]
scaler = loaded_toolkit["scaler"]
model = loaded_toolkit["model"]

# Function to predict
def predict(*args, encoder = encoder, scaler = scaler, model = model):
    input_data = pd.DataFrame([args], columns= expected_inputs)
    input_data.fillna(0, inplace= True)
    
    encoder = loaded_toolkit["encoder"]
    scaler = loaded_toolkit["scaler"]
    model = loaded_toolkit["model"]

    # Encoding the categorical columns
    encoded_categoricals = encoder.transform(input_data[categoricals])
    encoded_categoricals = pd.DataFrame(encoded_categoricals, columns= encoder.get_feature_names_out().tolist())
    
    # Adding the categorical columns to the input dataframe
    df_encoded = input_data.join(encoded_categoricals)
    df_encoded.drop(columns= categoricals, inplace=True)
    
    # Scaling the numeric columns
    df_encoded[columns_to_scale] = scaler.transform(df_encoded[columns_to_scale])
    
    # Restricting column names to alpha-numeric characters
    df_processed = df_encoded.rename(columns= lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    
    # Making the prediction
    model_output = model.predict(df_processed)
    return float(model_output[0])



# ----- App Interface
with gr.Blocks() as demo:
    gr.Markdown("# Telecom Customer Churn Prediction")

    # ----- Phase 1: Receiving Inputs
    gr.Markdown("**Demographic Data**")
    with gr.Row():
        gender = gr.Dropdown(label="Gender", choices=["Male", "Female"])
        SeniorCitizen = gr.Radio(label="Senior Citizen", choices=["Yes", "No"])
        Partner = gr.Radio(label="Partner", choices=["Yes", "No"])
        Dependents = gr.Radio(label="Dependents", choices=["Yes", "No"])

    gr.Markdown("**Service Length and Charges (USD)**")
    with gr.Row():
        Contract = gr.Dropdown(label="Contract", choices=["Month-to-month", "One year", "Two year"])
        tenure = gr.Slider(label="Tenure (months)", minimum=1, step=1, interactive=True)
        MonthlyCharges = gr.Slider(label="Monthly Charges", step=0.05)
        TotalCharges = gr.Slider(label="Total Charges", step=0.05)

    # Phone Service Usage
    gr.Markdown("**Phone Service Usage**")
    with gr.Row():
        PhoneService = gr.Radio(label="Phone Service", choices=["Yes", "No"])
        MultipleLines = gr.Dropdown(label="Multiple Lines", choices=[
                                    "Yes", "No", "No phone service"])

    # Internet Service Usage
    gr.Markdown("**Internet Service Usage**")
    with gr.Row():
        InternetService = gr.Dropdown(label="Internet Service", choices=["DSL", "Fiber Optic", "No"])
        OnlineSecurity = gr.Dropdown(label="Online Security", choices=["Yes", "No", "No phone service"])
        OnlineBackup = gr.Dropdown(label="Online Backup", choices=["Yes", "No", "No phone service"])
        DeviceProtection = gr.Dropdown(label="Device Protection", choices=["Yes", "No", "No phone service"])
        TechSupport = gr.Dropdown(label="Tech Support", choices=["Yes", "No", "No phone service"])
        StreamingTV = gr.Dropdown(label="TV Streaming", choices=["Yes", "No", "No phone service"])
        StreamingMovies = gr.Dropdown(label="Movie Streaming", choices=["Yes", "No", "No phone service"])

    # Billing and Payment
    gr.Markdown("**Billing and Payment**")
    with gr.Row():
        PaperlessBilling = gr.Radio(
            label="Paperless Billing", choices=["Yes", "No"])
        PaymentMethod = gr.Dropdown(label="Payment Method", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    # Output Prediction
    output = gr.Number(label="Prediction")
    submit_button = gr.Button("Submit")
    
    submit_button.click(fn= predict,
                        outputs= output,
                        inputs= [gender, SeniorCitizen, Partner, Dependents, Contract, tenure, MonthlyCharges, TotalCharges, PaymentMethod, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling],
                        )

demo.launch()
