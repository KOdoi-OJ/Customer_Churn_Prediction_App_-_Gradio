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
expected_inputs = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                   "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
columns_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
categoricals = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]



# ----- Helper Functions
# Function to load ML toolkit
def load_ml_toolkit(file_path=r"gradio_src\Gradio_App_toolkit"):
    """
    This function loads the ML items into this file. It takes the path to the ML items to load it.

    Args:
        file_path (regexp, optional): It receives the file path to the ML items, but defaults to the "gradio_src" folder in the repository. The full default relative path is r"gradio_src\Gradio_App_toolkit".

    Returns:
        file: It returns the pickle file (which in this case contains the Machine Learning items.)
    """
    
    with open(file_path, "rb") as file:
        loaded_toolkit = pickle.load(file)
    return loaded_toolkit


# Importing the toolkit
loaded_toolkit = load_ml_toolkit(r"gradio_src\Gradio_App_toolkit")
encoder = loaded_toolkit["encoder"]
scaler = loaded_toolkit["scaler"]

# Importing the model
model = XGBClassifier()
model.load_model(r"gradio_src\xgb_model.json")


# Function to process inputs and return prediction
def process_and_predict(*args, encoder=encoder, scaler=scaler, model=model):
    """
    This function processes the inputs and returns the predicted churn status of the customer.
    It receives the user inputs, the encoder, scaler and model. The inputs are then put through the same process as was done during modelling, i.e. encode categoricals,

    Args:
        encoder (OneHotEncoder, optional): It is the encoder used to encode the categorical features before training the model, and should be loaded either as part of the ML items or as a standalone item. Defaults to encoder, which comes with the ML Items dictionary.
        scaler (MinMaxScaler, optional): It is the scaler (MinMaxScaler) used to scale the numeric features before training the model, and should be loaded either as part of the ML Items or as a standalone item. Defaults to scaler, which comes with the ML Items dictionary.
        model (XGBoost, optional): This is the model that was trained and is to be used for the prediction. Since XGBoost seems to have issues with Pickle, import as a standalone. It defaults to "model", as loaded.

    Returns:
        Prediction (label): Returns the label of the predicted class, i.e. one of whether the given customer will churn or not.
    """
    
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
max_tenure = 1.61803398875 * 72 # Applied the Golden Ratio to the maximum value from the training data to leave room for increased customer tenures while still ensuring a limit on the possible inputs. 
max_monthly_charges = 1.61803398875 * 200 # Applied the Golden Ratio to the maximum amount of monthly charges from the training data to leave room for increased amounts while still ensuring a limit on the possible inputs. 
max_total_charges = 1.61803398875 * 8684.8 # Applied the Golden Ratio to the maximum amount of total charges from the training data to leave room for increased amounts while still ensuring a limit on the possible inputs. 
yes_or_no = ["Yes", "No"] # To be used for the variables whose possible options are "Yes" or "No".
internet_service_choices = ["Yes", "No", "No internet service"] # A variable for the choices available for the "Internet Service" variable



# ----- App Interface
# Inputs
gender = gr.Dropdown(label="Gender", choices=["Female", "Male"], value="Female") # Whether the customer is a male or a female
SeniorCitizen = gr.Radio(label="Senior Citizen", choices=yes_or_no, value="No") # Whether a customer is a senior citizen or not
Partner = gr.Radio(label="Partner", choices=yes_or_no, value="No") # Whether the customer has a partner or not
Dependents = gr.Radio(label="Dependents", choices=yes_or_no, value="No") # Whether the customer has dependents or not

tenure = gr.Slider(label="Tenure (months)", minimum=1, step=1, interactive=True, value=1, maximum= max_tenure) # Number of months the customer has stayed with the company

PhoneService = gr.Radio(label="Phone Service", choices=yes_or_no, value="Yes") # Whether the customer has a phone service or not
MultipleLines = gr.Dropdown(label="Multiple Lines", choices=["Yes", "No", "No phone service"], value="No") # Whether the customer has multiple lines or not

InternetService = gr.Dropdown(label="Internet Service", choices=["DSL", "Fiber optic", "No"], value="Fiber optic") # Customer's internet service provider
OnlineSecurity = gr.Dropdown(label="Online Security", choices=internet_service_choices, value="No") # Whether the customer has online security or not
OnlineBackup = gr.Dropdown(label="Online Backup", choices=internet_service_choices, value="No") # Whether the customer has online backup or not
DeviceProtection = gr.Dropdown(label="Device Protection", choices=internet_service_choices, value="No") # Whether the customer has device protection or not
TechSupport = gr.Dropdown(label="Tech Support", choices=internet_service_choices, value="No") # Whether the customer has tech support or not
StreamingTV = gr.Dropdown(label="TV Streaming", choices=internet_service_choices, value="No") # Whether the customer has streaming TV or not
StreamingMovies = gr.Dropdown(label="Movie Streaming", choices=internet_service_choices, value="No") # Whether the customer has streaming movies or not

Contract = gr.Dropdown(label="Contract", choices=["Month-to-month", "One year", "Two year"], value="Month-to-month", interactive= True) # The contract term of the customer
PaperlessBilling = gr.Radio(label="Paperless Billing", choices=yes_or_no, value="Yes") # Whether the customer has paperless billing or not
PaymentMethod = gr.Dropdown(label="Payment Method", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], value="Electronic check") # The customer's payment method
MonthlyCharges = gr.Slider(label="Monthly Charges", step=0.05, maximum=max_monthly_charges) # The amount charged to the customer monthly
TotalCharges = gr.Slider(label="Total Charges", step=0.05, maximum=max_total_charges) # The total amount charged to the customer


# Output
gr.Interface(inputs=[gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges],
             outputs = gr.Label("Awaiting Submission..."),
            fn=process_and_predict, 
            title= "Telecom Customer Churn Prediction App", 
            description= """This app uses a machine learning model to predict whether or not a customer will churn based on inputs made by you, the user. The (XGBoost) model was trained and built based on the Telecom Churn Dataset"""
            ).launch(inbrowser= True,
                     show_error= True)