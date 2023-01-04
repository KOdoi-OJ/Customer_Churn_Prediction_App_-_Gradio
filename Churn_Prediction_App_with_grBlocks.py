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

# Import the model
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
with gr.Blocks() as turn_on_the_gradio:
    gr.Markdown("# Telecom Customer Churn Prediction")
    gr.Markdown("""This app uses a machine learning model to predict whether or not a customer will churn based on inputs made by you, the user. The (XGBoost) model was trained and built based on the Telecom Churn Dataset. You may refer to the expander at the bottom for more information on the inputs.""")
    
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
    
    # Expander for more info on columns
    with gr.Accordion("Open for information on inputs"):
        gr.Markdown("""This app receives the following as inputs and processes them to return the prediction on whether a customer, given the inputs, will churn or not.
                    - Contract: The contract term of the customer (Month-to-Month, One year, Two year)
                    - Dependents: Whether the customer has dependents or not (Yes, No)
                    - DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)
                    - Gender: Whether the customer is a male or a female
                    - InternetService: Customer's internet service provider (DSL, Fiber Optic, No)
                    - MonthlyCharges: The amount charged to the customer monthly
                    - MultipleLines: Whether the customer has multiple lines or not
                    - OnlineBackup: Whether the customer has online backup or not (Yes, No, No Internet)
                    - OnlineSecurity: Whether the customer has online security or not (Yes, No, No Internet)
                    - PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)
                    - Partner: Whether the customer has a partner or not (Yes, No)
                    - Payment Method: The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))
                    - Phone Service: Whether the customer has a phone service or not (Yes, No)
                    - SeniorCitizen: Whether a customer is a senior citizen or not
                    - StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No Internet service)
                    - StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)
                    - TechSupport: Whether the customer has tech support or not (Yes, No, No internet)
                    - Tenure: Number of months the customer has stayed with the company
                    - TotalCharges: The total amount charged to the customer
                    """)
    
    submit_button.click(fn = process_and_predict,
                        outputs = output,
                        inputs = [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges])

turn_on_the_gradio.launch(favicon_path=r"gradio_src\app_thumbnail.png",
                          inbrowser= True)