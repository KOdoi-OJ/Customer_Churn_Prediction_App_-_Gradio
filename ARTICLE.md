
# Build and Deploy a Customer Churn Prediction app with Gradio

![Preview Image](https://miro.medium.com/max/875/1*k1tY8fbSIi8qw2Kzmv4fSQ.png)

App Interface

Churn prediction is a critical task for any business that relies on recurring revenue, such as subscription-based services and membership organizations. By identifying customers who are at risk of churning, companies can take steps to retain them and prevent revenue loss. In this article, I will briefly walkthrough how to build and deploy a churn prediction app with Gradio, a platform that allows you to easily create and share interactive web apps with machine learning models.

I will use the model I built in  [_this project_](https://github.com/KOdoi-OJ/Telecom_Customer_Churn_Prediction), and build the interface, write the key function, show how to deploy it on Gradio, and use it to make predictions for users. Let’s get started!

This article will be a relatively short read because the principles and procedure for building and deploying an app on Gradio are similar to Streamlit (as shown in this article I published). Whether you are familiar with the topic or new to the topic, you can read this article in its entirety to learn everything you need to know to get started. I hope you find it useful.

_I will begin from exporting the relevant items to be used, setting up your environment, importing the items, building your interface, and completing the backend._

## 1.0 Introduction

**1.1 Gradio  
**Gradio, much like Streamlit, is “a free and open-source Python library that allows you to develop an easy-to-use customizable component demo for your machine learning model that anyone can use anywhere.” All this can be done with any of a  **_Python script_** _or_ **_Jupyter notebook._**

**1.2 Why deployment?  
**Deployment is the last stage of the CRISP-DM, the framework used to guide this Customer Churn Prediction project. As stated earlier, churn prediction is a critical task for companies that rely on recurring revenue, and the ability to accurately predict it will be like a superpower for the firms in informing their customer relations, decision making and planning.

It is therefore important to make available potentially useful apps such as this to enable organizations make better decisions and design better strategies; this is a key reason why I built this model into an app.

# 2.0 The Process

**2.1 Workflow overview  
**As indicated earlier, the workflow may be summarized as follows:

-   Export ML items
-   Set up environment
-   Import ML items
-   Build interface
-   Write the function to process inputs and display outputs
-   Deploy the app

**2.2 Toolkit Export  
**The process begins with exporting the key items used during your modelling process from your notebook. The toolkit typically includes the encoder, scaler, model, and pipeline (if used). For ease of access, these items may be put together in a dictionary and exported. In this case, Pickle will be used for the exports, so it must first be imported.

# Import Pickle  
import pickle

The dictionary can then be created and exported with pickle as shown below;

![](https://miro.medium.com/max/565/1*nqXjPFvgDv2IhHIci5LqjA.png)

Collect ML items and export them with Pickle

Note that the values in the dictionary are the names of the variables that represent my encoder, scaler, and model. And the names of the output file can be changed as desired.

Since your workflow likely used specific libraries and modules, they also have to be exported with the help of the OS library into a text file called  _requirements_:

# Import OS  
import os

After importing OS, you may then export the requirements with:

# Exporting the requirements  
requirements = "\n".join(f"{m.__name__}=={m.__version__}" for m in globals().values() if getattr(m, "__version__", None))  
  
with open("requirements.txt", "w") as f:  
    f.write(requirements)

Other things being equal, this should be the last major action you take in your notebook. Next up is VSCode.  _You can use Jupyter, I used VSCode because it was relatively easier to debug._

**2.3 Setting up your environment  
**This step involves creating the folder or repository for your app. You may want to create a  _resources_ folder to hold the items you have exported from your notebook. The  _requirements_ file should be at the root of your repository or main folder.

To prevent any conflicts with your variables, you may use the following code to create a virtual environment, activate it in your terminal, and install the requirements in your  _requirements file_:

# Create and activate virtual environment  
python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt

**2.4 Importing ML Items  
**Be sure to switch to the workspace of the virtual environment. You’ll know it is active when you have  _(venv)_ preceding the path to your current working directory in the terminal.

Next is to define a function to load your ML items. In my case, I used the code below which has a default value for the file path:

# Function to load ML toolkit  
def load_ml_toolkit(file_path=r"gradio_src\Gradio_App_toolkit"):  
    with open(file_path, "rb") as file:  
        loaded_toolkit = pickle.load(file)  
    return loaded_toolkit

You may then instantiate the toolkit and each of the items you exported;

# Importing the toolkit  
loaded_toolkit = load_ml_toolkit(r"gradio_src\Gradio_App_toolkit")  
encoder = loaded_toolkit["encoder"]  
scaler = loaded_toolkit["scaler"]  
  
# Import the model  
model = XGBClassifier()  
model.load_model(r"gradio_src\xgb_model.json")

You may have noticed that I loaded the XGBoost model separately; this is because it was not working well with Pickle, so I had to use the  _save_model()_ function to export it and load it as shown above.

# Exporting the model  
best_xgb_model.save_model("xgb_model.json")

**2.5 Building your interface  
**From there you build your interface using the components provided by Gradio. You may go with a simple interface in gr.Interface() or one that allows for a higher level of customization in Gradio Blocks i.e. gr.Blocks(). I have scripts that show how to use each in  [_the repository_](https://github.com/KOdoi-OJ/Customer_Churn_Prediction_App_-_Gradio), you may check them out. In any case — as with Streamlit — if you go with Gradio Blocks the most common components you’re going to use are:

-   _gr._Column_()_: to define a column (vertical space) in your workspace.
-   _gr.Row()_: to define a row (horizontal space) in your workspace
-   _gr.Dropdown()_: for a dropdown with options
-   _gr.Radio()_: for a radio
-   _gr.Slider()_: for a slider
-   _gr.Accordion()_: for an expander
-   _gr.Markdown()_: to write text within the workspace
-   _gr.Button()_: for a button which will trigger a sequence of events when clicked

You may begin app interface design with the code block below, and use a mix of rows and columns to control layout, while you use dropdowns, sliders and radios to receive inputs;

# Set up blocks  
with gr.Blocks() as gr_app:

**2.6 Setting up the backend  
**Aside the differences in options for customizing their interfaces, the major difference between Streamlit and Gradio is that while Streamlit allows the script to flow from top to bottom to process inputs as designed, Gradio requires you to define a function to process and inputs. It requires the three main things (inputs, outputs, and function) at one place. As can be seen below:

 # Output Prediction with Gradio Blocks  
    output = gr.Label("Awaiting Submission...")  
    submit_button = gr.Button("Submit")  
      
    # Code Block to Process Inputs and Return Outputs  
    submit_button.click(fn = process_and_predict,  
                        outputs = output,  
                        inputs = [gender, SeniorCitizen, Partner, Dependents])

For gr.Interface() interface, you may have something like this as a summary of the process:

# Output Prediction with gr.Interface()  
gr.Interface(inputs=[gender, SeniorCitizen, Partner, Dependents, tenure],  
            outputs=gr.DataFrame(headers= ["Churn Status Prediction"]),   
            fn=process_and_predict  
            ).launch(inbrowser= True)

_Please note that I have truncated the inputs to ensure that the code blocks fit fairly within the screen. Also note that in both cases, the components to receive inputs have been assigned to the variables that are referenced here at the processing stage._

Notice the difference in the output variables? Yes, that’s some of the options available for you to display outputs in Gradio.

After building the interface as you desire, you may then design the function to process the inputs and return outputs to the user. Here, again, the workflow must be same as in your notebook, and is typically:  _Inputs -> Encoding -> Scaling -> Predicting -> Returning predictions._

You may refer to the notebook for details on the inputs and the function. Unfortunately, the disadvantage of building a Gradio app using VSCode — at least for me — was that I had to terminate the terminal and rerun it if I wanted to see or test the app. It had no option to  **_“Always rerun”_** to automatically effect changes to the script like Streamlit.

In any case, you must state the  _launch_ function before the app can run. This can be done as follows:

# Launch the app in Gradio Blocks  
gr_app.launch(inbrowser= True, share=True)

The “_inbrowser = True_” argument is specified so that the app loads in browser when the script is run.

**2.7 Deployment  
**You have the option to deploy for free by adding “_share = True”_ inside your launch function. The deployed app stays online for just about 72 hours. For an extended period of hosting, you may explore more options on the Gradio site.

# 3.0 Final Notes

Thank you for reading this far, I hope the article was helpful to you.