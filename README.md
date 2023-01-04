# Customer Churn Prediction App with Gradio

## Introduction

A churn prediction app built with Gradio and wrapped around a machine learning model.
This project is more like a quest to discover how to embeded it into a web app with a user-friendly interface, in this case, [Gradio](https://gradio.app/). The objective is to have an interface that makes it easier for users to interact with an ML model, regardless of their level of knowledge in machine learning.

## Process Description

The process begins with exporting the necessary items from the notebook, building an interface that works correctly, importing the necessary items for modelling, and then writing the code to process inputs. The process can be summarized as:

- Export machine learning items from notebook,
- Import machine learning items into the app script,
- Build an interface,
- Write a function to process inputs,
- Pass values through the interface,
- Recover these values in backend,
- Apply the necessary processing,
- Submit the processed values to the ML model to make the predictions,
- Process the predictions obtained and display them on the interface.

## Installation

To setup and run this project you need to have [`Python3`](https://www.python.org/) installed on your system. Then you can clone this repo. At the repo's root, use the code from below which applies:

- Windows:

        python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

- Linux & MacOs:

        python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

    **NB:** For MacOs users, please install `Xcode` if you have an issue.

You can then run the app (still at the repository root):

- App built with Gradio Blocks

        python Churn_Prediction_App_with_grBlocks.py

- App built with the simple Gradio Interface

        python Churn_Prediction_App_with_grInterface.py

- With `inbrowser = True` defined, it should open a browser tab automatically. If it doesn't, type this address in your browser: <http://127.0.0.1:7860/>

## Screenshots

<table>
    <tr>
        <th>Gradio Churn Prediction App with Simple Gradio Interface</th>
        <th>Gradio Churn Prediction App with Gradio Blocks</th>
    </tr>
    <tr>
        <td><img src="screenshots\Churn_Prediction_App_with_grInterface.png"/></td>
        <td><img src="screenshots\Churn_Prediction_App_with_grBlocks.png"/></td>
    </tr>
</table>

## Contact Information

- [Kwame Otchere](https://kodoi-oj.github.io/)
- [![Twitter](https://img.shields.io/twitter/follow/kwameoo_?style=social)](https://twitter.com/kwameoo_)
