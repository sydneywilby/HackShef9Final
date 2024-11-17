## FIREFLY SYSTEM (TensorFlow, MongoDB, Auth0)

### What is it?

A secure remote monitoring system for fires, especially wildfires utilising TensorFlow, MongoDB and Auth0. Old devices can be turned into fire recognition sensors and accessed easily using the simple and secure web portal.

### How to use it

The Dataset

- Choose a dataset, such as [this](https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset)

- Make sure the directory of the images is separated into two subdirectories by class (Fire and Non Fire)

Training the model

- Set the directory of the images in the model.py file

- Run the model.py file

Run the main website

- Set your environment variables using .env.example as a base

- Run the website.py file

## Other features (StreamLit)

### View system status of the FIREFLY SYSTEM

streamlit run "./streamlit/system.py"

### Test the model against images stored locally

streamlit run "./streamlit/aitester.py"
