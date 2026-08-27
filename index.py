# index.py
from streamlit_lambda import streamlit_lambda
from app import app as streamlit_app  # Votre instance Streamlit

# Le wrapper streamlit_lambda transforme votre app en fonction handler
lambda_app = streamlit_lambda(streamlit_app)

def handler(event, context):
    return lambda_app(event, context)
