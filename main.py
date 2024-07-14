# main.py
import streamlit as st
from predict import load_model_and_labels, predict_disease
from llm_handler import get_response_from_llm
import os
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


def reset_chat():
    st.session_state.conversation = []
    st.session_state.disease = None
    if 'llm_chain' in st.session_state:
        del st.session_state.llm_chain


if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'disease' not in st.session_state:
    st.session_state.disease = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_labels' not in st.session_state:
    st.session_state.class_labels = None
if 'new_upload' not in st.session_state:
    st.session_state.new_upload = False

st.title("Plant Disease Classification and Information")

uploaded_file = st.file_uploader("Choose an image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    
    if uploaded_file != st.session_state.last_uploaded_file:
        st.session_state.new_upload = True
        reset_chat()
        st.session_state.last_uploaded_file = uploaded_file
    
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    if st.session_state.model is not None and st.session_state.class_labels is not None:
        model = st.session_state.model
        class_labels = st.session_state.class_labels
    else:
        model, class_labels = load_model_and_labels()
        st.session_state.model = model
        st.session_state.class_labels = class_labels    
    
    disease = predict_disease(image, model,class_labels)
    st.session_state.disease = disease
    st.write(f"Predicted Disease: {disease}")

    if st.button("Get General Information About the Disease", type="primary"):
        prompt = f'Give me general information about this disease: {disease}'
        response = get_response_from_llm(prompt)
        st.session_state.conversation.append(("Human", prompt))
        st.session_state.conversation.append(("AI", response))
        st.rerun()  
    
    for role, message in st.session_state.conversation:
        if role == "Human":
            st.chat_message("human").write(message)
        else:
            st.chat_message("assistant").write(message)


    user_input = st.text_input("Ask a follow-up question about the disease and click submit:")
    if st.button("Submit"):
        response = get_response_from_llm(user_input)
        st.session_state.conversation.append(("Human", user_input))
        st.session_state.conversation.append(("AI", response))
        st.rerun()  
