# llm_handler.py

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os

def initialize_llm():
    template = """
    You are an expert in plant diseases. Use the following conversation history and human input to provide detailed information and tips about the plant disease.

    Conversation history:
    {chat_history}

    H: {human_input}
    AI Assistant: """

    prompt = PromptTemplate.from_template(template)

    if "memory" not in st.session_state or st.session_state.new_upload:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.memory = memory
        print(f"memory created {memory.load_memory_variables({})}")
    else:
        memory = st.session_state.memory
        print(f"memory found {memory.load_memory_variables({})}")

    api_key = os.getenv("CHATGROQ_API_KEY")
    llm = ChatGroq(
        temperature=0.2,
        model_name='Mixtral-8x7b-32768',
        groq_api_key=api_key
    )

    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
    return conversation

def get_response_from_llm(human_input):
    if 'llm_chain' not in st.session_state:
        st.session_state.llm_chain = initialize_llm()
    
    response = st.session_state.llm_chain.predict(human_input=human_input)
    return response