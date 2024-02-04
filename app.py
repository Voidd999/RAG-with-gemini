import os
import time
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai


# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Chat with Gemini-Pro!",
    page_icon=":gemini:",  
    layout="centered",  
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel("gemini-pro")

def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

st.title("Google's Gemini Pro - ChatBot")

for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

user_prompt = st.chat_input("Ask Gemini-Pro...")
if user_prompt:
    st.chat_message("user").markdown(user_prompt)

    response = st.session_state.chat_session.send_message(user_prompt)

   
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = response
        
        for chunk in response:
          
            for ch in chunk.text.split(" "):
                full_response += ch + " "
                time.sleep(0.05)
        
                message_placeholder.write(full_response + "|")
       
        message_placeholder.write(full_response)
