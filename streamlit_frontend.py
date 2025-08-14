import streamlit as st

with st.chat_message('user'):
    st.text('Hi')

with st.chat_message('assistant'):
    st.text('How may I help you?')