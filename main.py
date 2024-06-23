import streamlit as st
from aiModel import get_chain, create_vector_db

st.title("English to Hindi Translation")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()
    pass

question = st.text_input("Question: ")

if question:
    chain = get_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])