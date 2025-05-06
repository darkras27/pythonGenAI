import streamlit as st
from streamlit.logger import get_logger
logger = get_logger(__name__)

import os
if os.getenv('USER', 'NONE') == 'appuser': #streamlit
    ht_token = st.secrets["HF_TOKEN"]
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = ht_token
else:
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.environ['HF_API_KEY']

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace


st.title("My GEN AI Application")
repo_id = "microsoft/Phi-3-mini-4k-instruct"
temp = 1
print(repo_id, temp)
logger.info(f"repo_id: {repo_id}, temp: {temp}")

with st.form("sample_app"):
    txt = st.text_area("Enter Text", "What GPT stands for?")
    sub = st.form_submit_button("Submit")
    if sub:
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            temperature=temp,
        )
        chat = ChatHuggingFace(llm=llm, verbose=True)
        logger.info(f"Invoking")
        ans = chat.invoke(txt)
        st.info(ans.content)
        logger.info("Done")
                