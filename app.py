import streamlit as st
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

from utils.model_handler import load_llm_pipeline, load_embedding_model
from utils.conversation import load_vector_store, create_rag_chain

try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    st.error("Fichier 'config.yaml' introuvable.")
    st.stop()

hf_token = os.getenv("HUGGINGFACE_TOKEN")

st.set_page_config(page_title="Agent Médical", page_icon="💊", layout="centered")
st.title("💊 Agent d'Information sur les Médicaments")

with st.spinner("Initialisation de l'agent IA... Veuillez patienter."):
    llm_pipeline, tokenizer = load_llm_pipeline(config['llm']['model_id'], config['llm_params'], token=hf_token)
    embedding_model = load_embedding_model(config['embedding']['model_name'])
    retriever = load_vector_store(config['paths']['faiss_index'], embedding_model)
    rag_chain = create_rag_chain(retriever, llm_pipeline, config['llm_params'], tokenizer)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Posez-moi une question sur un médicament."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche..."):
            response = rag_chain.invoke(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})