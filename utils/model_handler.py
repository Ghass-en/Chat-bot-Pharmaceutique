# utils/model_handler.py (VERSION FINALE AVEC DISK OFFLOAD EXPLICITE)

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
import os

# --- NOUVEL IMPORT POUR LA GESTION EXPLICITE DU DISQUE ---
from accelerate import disk_offload

@st.cache_resource
def load_llm_pipeline(model_id, llm_params, token=None):
    print(f"Chargement du modèle LLM : {model_id}...")
    
    offload_folder = "./offload"
    if not os.path.exists(offload_folder):
        os.makedirs(offload_folder)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)

    # --- INSTRUCTION EXPLICITE POUR UTILISER LE DISQUE ---
    # On utilise le 'context manager' disk_offload comme recommandé par l'erreur
    with disk_offload(offload_dir=offload_folder):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # On spécifie le type pour la performance
            trust_remote_code=True,
            token=token
            # Note : On retire device_map, car disk_offload le gère
        )
    # --------------------------------------------------------
    
    # On doit spécifier sur quel device le pipeline doit tourner (CPU dans ce cas)
    text_generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=-1 # device=-1 force le CPU
    )
    
    print(f"Modèle LLM '{model_id}' chargé avec succès en utilisant le disk offload.")
    return text_generator, tokenizer

@st.cache_resource
def load_embedding_model(model_name):
    print(f"Chargement du modèle d'embedding : {model_name}...")
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    print("Modèle d'embedding chargé.")
    return embeddings_model