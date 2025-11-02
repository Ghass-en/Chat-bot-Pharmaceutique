import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

@st.cache_resource
def load_vector_store(index_path, _embedding_model):
    print(f"Chargement de l'index vectoriel depuis : {index_path}...")
    vector_store = FAISS.load_local(index_path, _embedding_model, allow_dangerous_deserialization=True)
    print("Index vectoriel chargé.")
    return vector_store.as_retriever(search_kwargs={'k': 2})

def create_rag_chain(retriever, llm_pipeline, llm_params, tokenizer):
    prompt_template = """<|user|>
Vous êtes un agent d'information médicale. Utilisez UNIQUEMENT le CONTEXTE suivant pour répondre à la QUESTION.
Si le contexte ne contient pas la réponse, dites : "L'information demandée n'est pas disponible dans ma base de connaissances pour le moment."

CONTEXTE:
{context}

QUESTION: {question}<|end|>
<|assistant|>
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = HuggingFacePipeline(pipeline=llm_pipeline, model_kwargs=llm_params)

    def format_and_truncate_docs(docs):
        context = "\n\n".join(doc.page_content for doc in docs)
        max_context_tokens = 3000
        encoded_context = tokenizer.encode(context, truncation=True, max_length=max_context_tokens)
        return tokenizer.decode(encoded_context)

    rag_chain = (
        {"context": retriever | format_and_truncate_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain