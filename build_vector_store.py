import pandas as pd
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CSV_FILE_PATH = "data/drugs_data.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"

def create_vector_store():
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Index FAISS déjà présent. Pour recréer, supprimez le dossier '{FAISS_INDEX_PATH}'.")
        return
    if not os.path.exists(CSV_FILE_PATH):
        print(f"ERREUR: Fichier '{CSV_FILE_PATH}' introuvable.")
        return

    print("Chargement des données...")
    df = pd.read_csv(CSV_FILE_PATH)

    print("Préparation des données...")
    df['full_info'] = df.apply(lambda row: " ".join([
        f"{col.replace('_', ' ').capitalize()}: {row[col]}"
        for col in df.columns if col not in ['drug_name', 'full_info'] and pd.notna(row[col])
    ]), axis=1)

    docs = [Document(page_content=row['full_info'], metadata={"drug_name": row['drug_name']})
            for _, row in df.iterrows() if 'full_info' in row and isinstance(row['full_info'], str) and row['full_info'].strip()]
    
    if not docs:
        print("ERREUR: Aucun document n'a pu être créé. Vérifiez le CSV.")
        return

    print("Chargement du modèle d'embedding...")
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Création de l'index FAISS...")
    vectorstore = FAISS.from_documents(docs, embeddings_model)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Index sauvegardé dans '{FAISS_INDEX_PATH}'.")

if __name__ == "__main__":
    create_vector_store()