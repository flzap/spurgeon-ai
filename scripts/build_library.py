# Eventuellement, utiliser langchain index pour ne pas construire tout le db a chaque fois

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# GESTION DES CHEMINS 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
BASE_DIR = os.path.join(SCRIPT_DIR, "..")               
DATA_PATH = os.path.join(BASE_DIR, "data")              
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "spurgeon_knowledge"

def build_mega_index():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Erreur : Le dossier data est introuvable ici : {DATA_PATH}")
        return

    # CHARGEMENT 
    print(f"--- 📂 Chargement de TOUS les sermons dans {DATA_PATH} ---")
    # On utilise ton encodage cp1257 pour ne pas avoir de bugs de caractères
    loader = DirectoryLoader(
        DATA_PATH, 
        glob="**/*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'} 
    )
    docs = loader.load()
    print(f"✅ {len(docs)} fichiers trouvés.")

    # DÉCOUPAGE
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"✅ {len(splits)} morceaux de texte créés.")

    # CRÉATION DES VECTEURS ET SAUVEGARDE
    print(f"--- 🧠 Génération des vecteurs ({EMBEDDING_MODEL}) ---")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # On spécifie le collection_name que tu utilisais déjà
    vector_db = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_PATH,
        collection_name=VECTOR_STORE_NAME
    )
    print(f"🚀 Base de données mise à jour avec succès!")

if __name__ == "__main__":
    build_mega_index()