# index_spurgeon.py
# Run this script every time you add a new book
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, "..", "data", "chs_all-of-grace.txt")
DB_DIR = os.path.join(SCRIPT_DIR, "..", "chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "spurgeon_knowledge"

if not os.path.exists(FILE_PATH):
    print(f"ERREUR : Le fichier est introuvable ici : {os.path.abspath(FILE_PATH)}")
    exit()

# Windows-1257 encoding (cp1257)
try:
    loader = TextLoader(FILE_PATH, encoding="cp1257")
    docs = loader.load()
    print("Texte chargé avec succès (Windows-1257)")
except Exception as e:
    print(f"Erreur lors du chargement : {e}")
    exit()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=150
)

chunks = text_splitter.split_documents(docs)
print(f"Document découpé en {len(chunks)} segments.")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR,
    collection_name=VECTOR_STORE_NAME
)

print("Terminé ! La base de données est prête à être utilisée.")