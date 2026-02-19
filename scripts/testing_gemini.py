import os
import sys
import threading
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Charger les variables (.env)
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(SCRIPT_DIR, "..", "prompt_template.txt")
DB_DIR = os.path.join(SCRIPT_DIR, "..", "chroma_db")
EMBEDDING_MODEL = "all-minilm" # "nomic-embed-text"
VECTOR_STORE_NAME = "spurgeon_minilm" #"spurgeon_knowledge"

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL_NAME")
TEMPERATURE = 0.7


def loading_animation(stop_event):
    chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r⏳ Spurgeon is reflecting {chars[i % len(chars)]} ")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " " * 30 + "\r")


embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


llm = ChatGoogleGenerativeAI(
    model=MODEL,
    api_key=API_KEY,
    temperature=TEMPERATURE
)

vector_db = Chroma(
    persist_directory=DB_DIR, 
    embedding_function=embeddings,
    collection_name=VECTOR_STORE_NAME
)


def test_chroma_connection():
    try:
        count = vector_db._collection.count()
        print(f"📊 Nombre de documents trouvés dans la collection : {count}")
        if count == 0:
            print("⚠️ ALERTE : La base de données est vide ! Vérifie le chemin.")
        else:
            print("✅ La connexion est bonne, le problème vient peut-être des embeddings.")
    except Exception as e:
        print(f"❌ Erreur de connexion Chroma : {e}")


retriever = vector_db.as_retriever(search_kwargs={"k": 4})

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    template = f.read()
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    formatted_chunks = []
    
    for doc in docs:
        # 1. On extrait les métadonnées de l'objet Document
        m = doc.metadata
        title = m.get("title", "Unknown Title")
        s_no = m.get("sermon_no", "N/A")
        ref = m.get("bible_ref", "N/A")
        verse = m.get("bible_verse", "N/A")
        date = m.get("date", "Unknown Date")
        
        # 2. On crée un "En-tête de contexte" pour ce morceau précis
        # On injecte le titre et la référence biblique AU DÉBUT du texte
        header = f"[SERMON No {s_no}: {title} | REF: {ref} | VERSE: {verse} | DATE: {date}]"
        
        # 3. On combine l'en-tête avec le contenu du sermon
        full_chunk = f"{header}\n{doc.page_content}"
        formatted_chunks.append(full_chunk)
    
    # 4. On joint tous les morceaux avec un séparateur clair
    return "\n\n---\n\n".join(formatted_chunks)
# def format_docs(docs):
#    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print("\n--- 🎩 Welcome to Mr. Spurgeon's Study ---")
    
    while True:
        query = input("\n👤 You: ")
        if query.lower() in ["exit", "q", "quit"]:
            break
            
        stop_loading = threading.Event()
        loader_thread = threading.Thread(target=loading_animation, args=(stop_loading,))
        loader_thread.start()
        
        try:
            source_docs = retriever.invoke(query)
            print("\n--- 🔍 WHAT THE BOT ACTUALLY READ ---")
            for i, doc in enumerate(source_docs):
                print(f"[{i+1}] {doc.page_content[:150]}...")
            print("------------------------------------\n")
            stream = chain.stream(query)
            
            first_chunk = next(stream)
            stop_loading.set()
            loader_thread.join()
            
            print(f"\n📖 SPURGEON: {first_chunk}", end="", flush=True)
            
            for chunk in stream:
                print(chunk, end="", flush=True)
                
        except Exception as e:
            stop_loading.set()
            print(f"\nError: {e}")
            
        print("\n" + "-" * 50)

