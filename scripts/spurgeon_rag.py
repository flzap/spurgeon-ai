import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
        formatted_chunks = []
        for doc in docs:
            m = doc.metadata
            title = m.get("title", "Unknown Title")
            s_no = m.get("sermon_no", "N/A")
            ref = m.get("bible_ref", "N/A")
            verse = m.get("bible_verse", "N/A")
            date = m.get("date", "Unknown Date")
            header = f"[SERMON No {s_no}: {title} | REF: {ref} | VERSE: {verse} | DATE: {date}]"
            formatted_chunks.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted_chunks)

def init_rag_system():
    """Initialise la base vectorielle, le LLM et la chaîne LangChain."""
    load_dotenv()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(BASE_DIR, "..", "chroma_db")

    PROMPT = "prompt_template.txt"
    PROMPT_PATH = os.path.join(BASE_DIR, "..", PROMPT)

    EMBEDDING_MODEL = "all-minilm"
    VECTOR_STORE_NAME = "spurgeon_minilm"
    TOP_K = 4 

    API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL = os.getenv("GEMINI_MODEL_NAME")
    TEMPERATURE = 0.7

    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        api_key=API_KEY,
        temperature=TEMPERATURE,
        streaming=True
    )
    
    vector_db = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=embeddings,
        collection_name=VECTOR_STORE_NAME
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K})
    
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read()
    prompt_template = ChatPromptTemplate.from_template(template)
    
    chain = prompt_template | llm | StrOutputParser()
    
    return retriever, chain

def get_spurgeon_stream(query, retriever, chain):
    """Récupère les documents (imprime dans le terminal) et retourne le flux de réponse."""
    # 1. Récupération des documents
    source_docs = retriever.invoke(query)
    
    # 2. Affichage DANS LE TERMINAL SEULEMENT
    print("\n" + "="*50)
    print("\n--- 🔍 WHAT THE BOT ACTUALLY READ ---")
    for i, doc in enumerate(source_docs):
        print(f"[{i+1}] {doc.page_content[:150]}...")
    print("="*50 + "\n")

    formatted_context = format_docs(source_docs)

    return chain.stream({"context": formatted_context, "question": query})