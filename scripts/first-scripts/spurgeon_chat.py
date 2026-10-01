import os
import sys
import threading
import time
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(SCRIPT_DIR, "..", "prompt_template.txt")
DB_DIR = os.path.join(SCRIPT_DIR, "..", "chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "spurgeon_knowledge"
MODEL = "llama3.2"
TEMPERATURE = 0.1

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
llm = ChatOllama(model=MODEL, temperature=TEMPERATURE)

vector_db = Chroma(
    persist_directory=DB_DIR, 
    embedding_function=embeddings,
    collection_name=VECTOR_STORE_NAME
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    template = f.read()
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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


# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an AI language model assistant. Your task is to generate five
#     different versions of the given user question to retrieve relevant documents from
#     a vector database. By generating multiple perspectives on the user question, your
#     goal is to help the user overcome some of the limitations of the distance-based
#     similarity search. Provide these alternative questions separated by newlines.
#     Original question: {question}""",
# )

# retriever = MultiQueryRetriever.from_llm(
#     vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
# )

# template = """
# You are Charles H. Spurgeon. Answer the question based ONLY on the following context:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
## Test
# res = chain.invoke("What does it mean to be saved by grace?")
# print(res)


