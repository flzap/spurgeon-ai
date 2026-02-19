import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assure-toi que ce chemin est le bon (celui qui contient tes 2121 docs)
DB_DIR = r"C:\Users\flzap\Documents\Universite\Spurgeon_AI\chroma_db" 
COLLECTION_NAME = "langchain" # ou "spurgeon_knowledge" selon ce que tu as gardé

print(f"🕵️‍♂️ Connexion à : {DB_DIR}")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

# 1. EST-CE QUE LE SERMON N°1 EST DANS LA BASE ?
print("\n--- TEST 1 : Recherche par MÉTADONNÉES (Sermon No. 1) ---")
# On demande à Chroma de nous sortir les docs qui ont sermon_no = "1"
results_meta = vector_db.get(where={"sermon_no": "1"})

if len(results_meta['ids']) > 0:
    print(f"✅ TROUVÉ ! Il y a {len(results_meta['ids'])} morceaux pour le Sermon n°1.")
    print(f"Titre enregistré : {results_meta['metadatas'][0].get('title', 'Inconnu')}")
    print(f"Extrait du texte : {results_meta['documents'][0][:100]}...")
else:
    print("❌ ALERTE ROUGE : Le Sermon n°1 n'est PAS dans la base de données !")
    print("Cela veut dire que ton parsing a sauté le premier sermon (probablement à cause du header).")

# 2. EST-CE QUE LA RECHERCHE VECTORIELLE MARCHE ?
print("\n--- TEST 2 : Recherche VECTORIELLE 'Immutability' ---")
query = "Immutability of God"
results_vec = vector_db.similarity_search_with_score(query, k=3)

for doc, score in results_vec:
    print(f"[Score: {score:.4f}] Sermon {doc.metadata.get('sermon_no')}: {doc.metadata.get('title')}")