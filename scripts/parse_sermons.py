## TOOK 18 minutes for volume 1
import re
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_spurgeon_volume(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Split the file by dash lines (sermon separators)
    # Use a regular expression to find long sequences of underscores
    raw_sermons = re.split(r'_{30,}', content)
    
    prepared_documents = []
    is_header = True
    
    for raw_text in raw_sermons:
        # Ignore empty fragments or those that are too short
        if len(raw_text.strip()) < 100:  
            continue
    
        if is_header:
            is_header = False
            continue  # Ignore the first segment which is the volume header
        
        # Clean lines and extract the title
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        title = lines[0]
        
        # Stop processing if we reach the index section
        if title == "Index of Scripture References":
            break 

        # Extract Date
        date_match = re.search(r"Delivered on (.*), by the", raw_text)
        if not date_match:
            raise ValueError(f"Missing Date in: {title}")
        date = date_match.group(1)

        # Extract Location
        place_match = re.search(r"At (.*)", raw_text)
        if not place_match:
            raise ValueError(f"Missing Location in: {title}")
        # Strip trailing whitespace and the final period if it exists
        place = place_match.group(1).strip().rstrip('.')

        # Extract Sermon Number (handles single No. or ranges like No. 7-8)
        num_match = re.search(r"\(No\. (\d+(?:[\-–—]\d+)?)\)", raw_text)
        if not num_match:
            raise ValueError(
                f"CRITICAL ERROR: Unable to detect sermon number (No. X) "
                f"in sermon titled: '{title}'. \n")
        
        sermon_no = num_match.group(1)

        # Extract Biblical Reference
        ref_pattern = r"([1-2]?\s?[A-Z][a-z]+\.?\s+\d+:\d+(?:[\-–—]\d+)?)"
        ref_match = re.search(ref_pattern, raw_text)

        if not ref_match:
            raise ValueError(f"ERROR: Biblical reference not found in sermon: '{title}'")

        bible_ref = ref_match.group(1)
            
        # 2. Search for the verse text between quotes
        # Secret: Allow EVERYTHING (including newlines) between quotes and the reference
        # Using [\s\S]+? which means "absolutely any character, including newlines"
        verse_pattern = r"\"([\s\S]+?)\"\s*[\-–—]+\s*" + re.escape(bible_ref)
        verse_match = re.search(verse_pattern, raw_text)

        if not verse_match:
            raise ValueError(f"ERROR: Verse text not found for reference {bible_ref} in: '{title}'")
        
        # Clean up verse text: remove newlines and collapse multiple spaces
        verse_text = verse_match.group(1).replace('\n', ' ').strip()
        verse_text = re.sub(r'\s+', ' ', verse_text)

        # Clean up the biblical reference (normalize spaces/newlines)
        bible_ref = bible_ref.replace('\n', ' ').strip()
        bible_ref = re.sub(r'\s+', ' ', bible_ref)
        
        # Console Output for Verification
        # print_sermon_metadata(title, sermon_no, date, place, bible_ref, verse_text)
        # Create the LangChain Document object
        doc = Document(
            page_content=raw_text.strip(),
            metadata={
            "title": title,
                "sermon_no": sermon_no,
                "date": date,
                "location": place,
                "bible_ref": bible_ref,
                "bible_verse": verse_text,
                "source": file_path
            }
         )
        prepared_documents.append(doc)
        # Save to json? Use this json as database instead of the raw text file

    return prepared_documents

def print_sermon_metadata(title, sermon_no, date, location, bible_ref, verse_text):
    print('Title:', title)
    print('Sermon No:', sermon_no)
    print('Date:', date)
    print('Location:', location)
    print('Biblical Reference:', bible_ref)
    print('Verse Text:', verse_text)
    print('---\n')

# Usage
# Set the path relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOL_PATH = os.path.join(SCRIPT_DIR, "..", "data/sermons01.txt")

docs = process_spurgeon_volume(VOL_PATH)


BASE_DIR = os.path.join(SCRIPT_DIR, "..")               
DATA_PATH = os.path.join(BASE_DIR, "data")              
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL = "all-minilm" # "nomic-embed-text"
VECTOR_STORE_NAME = "spurgeon_minilm" #"spurgeon_knowledge"

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=200
    )

splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_db = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings, 
    persist_directory=DB_PATH,
    collection_name=VECTOR_STORE_NAME
)
print("✅ Indexation intelligente terminée !")