import streamlit as st
import time
from scripts.spurgeon_rag import init_rag_system, get_spurgeon_stream

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Spurgeon AI",
    page_icon="📖",
    layout="centered"
)

# --- CSS PERSONNALISÉ ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; max-width: 800px; }
    h1 { color: var(--text-color); font-family: 'Helvetica', sans-serif; text-align: center; margin-top: -20px; font-size: 3rem !important; }
    .highlight { 
        background: linear-gradient(135deg, #00A896 0%, #45E6D0 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: bold; display: inline-block;
    }
    .subtitle { text-align: center; opacity: 0.7; margin-top: -15px; margin-bottom: 40px; font-size: 1.2rem; }
    .stChatMessage { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# --- INITIALISATION RAG (MISE EN CACHE) ---
@st.cache_resource
def load_backend():
    return init_rag_system()

retriever, rag_chain = load_backend()

# --- INTERFACE UTILISATEUR ---
col1, col2, col3 = st.columns([1.5, 1, 1.5]) 
with col2:
    st.image("images/spurgeon_avatar.png", width="stretch")
    
st.markdown("<h1>SPURGEON <span class='highlight'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Votre assistant théologique numérique</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    avatar_image = "images/spurgeon_avatar.png" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar_image): 
        st.markdown(message["content"])



# --- LOGIQUE DE CHAT ---
if prompt := st.chat_input("Posez votre question à M. Spurgeon..."):

    if prompt.lower() in ["exit", "q", "quit", "bye"]:
        st.session_state.messages.clear()
        st.success("Le bureau de M. Spurgeon est fermé. Historique nettoyé !")
        st.rerun() # Rafraîchit la page

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="images/spurgeon_avatar.png"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("M. Spurgeon est en réfléction..."):
            stream = get_spurgeon_stream(prompt, retriever, rag_chain)
            try:
                first_chunk = next(stream)
            except StopIteration:
                first_chunk = ""

        for char in first_chunk:
            full_response += char
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
            
        for chunk in stream:
            for char in chunk:
                full_response += char
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)
            
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


with st.sidebar:
    st.header("Paramètres")
    st.slider("Température (Créativité)", 0.0, 1.0, 0.7)
    st.selectbox("Modèle", ["Gemini 3 Pro", "Gemini 2.5 Flash"])