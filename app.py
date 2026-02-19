import streamlit as st
import time

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Spurgeon AI",
    page_icon="📖",
    layout="centered"
)

# --- CSS POUR RÉDUIRE L'ESPACEMENT ET AJUSTER LA TAILLE ---
# --- CSS POUR RÉDUIRE L'ESPACEMENT ET AJUSTER LE STYLE AI ---
st.markdown("""
<style>
    /* 1. Réduire l'espace blanc en haut */
    .block-container {
        padding-top: 2rem !important;
        max-width: 800px;
    }
    
    /* 2. Style du titre principal */
    h1 {
        color: var(--text-color); 
        font-family: 'Helvetica', sans-serif;
        text-align: center;
        margin-top: -20px; 
        padding-bottom: 0px;
        font-size: 3rem !important; 
    }
    
    /* 3. EFFET GRADIENT POUR LE TEXTE 'AI' (Comme l'avatar) */
    .highlight {
        background: linear-gradient(135deg, #00A896 0%, #45E6D0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        display: inline-block;
    }
    
    /* 4. Sous-titre et messages */
    .subtitle {
        text-align: center; 
        opacity: 0.7; 
        margin-top: -15px; 
        margin-bottom: 40px;
        font-size: 1.2rem;
    }
    
    .stChatMessage {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- FONCTION DE RÉPONSE (SIMULATION) ---
def get_spurgeon_response(user_input):
    time.sleep(1) 
    return f"En vérité, je vous le dis : votre question sur '{user_input}' mérite réflexion. Laissez-moi consulter les Écritures."

# --- INTERFACE UTILISATEUR ---

# 1. En-tête avec Logo mieux proportionné
# En changeant les proportions des colonnes (1.5 / 1 / 1.5), l'image est plus concentrée au centre
col1, col2, col3 = st.columns([1.5, 1, 1.5]) 
with col2:
    st.image("images/spurgeon_avatar.png", width="stretch")
    
# Titre et sous-titre sortis de la colonne pour un meilleur centrage global
st.markdown("<h1>SPURGEON <span class='highlight'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Votre assistant théologique numérique</p>", unsafe_allow_html=True)

# 2. Gestion de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Affichage des messages
for message in st.session_state.messages:
    avatar_image = "images/spurgeon_avatar.png" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar_image): 
        st.markdown(message["content"])

# 4. Zone de saisie
if prompt := st.chat_input("Posez votre question à M. Spurgeon..."):
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="images/spurgeon_avatar.png"):
        message_placeholder = st.empty()
        full_response = ""
        
        response_text = get_spurgeon_response(prompt)
        
        for chunk in response_text.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})


with st.sidebar:
    st.header("Paramètres")
    st.slider("Température (Créativité)", 0.0, 1.0, 0.7)
    st.selectbox("Modèle", ["Gemini 3 Pro", "Gemini 2.5 Flash"])