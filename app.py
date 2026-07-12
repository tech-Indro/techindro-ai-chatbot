import streamlit as st
import os
import io
import json
import sqlite3
import shutil
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import PyPDF2
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai

# RAG & ML imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from current directory or backend/
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    load_dotenv(os.path.join("backend", ".env"))

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    # Strip whitespace and enclosing quotes
    api_key = api_key.strip().strip("'\"")
    # Verify if it's a default placeholder, too short, or the leaked key
    if api_key in ["YOUR_GEMINI_API_KEY_HERE", "your_api_key_here", "AIzaSyBqU-Tbo_eIUOEY76pKyMuJU0yTFiTNVoA", ""] or len(api_key) < 15:
        api_key = None
    else:
        genai.configure(api_key=api_key, transport='rest')




# ─── Constants ──────────────────────────────────────────────────────
DB_PATH = os.path.join("instance", "database.db")
VECTOR_STORE_DIR = os.path.join("instance", "vector_stores")
os.makedirs("instance", exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# System personas (identical to Flask backend)
ROLES = {
    "Student": "You are a helpful AI assistant for students. Answer study-related queries in simple language.",
    "Teacher": "You are an AI assistant for teachers. Help with lesson plans, assessments, and teaching tips.",
    "Farmer": "You are an AI assistant for farmers. Give advice on crops, weather, and best practices.",
    "Doctor": "You are an AI assistant for doctors. Provide medical references and research support. Do not give direct diagnoses.",
    "Women": "You are an AI assistant for women's support. Give advice on health, career, safety, and empowerment."
}

# ─── Cache Resources ────────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    """Load and cache the sentence-transformers model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ─── Database Helpers ───────────────────────────────────────────────
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Create tables matching Flask SQLAlchemy schema
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username VARCHAR(150) UNIQUE NOT NULL,
        password VARCHAR(150) NOT NULL
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_session (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title VARCHAR(200) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user (id)
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS message (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        role VARCHAR(50) NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES chat_session (id)
    )""")
    conn.commit()
    conn.close()

# Initialize Database
init_db()

# ─── Page config ────────────────────────────────────────────────────
st.set_page_config(page_title="Raplica AI Chatbot", page_icon="🤖", layout="wide")

# Theme styling to matches premium look
st.markdown("""
<style>
    .stApp {
        background-color: #fbf8f4;
        color: #1a1a1a;
    }
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        color: #4A5EE5;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f5f0e9;
    }
    div[data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "tts_text" not in st.session_state:
    st.session_state.tts_text = None
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = None


# ─── AUTHENTICATION FLOW ─────────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown('<div class="main-header">rapl<span style="color:#b57452">•</span>ca AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Role Conversational Intelligence with RAG and Voice</div>', unsafe_allow_html=True)
    
    auth_mode = st.radio("Choose Authentication Mode", ["Sign In", "Register"], horizontal=True)
    
    with st.form("auth_form"):
        username_input = st.text_input("Username").strip()
        password_input = st.text_input("Password", type="password")
        submit_btn = st.form_submit_button(auth_mode)
        
        if submit_btn:
            if not username_input or not password_input:
                st.error("Please fill in all fields.")
            else:
                conn = get_db_connection()
                cursor = conn.cursor()
                if auth_mode == "Sign In":
                    user = cursor.execute("SELECT * FROM user WHERE username = ?", (username_input,)).fetchone()
                    if user and check_password_hash(user['password'], password_input):
                        st.session_state.logged_in = True
                        st.session_state.username = user['username']
                        st.session_state.user_id = user['id']
                        conn.close()
                        st.success("Successfully logged in!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                    conn.close()
                else: # Register
                    existing = cursor.execute("SELECT * FROM user WHERE username = ?", (username_input,)).fetchone()
                    if existing:
                        st.error("Username already exists.")
                        conn.close()
                    else:
                        hashed = generate_password_hash(password_input, method='pbkdf2:sha256')
                        cursor.execute("INSERT INTO user (username, password) VALUES (?, ?)", (username_input, hashed))
                        conn.commit()
                        new_user = cursor.execute("SELECT * FROM user WHERE username = ?", (username_input,)).fetchone()
                        st.session_state.logged_in = True
                        st.session_state.username = new_user['username']
                        st.session_state.user_id = new_user['id']
                        conn.close()
                        st.success("Registration successful!")
                        st.rerun()
    st.stop()

# ─── LOGGED IN USER INTERFACE ────────────────────────────────────────

# Sidebar Configuration & Chat History
with st.sidebar:
    st.markdown(f"### 👤 Welcome, **{st.session_state.username}**")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_id = None
        st.session_state.current_session_id = None
        st.session_state.tts_text = None
        st.rerun()
        
    st.divider()
    
    st.subheader("⚙️ Chat Settings")
    
    # API Key Input
    if not api_key:
        api_key_val = st.text_input(
            "Enter Gemini API Key", 
            value=st.session_state.user_api_key or "", 
            type="password", 
            help="Get a free key from Google AI Studio"
        )
        if api_key_val:
            val_clean = api_key_val.strip().strip("'\"")
            if val_clean == "AIzaSyBqU-Tbo_eIUOEY76pKyMuJU0yTFiTNVoA":
                st.sidebar.error("⚠️ This API key is leaked and blocked by Google. Please get a new key.")
                st.session_state.user_api_key = None
            else:
                st.session_state.user_api_key = val_clean
                genai.configure(api_key=st.session_state.user_api_key, transport='rest')

            
    current_role = st.selectbox("Assistant Persona / Role", list(ROLES.keys()), index=0)
    current_lang = st.selectbox("Preferred Language", ["Hindi", "English", "Sanskrit", "Tamil", "Bhojpuri"], index=0)

    # API key status indicator (masked for security)
    active_k = api_key or st.session_state.user_api_key
    if active_k:
        masked_key = f"{active_k[:4]}...{active_k[-4:]}" if len(active_k) > 8 else "..."
        st.caption(f"🔑 Key Loaded: `{masked_key}` (length: {len(active_k)})")
    else:
        st.caption("⚠️ No Gemini API Key loaded. Please configure one above.")



    
    st.divider()
    
    # Session Controls
    st.subheader("💬 Chat Sessions")
    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.tts_text = None
        st.rerun()
        
    # Load and display user sessions from DB
    conn = get_db_connection()
    cursor = conn.cursor()
    sessions = cursor.execute(
        "SELECT * FROM chat_session WHERE user_id = ? ORDER BY created_at DESC", 
        (st.session_state.user_id,)
    ).fetchall()
    conn.close()
    
    session_options = {s['id']: s['title'] for s in sessions}
    if session_options:
        # Determine the index for selectbox
        sel_idx = 0
        if st.session_state.current_session_id in session_options:
            sel_idx = list(session_options.keys()).index(st.session_state.current_session_id)
        else:
            # Set to the newest session if current is None
            st.session_state.current_session_id = list(session_options.keys())[0]
            sel_idx = 0
            
        chosen_session_id = st.selectbox(
            "Select Conversation History",
            options=list(session_options.keys()),
            format_func=lambda x: session_options[x],
            index=sel_idx
        )
        if chosen_session_id != st.session_state.current_session_id:
            st.session_state.current_session_id = chosen_session_id
            st.session_state.tts_text = None
            st.rerun()
    else:
        st.caption("No saved chats yet. Start talking to save a session!")

# Main area title
st.markdown('<div class="main-header">rapl<span style="color:#b57452">•</span>ca AI</div>', unsafe_allow_html=True)
st.caption(f"Role: {current_role} | Language: {current_lang}")

# Load active messages
messages_list = []
if st.session_state.current_session_id:
    conn = get_db_connection()
    cursor = conn.cursor()
    msgs_raw = cursor.execute(
        "SELECT * FROM message WHERE session_id = ? ORDER BY timestamp ASC", 
        (st.session_state.current_session_id,)
    ).fetchall()
    conn.close()
    for m in msgs_raw:
        messages_list.append({"role": m['role'], "content": json.loads(m['content'])})

# Render history messages
for msg in messages_list:
    with st.chat_message("user" if msg['role'] == "user" else "assistant"):
        content = msg['content']
        if isinstance(content, list):
            joined_content = "<br>".join(content)
            st.markdown(joined_content, unsafe_allow_html=True)
        else:
            st.markdown(content)

# ─── RAG Document Uploader & FAISS Index ──────────────────────────────
st.divider()
st.subheader("📁 Upload Documents (RAG Search)")
uploaded_file = st.file_uploader(
    "Upload a PDF, TXT, or Image to index in this chat session", 
    type=["pdf", "txt", "png", "jpg", "jpeg"]
)

# Manage index creation
embeddings = get_embeddings()
session_vs_path = os.path.join(VECTOR_STORE_DIR, str(st.session_state.current_session_id or "temp"))

if uploaded_file:
    # We need a session_id to save the FAISS database on disk.
    # Create session now if none exists yet.
    if not st.session_state.current_session_id:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_session (user_id, title) VALUES (?, ?)", 
            (st.session_state.user_id, f"Chat with {uploaded_file.name}")
        )
        conn.commit()
        st.session_state.current_session_id = cursor.lastrowid
        conn.close()
        session_vs_path = os.path.join(VECTOR_STORE_DIR, str(st.session_state.current_session_id))
        st.rerun()

    # Process and parse file
    file_ext = uploaded_file.name.split('.')[-1].lower()
    extracted_text = ""
    attachment_type = None
    attachment = None

    if file_ext in ['jpg', 'jpeg', 'png']:
        attachment = Image.open(uploaded_file)
        attachment_type = "image"
        st.image(attachment, caption="Uploaded Image", width=300)
    else:
        attachment_type = "text"
        if file_ext == "pdf":
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    p_text = page.extract_text()
                    if p_text:
                        extracted_text += p_text + "\n"
            except Exception as e:
                st.error(f"Error parsing PDF: {e}")
        elif file_ext == "txt":
            extracted_text = uploaded_file.read().decode("utf-8")

    # Index text files via FAISS
    if attachment_type == "text" and extracted_text.strip():
        with st.spinner("Indexing document chunks into FAISS vector store..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(extracted_text)
            if chunks:
                if os.path.exists(session_vs_path) and os.listdir(session_vs_path):
                    db_index = FAISS.load_local(session_vs_path, embeddings, allow_dangerous_deserialization=True)
                    db_index.add_texts(chunks)
                    db_index.save_local(session_vs_path)
                else:
                    db_index = FAISS.from_texts(chunks, embeddings)
                    db_index.save_local(session_vs_path)
                st.success(f"✅ Successfully indexed {len(chunks)} text chunks from '{uploaded_file.name}'!")

# ─── Chat Inputs (Text & Voice) ───────────────────────────────────────
st.divider()
st.subheader("🗣️ Say or Type your Message")

# Voice input widget
audio_query = st.audio_input("Record a voice query")

# Text input widget
user_text_input = st.chat_input("Ask a question...")

query_to_process = None
is_audio = False

# Process input channels
if audio_query:
    audio_bytes = audio_query.read()
    if audio_bytes:
        # Check API Key before transcription
        active_key = api_key or st.session_state.user_api_key
        if not active_key:
            st.error("⚠️ Gemini API Key not found. Please enter it in the sidebar.")
            st.stop()
        else:
            genai.configure(api_key=active_key, transport='rest')
            
        # Transcribe audio using Gemini 1.5/2.0

        with st.spinner("🎙️ Transcribing voice recording..."):

            try:
                # Setup multimodal prompt to transcribe audio query
                transcribe_model = genai.GenerativeModel('gemini-1.5-flash')
                response = transcribe_model.generate_content([
                    {"mime_type": "audio/wav", "data": audio_bytes},
                    "Provide a clean transcription of this audio query in the spoken language. Do not add any commentary."
                ])
                query_to_process = response.text.strip()
                is_audio = True
                st.info(f"🎙️ **Transcribed Voice Query:** '{query_to_process}'")
            except Exception as e:
                st.error(f"Voice Transcription Error: {e}")

if user_text_input:
    query_to_process = user_text_input.strip()

# ─── PROCESS CHAT RESPONSE ───────────────────────────────────────────
if query_to_process:
    # Ensure key is set and library is configured
    active_key = api_key or st.session_state.user_api_key
    if not active_key:
        st.error("⚠️ Gemini API Key not found. Please enter it in the sidebar.")
        st.stop()
    else:
        genai.configure(api_key=active_key, transport='rest')

        
    # 1. Manage session if none exists

    if not st.session_state.current_session_id:
        conn = get_db_connection()
        cursor = conn.cursor()
        title = query_to_process[:30] + "..." if query_to_process else "New Chat"
        cursor.execute("INSERT INTO chat_session (user_id, title) VALUES (?, ?)", (st.session_state.user_id, title))
        conn.commit()
        st.session_state.current_session_id = cursor.lastrowid
        conn.close()
        session_vs_path = os.path.join(VECTOR_STORE_DIR, str(st.session_state.current_session_id))

    # 2. Perform FAISS retrieval
    has_vector_store = os.path.exists(session_vs_path) and os.listdir(session_vs_path)
    retrieved_context = ""
    if has_vector_store:
        try:
            db_index = FAISS.load_local(session_vs_path, embeddings, allow_dangerous_deserialization=True)
            docs = db_index.similarity_search(query_to_process, k=4)
            retrieved_context = "\n\n".join(doc.page_content for doc in docs)
        except Exception as e:
            st.error(f"RAG search error: {e}")

    # Build prompt elements
    user_parts = []
    if retrieved_context:
        prompt_with_context = (
            "Use the following pieces of context to answer the question at the end.\n"
            "If you don't know the answer or if the context doesn't contain the answer, "
            "say that you don't know based on the provided documents.\n\n"
            f"Context:\n{retrieved_context}\n\n"
            f"Question: {query_to_process}"
        )
        user_parts.append(prompt_with_context)
    else:
        user_parts.append(query_to_process)

    # 3. Assemble database entry elements
    db_parts = []
    if is_audio:
        db_parts.append("[🎙️ Voice message]")
    if uploaded_file:
        if file_ext in ['jpg', 'jpeg', 'png']:
            db_parts.append("[🖼️ Image uploaded]")
            user_parts.append(attachment)
        else:
            db_parts.append(f"[📎 Document: {uploaded_file.name}]")
    db_parts.append(query_to_process)

    # Save user message to SQLite DB
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO message (session_id, role, content) VALUES (?, ?, ?)", 
        (st.session_state.current_session_id, "user", json.dumps(db_parts))
    )
    conn.commit()
    conn.close()

    # Render user query right away
    with st.chat_message("user"):
        st.markdown("<br>".join(db_parts), unsafe_allow_html=True)

    # 4. Generate Gemini response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("🤖 Thinking..."):
            # Load conversation history context
            conn = get_db_connection()
            cursor = conn.cursor()
            previous_messages = cursor.execute(
                "SELECT * FROM message WHERE session_id = ? ORDER BY timestamp ASC", 
                (st.session_state.current_session_id,)
            ).fetchall()
            conn.close()

            gemini_messages = []
            for m in previous_messages:
                # Fetch last user message directly via Python query_parts instead of database serialization
                if m['role'] == 'user':
                    continue
                gemini_messages.append({"role": "model", "parts": json.loads(m['content'])})

            # Append current query
            gemini_messages.append({"role": "user", "parts": user_parts})

            # Base system framing
            system_prompt = ROLES.get(current_role, ROLES["Student"])
            system_prompt += f"\n\nIMPORTANT: You MUST respond entirely in the {current_lang} language."

            try:
                model = genai.GenerativeModel('gemini-flash-lite-latest', system_instruction=system_prompt)
                generation_config = {
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "max_output_tokens": 2048
                }
                response = model.generate_content(gemini_messages, generation_config=generation_config)
                reply = response.text
                
                # Render response
                message_placeholder.markdown(reply)

                # Save response to SQLite
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO message (session_id, role, content) VALUES (?, ?, ?)", 
                    (st.session_state.current_session_id, "model", json.dumps([reply]))
                )
                conn.commit()
                conn.close()

                # Trigger Text-to-Speech client-side voice
                st.session_state.tts_text = reply
                st.rerun()

            except Exception as e:
                message_placeholder.error(f"Error calling model: {e}")

# ─── Client-side Text-to-Speech (TTS) Execution ─────────────────────
if st.session_state.tts_text:
    js_text = st.session_state.tts_text.replace("'", "\\'").replace("\n", " ")
    # Pick Hindi/English voice parameters
    voice_lang = 'hi-IN' if current_lang == 'Hindi' or current_lang == 'Bhojpuri' else 'en-US'
    
    st.components.v1.html(
        f"""
        <script>
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance('{js_text}');
            msg.lang = '{voice_lang}';
            msg.pitch = 1.1; // Feminine default pitch
            window.speechSynthesis.speak(msg);
        }}
        </script>
        """,
        height=0,
    )
    # Clear the text after triggering speech
    st.session_state.tts_text = None
