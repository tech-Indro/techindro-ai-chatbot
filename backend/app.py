from flask import Flask, request, jsonify, send_from_directory, session
import os
import io
import json
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import PyPDF2
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# RAG imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    api_key = api_key.strip().strip("'\"")
    if api_key in ["YOUR_GEMINI_API_KEY_HERE", "your_api_key_here", "AIzaSyBqU-Tbo_eIUOEY76pKyMuJU0yTFiTNVoA", ""] or len(api_key) < 15:
        api_key = None
    else:
        genai.configure(api_key=api_key, transport='rest')


app = Flask(__name__, static_folder="../frontend", static_url_path="/")
app.config['SECRET_KEY'] = 'supersecretkey123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

# Initialize Hugging Face embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    sessions = db.relationship('ChatSession', backref='user', lazy=True)

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='session', lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    role = db.Column(db.String(50), nullable=False) # 'user' or 'model'
    content = db.Column(db.Text, nullable=False) # JSON serialized parts
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

with app.app_context():
    db.create_all()

ROLES = {
    "Student": "You are a helpful AI assistant for students. Answer study-related queries in simple language.",
    "Teacher": "You are an AI assistant for teachers. Help with lesson plans, assessments, and teaching tips.",
    "Farmer": "You are an AI assistant for farmers. Give advice on crops, weather, and best practices.",
    "Doctor": "You are an AI assistant for doctors. Provide medical references and research support. Do not give direct diagnoses.",
    "Women": "You are an AI assistant for women's support. Give advice on health, career, safety, and empowerment."
}

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# --- AUTH ROUTES ---
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    if User.query.filter_by(username=data.get("username")).first():
        return jsonify({"error": "Username already exists"}), 400
    hashed_password = generate_password_hash(data.get("password"), method='pbkdf2:sha256')
    new_user = User(username=data.get("username"), password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    login_user(new_user)
    return jsonify({"success": True, "username": new_user.username})

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    user = User.query.filter_by(username=data.get("username")).first()
    if user and check_password_hash(user.password, data.get("password")):
        login_user(user)
        return jsonify({"success": True, "username": user.username})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/api/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"success": True})

@app.route("/api/me", methods=["GET"])
def me():
    if current_user.is_authenticated:
        return jsonify({"logged_in": True, "username": current_user.username})
    return jsonify({"logged_in": False})

# --- HISTORY ROUTES ---
@app.route("/api/sessions", methods=["GET"])
@login_required
def get_sessions():
    sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.created_at.desc()).all()
    return jsonify([{"id": s.id, "title": s.title, "created_at": s.created_at.isoformat()} for s in sessions])

@app.route("/api/sessions/<int:session_id>", methods=["GET"])
@login_required
def get_session_messages(session_id):
    session = ChatSession.query.get_or_404(session_id)
    if session.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    messages = Message.query.filter_by(session_id=session.id).order_by(Message.timestamp.asc()).all()
    return jsonify([{"role": m.role, "content": json.loads(m.content)} for m in messages])

# --- CHAT ROUTE ---
@app.route("/chat", methods=["POST"])
def chat():
    text = request.form.get("message", "").strip()
    role = request.form.get("role", "Student")
    language = request.form.get("language", "Hindi")
    session_id = request.form.get("session_id")
    file = request.files.get("file")
    
    if not text and not file:
        return jsonify({"error": "Empty message"}), 400

    if not current_user.is_authenticated:
        return jsonify({"error": "Must be logged in to chat"}), 401

    # 1. Manage/Create Session first so we have chat_session.id
    if session_id and session_id != 'null':
        chat_session = db.session.get(ChatSession, int(session_id))
        if not chat_session or chat_session.user_id != current_user.id:
            return jsonify({"error": "Invalid session"}), 403
    else:
        # Create new session
        title = text[:30] + "..." if text else "New Chat with File"
        chat_session = ChatSession(user_id=current_user.id, title=title)
        db.session.add(chat_session)
        db.session.commit()

    attachment = None
    attachment_type = None
    extracted_text = ""
    filename = ""
    
    if file:
        filename = secure_filename(file.filename)
        file_ext = filename.split('.')[-1].lower()
        file_bytes = file.read()
        
        if file_ext in ['jpg', 'jpeg', 'png']:
            attachment = Image.open(io.BytesIO(file_bytes))
            attachment_type = "image"
        elif file_ext == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
            attachment_type = "text"
        elif file_ext == 'txt':
            extracted_text = file_bytes.decode("utf-8")
            attachment_type = "text"

    # 2. If document text is extracted, chunk it and create/update session-specific FAISS index
    session_vs_path = os.path.join(app.instance_path, "vector_stores", str(chat_session.id))
    if attachment_type == "text" and extracted_text.strip():
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(extracted_text)
        if chunks:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(session_vs_path), exist_ok=True)
            
            # Check if index already exists
            if os.path.exists(session_vs_path) and os.listdir(session_vs_path):
                db_index = FAISS.load_local(session_vs_path, embeddings, allow_dangerous_deserialization=True)
                db_index.add_texts(chunks)
                db_index.save_local(session_vs_path)
            else:
                db_index = FAISS.from_texts(chunks, embeddings)
                db_index.save_local(session_vs_path)

    # 3. Build database parts to save (simple & lightweight)
    db_user_parts = []
    if file:
        if attachment_type == "image":
            db_user_parts.append("[Image attached]")
        else:
            db_user_parts.append(f"[Document attached: {filename}]")
    if text:
        db_user_parts.append(text)

    # Save user message to database
    user_msg = Message(session_id=chat_session.id, role="user", content=json.dumps(db_user_parts))
    db.session.add(user_msg)
    db.session.commit()

    # 4. Perform similarity search if vector store exists
    has_vector_store = os.path.exists(session_vs_path) and os.listdir(session_vs_path)
    
    # We retrieve context based on the current user query `text`
    retrieved_context = ""
    if has_vector_store and text:
        try:
            db_index = FAISS.load_local(session_vs_path, embeddings, allow_dangerous_deserialization=True)
            docs = db_index.similarity_search(text, k=4)
            retrieved_context = "\n\n".join(doc.page_content for doc in docs)
        except Exception as e:
            app.logger.error(f"Error loading or searching vector store: {e}")

    # Build prompt/parts for Gemini
    user_parts = []
    if retrieved_context:
        prompt_with_context = (
            "Use the following pieces of context to answer the question at the end.\n"
            "If you don't know the answer or if the context doesn't contain the answer, "
            "say that you don't know based on the provided documents.\n\n"
            f"Context:\n{retrieved_context}\n\n"
            f"Question: {text}"
        )
        user_parts.append(prompt_with_context)
    else:
        if text:
            user_parts.append(text)

    if attachment_type == "image" and attachment:
        user_parts.append(attachment)

    # 5. Fetch previous messages for Gemini context
    previous_messages = Message.query.filter_by(session_id=chat_session.id).order_by(Message.timestamp.asc()).all()
    gemini_messages = []
    for m in previous_messages:
        # Skip the message we just added to send the context-augmented version instead
        if m.id == user_msg.id:
            continue
        gemini_messages.append({"role": "user" if m.role == "user" else "model", "parts": json.loads(m.content)})
    
    # Append current augmented query/parts
    gemini_messages.append({"role": "user", "parts": user_parts})

    system_prompt = ROLES.get(role, ROLES["Student"])
    system_prompt += f"\n\nIMPORTANT: You MUST respond entirely in the {language} language."
    model = genai.GenerativeModel('gemini-flash-lite-latest', system_instruction=system_prompt)
    
    try:
        # Optimize generation parameters for factual, context-aligned RAG output
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "max_output_tokens": 2048
        }
        response = model.generate_content(gemini_messages, generation_config=generation_config)
        reply = response.text

        
        # Save model response
        model_msg = Message(session_id=chat_session.id, role="model", content=json.dumps([reply]))
        db.session.add(model_msg)
        db.session.commit()

        return jsonify({"reply": reply, "session_id": chat_session.id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
