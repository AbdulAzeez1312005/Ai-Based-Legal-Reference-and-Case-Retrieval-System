"""Flask API for the AI-Based Legal Reference and Case Retrieval System."""
from __future__ import annotations

import base64
import binascii
import logging
import os
import re
from datetime import datetime
from functools import lru_cache, wraps
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

from legal_rag import extract_provision_references, retrieve_legal_context

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".cache" / "huggingface"))
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "local-development-key-change-me"),
    SQLALCHEMY_DATABASE_URI=os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'app.db').as_posix()}"),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
allowed_origins = {
    os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}
CORS(app, supports_credentials=True, origins=sorted(allowed_origins))
db = SQLAlchemy(app)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    conversations = db.relationship(
        "Conversation", backref="user", cascade="all, delete-orphan"
    )
    profile = db.relationship(
        "UserProfile", backref="user", cascade="all, delete-orphan", uselist=False
    )


class UserProfile(db.Model):
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), primary_key=True)
    photo_mime = db.Column(db.String(20), nullable=True)
    photo_data = db.Column(db.LargeBinary, nullable=True)


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False, default="New research chat")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    messages = db.relationship(
        "Message", backref="conversation", cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversation.id"), nullable=False)


def error_response(message: str, status: int = 400):
    return jsonify(error=message), status


def require_user(handler):
    @wraps(handler)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return error_response("Please sign in to continue.", 401)
        return handler(*args, **kwargs)
    return wrapped


def serialize_user(user: User) -> dict:
    profile = user.profile
    photo = None
    if profile and profile.photo_data and profile.photo_mime:
        photo = (
            f"data:{profile.photo_mime};base64,"
            f"{base64.b64encode(profile.photo_data).decode('ascii')}"
        )
    return {"id": user.id, "name": user.name, "email": user.email, "photo": photo}


def decode_profile_photo(value: object) -> tuple[bytes, str] | None:
    if not isinstance(value, str):
        raise ValueError("Choose a PNG, JPEG, or WebP profile photo.")
    match = re.fullmatch(
        r"data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/]*={0,2})",
        value,
    )
    if not match:
        raise ValueError("Choose a PNG, JPEG, or WebP profile photo.")
    mime_type, encoded = match.groups()
    if len(encoded) > 1_400_000:
        raise ValueError("Profile photos must be 1 MB or smaller.")
    try:
        photo_data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The selected profile photo could not be read.") from exc
    if not photo_data or len(photo_data) > 1_000_000:
        raise ValueError("Profile photos must be 1 MB or smaller.")
    valid_signature = {
        "image/png": photo_data.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": photo_data.startswith(b"\xff\xd8\xff"),
        "image/webp": (
            len(photo_data) >= 12
            and photo_data.startswith(b"RIFF")
            and photo_data[8:12] == b"WEBP"
        ),
    }[mime_type]
    if not valid_signature:
        raise ValueError("The selected file does not match its image type.")
    return photo_data, mime_type


def serialize_conversation(conversation: Conversation, include_messages: bool = False) -> dict:
    value = {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
    }
    if include_messages:
        value["messages"] = [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in conversation.messages
        ]
    return value


@lru_cache(maxsize=1)
def initialize_rag():
    """Load the local embedding model and connect to the configured Pinecone index."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "legal-reference-m4")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing from backend/.env.")
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer

    client = Pinecone(api_key=api_key)
    index = client.Index(index_name)
    embedding_model = SentenceTransformer(
        os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    return index, embedding_model


def _ollama_messages(question: str, context: str, history: list[Message]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{
        "role": "system",
        "content": (
            "You are an Indian legal research assistant. Treat the retrieved excerpts as "
            "the only authority for this answer. Do not use prior legal knowledge to fill "
            "missing text. Never invent a section, quotation, source, page, condition, "
            "exception, or case holding. Preserve the source's exact legal force: do not "
            "change 'shall' to 'may', 'may' to 'shall', or alter a negation. Treat 'shall' "
            "as mandatory and 'may' as discretionary; never describe an action expressed "
            "with 'shall' as optional. If both terms occur, explain their effects separately. "
            "Keep provisos, "
            "exceptions, and their scope attached to the rule they qualify. If a passage "
            "starts or ends mid-provision, says 'provided that', refers to an omitted clause, "
            "or otherwise lacks surrounding text needed to decide the question, say the "
            "context is incomplete and limit the conclusion accordingly. If the retrieved "
            "text does not support the requested detail, say that plainly. Distinguish a "
            "statutory quotation from your summary. Quote the provision faithfully; do not "
            "silently repair or omit legally meaningful words. Cite every source using only "
            "the exact source names and page numbers shown in the context, and never guess "
            "a page. The application verifies the final citations against retrieval metadata.\n\n"
            "Format the answer with these headings: Relevant legal provisions, Legal summary, "
            "Citations & sources. Quote only the shortest passage that directly supports the "
            "answer. Keep the answer focused on the question."
        ),
    }]
    for message in history[-8:]:
        if message.role in {"user", "assistant"}:
            messages.append({"role": message.role, "content": message.content})
    messages.append({
        "role": "user",
        "content": f"Retrieved legal context:\n{context}\n\nQuestion:\n{question}",
    })
    return messages


def _with_verified_citations(answer: str, passages: list[dict]) -> str:
    source_pages: dict[str, set[int]] = {}
    for passage in passages:
        metadata = passage["metadata"]
        source = str(metadata.get("source", "Legal source"))
        pages = metadata.get("pages") or [metadata.get("page")]
        page_numbers = {
            int(page) for page in pages
            if page is not None
        }
        source_pages.setdefault(source, set()).update(page_numbers)
    citation_lines = []
    for source, pages in source_pages.items():
        page_label = "page" if len(pages) == 1 else "pages"
        page_text = ", ".join(str(page) for page in sorted(pages))
        citation_lines.append(
            f"- {source}" + (f", {page_label} {page_text}" if page_text else "")
        )
    citation_heading = re.compile(
        r"(?im)^\s*#{0,3}\s*Citations\s*(?:&|and)\s*sources\b.*$"
    )
    body = citation_heading.split(answer, maxsplit=1)[0].rstrip()
    citations = "\n".join(citation_lines)
    return f"{body}\n\nCitations & sources\n{citations}"


def generate_answer(question: str, history: list[Message] | None = None) -> tuple[str, list[dict]]:
    index, embedder = initialize_rag()
    history = history or []
    search_question = question
    if not extract_provision_references(question) and history:
        prior_user_questions = [
            item.content for item in history[-8:] if item.role == "user"
        ][-3:]
        if prior_user_questions:
            search_question = "\n".join(prior_user_questions + [question])

    passages, trace = retrieve_legal_context(
        search_question,
        index,
        embedder,
        namespace=os.getenv("PINECONE_NAMESPACE", ""),
        top_k=int(os.getenv("RETRIEVAL_TOP_K", "6")),
    )
    if not passages:
        ambiguity = next(
            (item for item in trace if item.get("mode") == "ambiguous-provision"),
            None,
        )
        if ambiguity:
            sources = ", ".join(str(source) for source in ambiguity["sources"])
            return (
                f'{ambiguity["provision"]} appears in multiple indexed sources: '
                f"{sources}. Please specify the Act or document you mean.",
                trace,
            )
        return (
            "The requested information was not found in the provided documents. "
            "Check that the documents have been ingested into the configured Pinecone "
            "index, or rephrase the question with the Act, case, section, or article name.",
            trace,
        )

    max_chars = max(4000, int(os.getenv("RAG_CONTEXT_CHARS", "20000")))
    context_parts: list[str] = []
    used = 0
    was_truncated = False
    for position, result in enumerate(passages, start=1):
        metadata = result["metadata"]
        source = metadata.get("source", "Legal source")
        page = metadata.get("page")
        provision = metadata.get("provision_type")
        number = metadata.get("provision_number")
        title = metadata.get("provision_title")
        label = f"{provision.title()} {number}" if provision and number else ""
        if title:
            label += f" - {title}"
        header = f"[{position}] Source: {source}"
        pages = metadata.get("pages") or ([page] if page else [])
        pages = sorted({int(value) for value in pages if value is not None})
        if len(pages) == 1:
            header += f", page {pages[0]}"
        elif pages:
            header += f", pages {', '.join(str(value) for value in pages)}"
        if label:
            header += f"; provision: {label}"
        passage = result["page_content"].strip()
        room = max_chars - used - len(header) - 2
        if room <= 0:
            was_truncated = True
            break
        if len(passage) > room:
            passage = passage[:room]
            was_truncated = True
        context_parts.append(f"{header}\n{passage}")
        used += len(header) + len(passage) + 2
        if was_truncated:
            break
    context = "\n\n".join(context_parts)
    if was_truncated:
        context += "\n\n[Application note: retrieved context was shortened; it may be incomplete.]"

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    payload = {
        "model": model,
        "messages": _ollama_messages(question, context, history),
        "stream": False,
        "options": {
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "16384")),
        },
    }
    response = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "240")),
    )
    response.raise_for_status()
    answer = (response.json().get("message") or {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")
    return _with_verified_citations(answer, passages), trace


@app.get("/api/health")
def health():
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    ollama_available = False
    try:
        response = requests.get(
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/tags",
            timeout=1.5,
        )
        ollama_available = response.ok
    except requests.RequestException:
        pass
    return jsonify(
        status="ok",
        pinecone_configured=bool(os.getenv("PINECONE_API_KEY")),
        pinecone_index=os.getenv("PINECONE_INDEX", "legal-reference-m4"),
        ollama_available=ollama_available,
        ollama_model=ollama_model,
    )


@app.post("/api/auth/signup")
def signup():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    if not name or not email or len(password) < 8:
        return error_response("Enter your name and email, and use a password of at least 8 characters.")
    if User.query.filter_by(email=email).first():
        return error_response("That email address is already registered.", 409)
    user = User(name=name, email=email, password_hash=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()
    session.clear()
    session["user_id"] = user.id
    return jsonify(user=serialize_user(user)), 201


@app.post("/api/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return error_response("Email or password is incorrect.", 401)
    session.clear()
    session["user_id"] = user.id
    return jsonify(user=serialize_user(user))


@app.post("/api/auth/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


@app.get("/api/auth/me")
@require_user
def current_user():
    user = db.session.get(User, session["user_id"])
    if user is None:
        session.clear()
        return error_response("Your session has expired. Please sign in again.", 401)
    return jsonify(user=serialize_user(user))


@app.put("/api/auth/profile")
@require_user
def update_profile():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return error_response("Invalid profile update.")
    name = str(data.get("name", "")).strip()
    if not name or len(name) > 120:
        return error_response("Enter a name of 1 to 120 characters.")
    remove_photo = data.get("remove_photo", False)
    if not isinstance(remove_photo, bool):
        return error_response("Invalid profile photo update.")
    photo_update = None
    if "photo" in data:
        try:
            photo_update = decode_profile_photo(data["photo"])
        except ValueError as exc:
            return error_response(str(exc))
        if remove_photo:
            return error_response("Choose a photo or remove the current photo, not both.")

    user = db.session.get(User, session["user_id"])
    if user is None:
        session.clear()
        return error_response("Your session has expired. Please sign in again.", 401)
    user.name = name
    if photo_update:
        profile = user.profile or UserProfile(user=user)
        profile.photo_data, profile.photo_mime = photo_update
        db.session.add(profile)
    elif remove_photo and user.profile:
        db.session.delete(user.profile)
    db.session.commit()
    return jsonify(user=serialize_user(user))


@app.get("/api/conversations")
@require_user
def list_conversations():
    conversations = (
        Conversation.query.filter_by(user_id=session["user_id"])
        .order_by(Conversation.created_at.desc()).all()
    )
    return jsonify(conversations=[serialize_conversation(item) for item in conversations])


@app.post("/api/conversations")
@require_user
def create_conversation():
    conversation = Conversation(user_id=session["user_id"])
    db.session.add(conversation)
    db.session.commit()
    return jsonify(conversation=serialize_conversation(conversation, True)), 201


@app.get("/api/conversations/<int:conversation_id>")
@require_user
def get_conversation(conversation_id: int):
    conversation = Conversation.query.filter_by(
        id=conversation_id, user_id=session["user_id"]
    ).first()
    if conversation is None:
        return error_response("Conversation not found.", 404)
    return jsonify(conversation=serialize_conversation(conversation, True))


@app.delete("/api/conversations/<int:conversation_id>")
@require_user
def delete_conversation(conversation_id: int):
    conversation = Conversation.query.filter_by(
        id=conversation_id, user_id=session["user_id"]
    ).first()
    if conversation is None:
        return error_response("Conversation not found.", 404)
    db.session.delete(conversation)
    db.session.commit()
    return jsonify(ok=True)


@app.post("/api/conversations/<int:conversation_id>/messages")
@require_user
def ask_question(conversation_id: int):
    conversation = Conversation.query.filter_by(
        id=conversation_id, user_id=session["user_id"]
    ).first()
    if conversation is None:
        return error_response("Conversation not found.", 404)
    question = str((request.get_json(silent=True) or {}).get("message", "")).strip()
    if not question:
        return error_response("Enter a question before sending.")
    history = list(conversation.messages)
    try:
        db.session.add(Message(
            role="user", content=question, conversation_id=conversation.id
        ))
        if conversation.title == "New research chat":
            conversation.title = question[:64]
        db.session.commit()
        answer, trace = generate_answer(question, history)
        db.session.add(Message(
            role="assistant", content=answer, conversation_id=conversation.id
        ))
        db.session.commit()
        app.logger.info("RAG retrieval trace: %s", trace)
        return jsonify(
            answer=answer,
            conversation=serialize_conversation(conversation),
            retrieval=trace,
        )
    except Exception:
        db.session.rollback()
        app.logger.exception("Question answering failed")
        return error_response(
            "The answer could not be generated. Check Ollama, Pinecone, and the "
            "embedding model, then retry.", 503,
        )


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"},
    )
