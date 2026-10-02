# app.py
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy import desc

# Local imports
from models import db, User, Message, Conversation
from rag_chain_handler import create_rag_chain

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-super-secret-key'

# --- START: DATABASE AND FOLDER FIX ---
# This block ensures the 'instance' folder exists where the database will be stored.
# It's a permanent fix for the 'unable to open database file' error.
instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
os.makedirs(instance_dir, exist_ok=True)

# Configure the database URI to point to a file inside the instance folder
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_dir, "users.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# --- END: DATABASE AND FOLDER FIX ---

# --- UPLOAD CONFIGURATION ---
UPLOAD_FOLDER = 'static/profile_pics'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- LOAD THE RAG CHAIN ONCE ON STARTUP ---
rag_chain = create_rag_chain()
if not rag_chain:
    print("FATAL: Could not initialize RAG chain.")

# --- AUTHENTICATION ROUTES ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))
        login_user(user)
        return redirect(url_for('chat'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))

    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not email or not first_name or not last_name or not password or not confirm_password:
             flash('All fields except profile picture are required.', 'error')
             return redirect(url_for('signup'))

        if not email.endswith('@gmail.com'):
            flash('Please use a valid @gmail.com address.', 'error')
            return redirect(url_for('signup'))

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists.', 'error')
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('signup'))

        profile_pic_filename = 'default.png'
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = str(uuid.uuid4()) + "_" + filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                profile_pic_filename = unique_filename
            elif file.filename != '':
                flash('Invalid file type for profile picture. Allowed types are png, jpg, jpeg, gif.', 'error')
                return redirect(url_for('signup'))

        new_user = User(
            email=email,
            first_name=first_name,
            last_name=last_name,
            password=generate_password_hash(password, method='pbkdf2:sha256'),
            profile_pic=profile_pic_filename
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# --- CORE APP ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/chat')
@login_required
def chat():
    conversations = Conversation.query.filter_by(user_id=current_user.id).order_by(desc(Conversation.timestamp)).all()
    return render_template('chat.html', user=current_user, conversations=conversations, current_conv_id=None, history=[])

@app.route('/chat/<int:conversation_id>')
@login_required
def conversation(conversation_id):
    conv = Conversation.query.get_or_404(conversation_id)
    if conv.user_id != current_user.id:
        flash("You do not have access to this conversation.")
        return redirect(url_for('chat'))
    
    conversations = Conversation.query.filter_by(user_id=current_user.id).order_by(desc(Conversation.timestamp)).all()
    history = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp).all()
    
    return render_template('chat.html', user=current_user, conversations=conversations, current_conv_id=conversation_id, history=history)

# --- API ENDPOINT (Unchanged) ---
@app.route('/api/ask', methods=['POST'])
@login_required
def ask():
    user_input = request.json.get('message')
    conversation_id = request.json.get('conversation_id')
    
    if not rag_chain or not user_input:
        return jsonify({"error": "Invalid request"}), 400

    if not conversation_id:
        new_title = user_input[:50]
        new_conv = Conversation(title=new_title, user_id=current_user.id)
        db.session.add(new_conv)
        db.session.commit()
        conversation_id = new_conv.id
        chat_history = []
    else:
        db_history = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp).all()
        chat_history = [
            HumanMessage(content=msg.content) if msg.sender == 'user' else AIMessage(content=msg.content)
            for msg in db_history
        ]
        
    try:
        user_message = Message(content=user_input, sender='user', conversation_id=conversation_id)
        db.session.add(user_message)

        response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
        ai_response = response.get("answer", "Sorry, I couldn't process that.")

        ai_message = Message(content=ai_response, sender='ai', conversation_id=conversation_id)
        db.session.add(ai_message)

        db.session.commit()
        
        return jsonify({"answer": ai_response, "conversation_id": conversation_id})

    except Exception as e:
        db.session.rollback()
        print(f"Error in /api/ask: {e}")
        return jsonify({"error": "An error occurred."}), 500

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    with app.app_context():
        db.create_all()
    app.run(debug=True)