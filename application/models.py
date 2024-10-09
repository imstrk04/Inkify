from .database import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# User Model
class User(db.Model):
    __tablename__ = 'user'
    
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)  # Adjusted to match the new schema
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Default is current timestamp

    text_to_handwriting = db.relationship('TextToHandwriting', backref='user', lazy=True)

    # Method to hash and set password
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    # Method to check password
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"

# TextToHandwriting Model
class TextToHandwriting(db.Model):
    __tablename__ = 'text_to_handwriting'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    input_text = db.Column(db.Text, nullable=False)
    handwriting_image = db.Column(db.LargeBinary)  # BLOB to store image
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<TextToHandwriting {self.id} for User {self.user_id}>"
