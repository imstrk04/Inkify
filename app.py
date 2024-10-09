from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from application.models import User, TextToHandwriting  # Import your models
from application.database import db  # Assuming your db is initialized here
import secrets
from flask_mail import Mail, Message
app = Flask(__name__)
import os
from dotenv import load_dotenv

load_dotenv()

# Configuring the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///inkify.db'  # Replace with your database path if needed
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable track modifications
app.secret_key = 'd4&g!kT@8#eZ$3v%X*2hM^nQ+5pF1^cW'

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # or your mail server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = 'okot iuoo tfis jerr'
app.config['MAIL_DEFAULT_SENDER'] = 'sadakopa2210221@ssn.edu.in'  # Default sender email

mail = Mail(app)

# Initialize the database with the Flask app
db.init_app(app)

# Create tables if they don't exist
with app.app_context():
    db.create_all()
    print("Tables created")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Create a new user instance
        new_user = User(
            username=username,
            email=email,
        )
        
        # Set the hashed password
        new_user.set_password(password)

        # Add the new user to the session and commit it
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    elif request.method == 'GET':
        return render_template("register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "GET":
        return render_template("login.html")
    elif request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        remember = 'remember' in request.form

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            # Generate OTP
            otp = secrets.randbelow(1000000)  # Generate a 6-digit OTP
            session['otp'] = otp  # Store OTP in session
            session['user_id'] = user.user_id  # Store user ID in session
            
            # Send OTP to user's email
            msg = Message('Ikify - Login OTP Code', recipients=[email])
            msg.body = f'Your OTP code is: {otp}'
            mail.send(msg)
            
            flash('An OTP has been sent to your email. Please enter it to continue.', 'info')
            return redirect(url_for('verify_otp'))  # Redirect to OTP verification page

        else:
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))

# OTP verification route
@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == "GET":
        return render_template("verify_otp.html")  # Create a separate template for OTP verification
    elif request.method == 'POST':
        entered_otp = request.form['otp']
        if int(entered_otp) == session.get('otp'):
            # OTP is valid
            user_id = session.get('user_id')
            # Log the user in, create a session, etc.
            session.pop('otp', None)  # Clear the OTP from session
            flash('You are now logged in!', 'success')
            return redirect(url_for('home'))  # Redirect to the home page after successful login
        else:
            flash('Invalid OTP. Please try again.', 'danger')
            return redirect(url_for('verify_otp'))  # Redirect back to OTP verification

@app.route('/home', methods = ['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template("home.html")

@app.route('/logout')
def logout():
    # Remove the user from the session
    session.pop('user_id', None)  
    return redirect(url_for('login')) 


if __name__ == "__main__":
    app.run(debug=True)
