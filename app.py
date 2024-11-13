from flask import Flask, request, render_template, redirect, url_for, flash, session, send_file
from flask_sqlalchemy import SQLAlchemy
from application.models import User, TextToHandwriting  # Import your models
from application.database import db  # Assuming your db is initialized here
import secrets
from flask_mail import Mail, Message
from functools import wraps
import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import requests
import io
import pywhatkit as pwk
import cv2
import numpy as np
import pytesseract
import base64
import fpdf
import os
import pywhatkit as kit
from difflib import SequenceMatcher
import tempfile
import re
from collections import defaultdict
from reportlab.pdfgen import canvas as pdf_canvas
import pickle
import random
from werkzeug.utils import secure_filename

app = Flask(__name__)
load_dotenv()

# Configuring the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///inkify.db'  # Replace with your database path if needed
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable track modifications
app.secret_key = 'd4&g!kT@8#eZ$3v%X*2hM^nQ+5pF1^cW'


app.config['UPLOAD_FOLDER'] = './uploads'
app.config['FONT_FOLDER'] = './fonts'

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # or your mail server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = 'okot iuoo tfis jerr'
app.config['MAIL_DEFAULT_SENDER'] = 'sadakopa2210221@ssn.edu.in'  # Default sender email

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

mail = Mail(app)

# Initialize the database with the Flask app
db.init_app(app)

# Create tables if they don't exist
with app.app_context():
    db.create_all()
    print("Tables created")

# Middleware to require login for certain routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to log in first.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# -------------------------- REGISTER PAGE --------------------------
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

# -------------------------- LOGIN PAGE --------------------------

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


@app.route('/logout')
@login_required  # Protect this route with the login_required middleware
def logout():
    # Remove the user from the session
    session.pop('user_id', None)  
    flash('You have been logged out.', 'success')
    return redirect(url_for('login')) 

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Generate OTP
            otp = secrets.randbelow(1000000)  # Generate a 6-digit OTP
            session['reset_otp'] = otp
            session['reset_user_id'] = user.user_id
            
            # Send OTP to user's email
            msg = Message('Inkify - Password Reset OTP', recipients=[email])
            msg.body = f'Your OTP for password reset is: {otp}'
            mail.send(msg)
            
            flash('An OTP has been sent to your email. Please check your inbox.', 'info')
            return redirect(url_for('verify_otp_for_reset'))
        else:
            flash('No account found with this email.', 'danger')
            return redirect(url_for('forgot_password'))
    
    return render_template("forgot_password.html")

@app.route('/verify_otp_for_reset', methods=['GET', 'POST'])
def verify_otp_for_reset():
    if request.method == 'POST':
        entered_otp = request.form['otp']
        if int(entered_otp) == session.get('reset_otp'):
            # OTP is valid, proceed to reset password
            session.pop('reset_otp', None)  # Clear OTP from session
            return redirect(url_for('reset_password'))
        else:
            flash('Invalid OTP. Please try again.', 'danger')
            return redirect(url_for('verify_otp_for_reset'))
    
    return render_template("verify_otp_for_reset.html")

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        new_password = request.form['new_password']
        user_id = session.get('reset_user_id')
        
        if user_id:
            user = User.query.get(user_id)
            user.set_password(new_password)  # Assuming you have a method to hash the password
            db.session.commit()
            
            session.pop('reset_user_id', None)  # Clear user ID from session
            flash('Your password has been reset successfully! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('An error occurred. Please try again.', 'danger')
            return redirect(url_for('forgot_password'))
    
    return render_template("reset_password.html")


# -------------------------- HOME PAGE --------------------------

@app.route('/home', methods=['GET', 'POST'])
@login_required  # Protect this route with the login_required middleware
def home():
    if request.method == 'GET':
        return render_template("home.html")
    
# About Us page route
@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

# FAQ page route
@app.route('/faq')
def faq():
    return render_template('faq.html')

# Contact Us page route
@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')
    
# -------------------------- TEXT TO HANDWRITING PAGE --------------------------

@app.route('/text_to_handwriting', methods=['POST', 'GET'])
def text_to_handwriting_route():
    if request.method == 'GET':
        return render_template("generate_handwriting.html")
    else:
        text = request.form['text']
        background_type = request.form['background']
        font_name = request.form['font']

        # Set paths for font and output
        font_path = os.path.join('static', 'fonts', font_name)
        output_path = os.path.join('static', 'output', 'handwriting.png')

        # Call the function to create the handwriting image
        text_to_handwriting(text, font_path, output_path, background_type)

        # Return result to display
        return render_template('handwriting_result.html', image_path='output/handwriting.png')

def text_to_handwriting(text, font_path, output_path, background_type, image_size=(800, 400), font_size=30, text_color=(0, 0, 0)):
    if background_type == 'ruled':
        background = Image.open('static/ruled_page.jpg').resize(image_size)
    else:
        background = Image.open('static/unruled_page.jpg').resize(image_size)

    img = background.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    x, y = 20, 20
    lines = text.split('\n')
    for line in lines:
        words = line.split(' ')
        current_line = ''
        
        for word in words:
            width, _ = draw.textbbox((0, 0), current_line + word, font=font)[2:4]
            if width < (image_size[0] - 40):
                current_line += (word + ' ')
            else:
                draw.text((x, y), current_line, font=font, fill=text_color)
                y += font_size + 10
                current_line = word + ' '

        if current_line:
            draw.text((x, y), current_line, font=font, fill=text_color)
            y += font_size + 10

    img.save(output_path)


def process_image(filename):
    # Read the image with OpenCV
    img = cv2.imread(filename)

    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Apply thresholding for a black-and-white image
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Recognize text with Tesseract OCR
    result = pytesseract.image_to_string(img)

    return result

@app.route("/home/handwriting_to_text_route", methods=["GET", "POST"])
def handwriting_to_text_route():
    if request.method == "POST":
        # Handle file upload
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Save the uploaded file to a temporary directory
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_filename = temp_file.name
                file.save(temp_filename)
            
            # Process image to extract text
            extracted_text = process_image(temp_filename)
            os.remove(temp_filename)  # Remove the temp file after processing
            
            # Return the result page with the recognized text
            return render_template("htt_result.html", recognized_text=extracted_text)

    return render_template("handwriting_to_text.html")

# ----------------- PERSONALISED HANDWRITIG GENERATOR -----------------


# Helper Functions
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def extract_letter_variants(image_path):
    binary_image = preprocess_image(image_path)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    font_dict = defaultdict(list)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        letter_crop = binary_image[y:y+h, x:x+w]
        letter_img = Image.fromarray(cv2.bitwise_not(letter_crop))
        recognized_text = pytesseract.image_to_string(letter_img, config='--psm 10').strip()
        
        if re.fullmatch(r"[A-Za-z0-9]", recognized_text):
            font_dict[recognized_text].append(cv2.bitwise_not(letter_crop))
    
    return font_dict

def save_font_blob(font_dict, filename):
    with open(filename, "wb") as f:
        pickle.dump(font_dict, f)

def load_font_blob(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def generate_handwritten_image_from_text(text, font_dict, font_size=(32, 32), spacing=5, a4_width=2480, a4_height=3508, margin=(100, 100)):
    canvas_width, canvas_height = a4_width, a4_height
    top_margin, left_margin = margin
    right_margin = canvas_width - left_margin
    bottom_margin = canvas_height - top_margin
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    
    x_offset = left_margin
    y_offset = top_margin
    max_line_width = canvas_width - left_margin - font_size[0]

    words = text.split(' ')
    lines = []
    current_line = []

    for word in words:
        word_width = sum(font_size[0] + spacing for _ in word)
        if x_offset + word_width <= max_line_width:
            current_line.append(word)
            x_offset += word_width
        else:
            lines.append(current_line)
            current_line = [word]
            x_offset = sum(font_size[0] + spacing for _ in word)
    
    if current_line:
        lines.append(current_line)

    for line in lines:
        x_offset = left_margin
        for word in line:
            for char in word:
                if char in font_dict and font_dict[char]:
                    variant_img = random.choice(font_dict[char])
                    variant_img = Image.fromarray(variant_img).resize(font_size)
                    canvas.paste(variant_img, (x_offset, y_offset))
                    x_offset += font_size[0] + spacing
                else:
                    print("Character not found in font dictionary")
                    return None

            x_offset += spacing
        
        y_offset += font_size[1] + spacing
        if y_offset + font_size[1] > bottom_margin:
            print("Text exceeds bottom margin, stopping.")
            break

    return canvas

@app.route('/home/personalized_handwriting', methods = ["GET"])
def personalized_handwriting():
    font_files = [f[:-4] for f in os.listdir(app.config['FONT_FOLDER']) if f.endswith('.pkl')]
    return render_template('personalised_handwriting.html', fonts=font_files)

@app.route('/upload_font')
def upload_font():
    return render_template('upload_font.html')

@app.route('/upload', methods=['POST'])
def upload_handwriting():
    font_name = request.form['font_name']
    if 'handwriting_image' not in request.files or not font_name:
        return "No file or font name provided", 400
    file = request.files['handwriting_image']
    if file.filename == '':
        return "No selected file", 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        font_dict = extract_letter_variants(filepath)
        font_file = os.path.join(app.config['FONT_FOLDER'], f"{font_name}.pkl")
        save_font_blob(font_dict, font_file)
        return redirect(url_for('personalized_handwriting'))
    
@app.route('/generate', methods=['POST'])
def generate_handwriting():
    text = request.form['text']
    font_name = request.form['font_name']
    file_format = request.form['format']
    font_path = os.path.join(app.config['FONT_FOLDER'], f"{font_name}.pkl")
    if not os.path.exists(font_path):
        return "Font not found", 404

    font_dict = load_font_blob(font_path)
    handwritten_image = generate_handwritten_image_from_text(text, font_dict)
    if handwritten_image:
        img_io = io.BytesIO()
        if file_format == 'png':
            handwritten_image.save(img_io, 'PNG')
            mime_type = 'image/png'
            file_ext = 'png'
        else:
            img_io_pdf = io.BytesIO()
            pdf = pdf_canvas.Canvas(img_io_pdf)
            img_path = "temp_img.png"
            handwritten_image.save(img_path)
            pdf.drawImage(img_path, 0, 0)
            pdf.save()
            img_io_pdf.seek(0)
            os.remove(img_path)
            return send_file(img_io_pdf, mimetype='application/pdf', download_name="handwritten_text.pdf")

        img_io.seek(0)
        return send_file(img_io, mimetype=mime_type, download_name=f"handwritten_text.{file_ext}")

    return "Error generating image", 500

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FONT_FOLDER'], exist_ok=True)
    app.run(debug=True, port = 8023)
