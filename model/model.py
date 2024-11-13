from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import cv2
import pytesseract
from PIL import Image
import numpy as np
import pickle
import random
import os
import io
import re
from collections import defaultdict
from reportlab.pdfgen import canvas as pdf_canvas

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['FONT_FOLDER'] = './fonts'

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

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

# Routes
@app.route('/')
def index():
    font_files = [f[:-4] for f in os.listdir(app.config['FONT_FOLDER']) if f.endswith('.pkl')]
    return render_template('index.html', fonts=font_files)

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
        return redirect(url_for('index'))

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

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FONT_FOLDER'], exist_ok=True)
    app.run(debug=True, port = 8024)
