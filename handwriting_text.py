import cv2
import numpy as np
from PIL import Image
import pytesseract
import fpdf
import os
from difflib import SequenceMatcher

# Configure path to Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_image(filename):
    # Read the image with OpenCV
    img = cv2.imread(filename)

    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Apply thresholding to get a black-and-white image
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Write the processed image (optional step, can be skipped)
    cv2.imwrite("processed_image.png", img)

    # Recognize text with Tesseract OCR
    result = pytesseract.image_to_string(Image.open("processed_image.png"))

    return result

def save_to_pdf(text, output_filename="converted.pdf"):
    pdf = fpdf.FPDF(format='letter')
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.write(5, text)
    pdf.ln()
    pdf.output(output_filename)
    print(f"Text saved to {output_filename}")
    
    # Automatically open the PDF if on Windows
    if os.name == 'nt':  # Windows
        os.startfile(output_filename)
    elif os.name == 'posix':  # macOS or Linux
        os.system(f'open "{output_filename}"')  # macOS
        # or use 'xdg-open' for Linux
        # os.system(f'xdg-open "{output_filename}"')  # Uncomment for Linux

def calculate_accuracy(reference_text, extracted_text):
    similarity = SequenceMatcher(None, reference_text, extracted_text).ratio()
    return f"The accuracy of the model is {similarity * 100:.2f}%\n"

if __name__ == "__main__":
    # Replace with the path to your image file
    image_path = r"C:\Users\sneha\OneDrive\Desktop\sneha\Handwritten_Text_Recognition\SAMPLE1.jpg"
    extracted_text = process_image(image_path)

    # Output the extracted text
    print("Extracted Text:")
    print(extracted_text)

    # Save the extracted text to a PDF
    save_to_pdf(extracted_text)

    # Define a reference text to compare accuracy
    reference_text = "We start With good\n\nBecause all businesses should\n\nbe doing something good"
    print(calculate_accuracy(reference_text, extracted_text))
