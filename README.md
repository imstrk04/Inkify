

# Inkify ✒️

Inkify is a web application that provides multiple handwriting-related features, such as **Text-to-Handwriting**, **Handwriting-to-Text**, and **Personalized Handwriting**. Users can create accounts to personalize and save their work. This project is built using Flask for the backend, with SQLite as the database, and HTML/CSS/JavaScript/Bootstrap for the frontend.


## Table of Contents
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Features
- **Text-to-Handwriting**: Converts digital text into a simulated handwritten style and saves it as an image.
- **Handwriting-to-Text**: Recognizes handwritten input from an image and converts it into digital text.
- **Personalized Handwriting**: Allows users to upload their own handwriting style to customize the generated output.
- **User Authentication**: Supports signup and login for a personalized experience.

## Technologies
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Backend**: Flask, Python
- **Database**: SQLite
- **Libraries**:
  - Pillow (for image processing)
  - OpenCV (for handwriting detection and analysis)
  - Tesseract (OCR for handwriting-to-text conversion)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/inkify.git
   cd inkify
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Rename `.env.example` to `.env` and add any required configuration variables.

5. **Initialize the Database**:
   ```bash
   flask shell
   from application import db
   db.create_all()
   ```

6. **Run the Application**:
   ```bash
   flask run
   ```

7. **Access the Application**:
   Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) to view Inkify in your browser.

## Usage
1. **Sign Up**: Create an account to access personalized handwriting settings.
2. **Text-to-Handwriting**: Enter text and generate a handwritten image output.
3. **Handwriting-to-Text**: Upload an image of handwritten text, and Inkify will convert it to digital text.
4. **Personalization**: Upload a sample of your handwriting, and Inkify will use it to style future outputs.

## Project Structure
```plaintext
Inkify/
├── __pycache__              # Cache files
├── application/             # Main application folder
│   ├── __init__.py          # Initializes the Flask app
│   ├── models.py            # Database models
│   ├── routes.py            # App routes and views
├── instance/                # Instance folder for configurations
├── static/                  # Static files (CSS, JS, images)
├── templates/               # HTML templates
├── .env                     # Environment variables
├── app.py                   # Main app entry point
├── handwriting_text.py      # Script for handwriting-to-text conversion
├── text_to_handwriting.py   # Script for text-to-handwriting conversion
├── demo.png                 # Demo image for the README
├── handwritten.png          # Example handwritten output
└── requirements.txt         # Required packages
```

## Future Enhancements
- **Enhanced Personalization**: Allow more customization of handwriting styles.
- **Export Options**: Enable exporting results as various image formats.
- **Advanced Handwriting Personalization**: Improve recognition and adaptability for custom handwriting styles.
- **Handwriting Fonts**: Add more handwritten font styles to choose from.
- **Mobile Optimization**: Enhance user interface for better performance on mobile devices.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
