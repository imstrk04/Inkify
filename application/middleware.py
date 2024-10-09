from flask import Flask
from flask import request, g
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

def log_request():
    """Middleware to log requests."""
    @app.before_request
    def before_request():
        # Log request details
        logging.info(f"Request: {request.method} {request.url}")
        # You can also log other details, such as the user's IP address, etc.
