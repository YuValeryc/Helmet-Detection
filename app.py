import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import logging

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads/'
RESULTS_FOLDER = 'static/results/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'weights/best.pt'

# --- APP INITIALIZATION ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.secret_key = 'a_super_secret_key_for_flash_messages' 

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --- MODEL LOADING ---
# Load the model once when the application starts for efficiency.
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        logging.info("YOLOv8 model loaded successfully!")
    else:
        model = None
        logging.error(f"Model file not found at {MODEL_PATH}. Please download it and place it in the 'weights' directory.")
except Exception as e:
    model = None
    logging.error(f"Error loading YOLOv8 model: {e}")

def is_allowed_file(filename):
    """Checks if the file's extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if model is None:
        flash("Model is not available. Please check server logs for details.", "error")
        return render_template('index.html', error="Model could not be loaded.")

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        if file and is_allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)
                logging.info(f"File saved to {upload_path}")

                # --- Perform Detection ---
                results = model(upload_path)

                # Save the result image
                result_filename = f"result_{filename}"
                result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                results[0].save(filename=result_path)
                logging.info(f"Result image saved to {result_path}")

                # Get the URL for the result image to display in the template
                result_image_url = url_for('static', filename=f'results/{result_filename}')
                
                return render_template('index.html', result_image=result_image_url)

            except Exception as e:
                logging.error(f"An error occurred during processing: {e}")
                flash(f"An error occurred during processing: {e}", "error")
                return redirect(request.url)

        else:
            flash('File type not allowed. Please upload a PNG, JPG, or JPEG file.', 'error')
            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)