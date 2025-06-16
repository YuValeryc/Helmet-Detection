import os
import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, Response
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import logging

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS_IMG = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_VID = {'mp4', 'avi', 'mov', 'mkv'}
MODEL_PATH = 'weights/best.pt'

# --- APP & SOCKETIO INITIALIZATION ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'a_very_secret_key'
socketio = SocketIO(app, async_mode='threading')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- MODEL LOADING ---
try:
    model = YOLO(MODEL_PATH)
    logging.info("YOLOv8 model loaded successfully!")
except Exception as e:
    model = None
    logging.error(f"Error loading YOLOv8 model: {e}")

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +                      IMAGE & VIDEO ROUTES                    +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def process_video_frames(video_path):
    """Generator function to process video frames and yield them."""
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Perform inference
        results = model(frame)
        annotated_frame = results[0].plot()

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the format required for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Check file type and process accordingly
            if allowed_file(filename, ALLOWED_EXTENSIONS_IMG):
                results = model(filepath)
                result_filename = f"result_{filename}"
                result_path = os.path.join('static', result_filename) # Save in static root for easy access
                results[0].save(filename=result_path)
                return render_template('index.html', result_file=result_filename, file_type='image')

            elif allowed_file(filename, ALLOWED_EXTENSIONS_VID):
                # For video, we will stream the result
                return render_template('index.html', result_file=filename, file_type='video')
            else:
                flash('Invalid file type. Please upload an image or video.', 'error')
                return redirect(request.url)

    return render_template('index.html')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    """Route to stream video frames."""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(process_video_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +                      WEBCAM STREAMING ROUTES                 +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@app.route('/webcam')
def webcam():
    """Render the webcam streaming page."""
    return render_template('webcam.html')

@socketio.on('image')
def handle_image(data_image):
    """Handle incoming webcam frames from the client."""
    if model is None:
        return

    # Decode base64 image
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(data_image.split(',')[1]))
    pil_image = Image.open(sbuf)
    
    # Convert to OpenCV format (NumPy array)
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Perform inference
    results = model(frame)
    annotated_frame = results[0].plot()

    # Encode processed frame back to base64
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    b64_string = base64.b64encode(buffer).decode('utf-8')
    
    # Send processed frame back to the client
    emit('response', {'image': b64_string})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)