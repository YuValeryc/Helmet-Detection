# Helmet Detection Web Application

A Flask-based web application that uses YOLOv8 for detecting helmets in images. This application allows users to upload images and get real-time helmet detection results.

## Features

- Image upload functionality
- Real-time helmet detection using YOLOv8
- Support for PNG, JPG, and JPEG image formats
- Automatic result display
- User-friendly web interface

## Prerequisites

- Python 3.8 or higher
- Flask
- Ultralytics YOLOv8
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YuValeryc/Helmet-Detection
cd Helmet-Detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Download the YOLOv8 model weights:
   - Place the `best.pt` model file in the `weights` directory
   - The model file should be named `best.pt`

## Project Structure

```
Helmet-Detection/
├── app.py              # Main Flask application
├── static/
│   ├── uploads/       # Directory for uploaded images
│   └── results/       # Directory for detection results
├── templates/
│   └── index.html     # Main template file
├── weights/
│   └── best.pt        # YOLOv8 model weights
└── requirements.txt    # Project dependencies
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

3. Upload an image using the web interface
4. View the detection results displayed on the page

## Configuration

The application uses the following default configurations:
- Upload folder: `static/uploads/`
- Results folder: `static/results/`
- Allowed file extensions: PNG, JPG, JPEG
- Model path: `weights/best.pt`

## Error Handling

The application includes comprehensive error handling for:
- Missing model file
- Invalid file types
- Upload errors
- Processing errors

## Logging

The application logs important events and errors to help with debugging and monitoring.

