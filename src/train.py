import os
import torch
from ultralytics import YOLO
import logging

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_CONFIG_PATH = 'data/data.yaml'

# params
TRAINING_PARAMS = {
    'model': 'yolov8s.pt',  
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'patience': 20,
    'project': 'runs/train',
    'name': 'helmet_detection_exp'
}

def check_gpu():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logging.info(f"Found {device_count} GPU(s). Using device {current_device}: {device_name}")
        return 'cuda'
    else:
        logging.warning("GPU not found. Training will run on CPU, which might be very slow.")
        return 'cpu'

def train_model():
    if not os.path.exists(DATA_CONFIG_PATH):
        logging.error(f"Data configuration file not found at '{DATA_CONFIG_PATH}'. Please ensure the file exists.")
        return
    device = check_gpu()

    try:
        logging.info(f"Loading pre-trained model: {TRAINING_PARAMS['model']}")
        model = YOLO(TRAINING_PARAMS['model'])
        model.to(device)

        logging.info("Starting model training with the following parameters:")
        for key, value in TRAINING_PARAMS.items():
            logging.info(f"  - {key}: {value}")
        
        results = model.train(
            data=DATA_CONFIG_PATH,
            epochs=TRAINING_PARAMS['epochs'],
            batch=TRAINING_PARAMS['batch'],
            imgsz=TRAINING_PARAMS['imgsz'],
            patience=TRAINING_PARAMS['patience'],
            project=TRAINING_PARAMS['project'],
            name=TRAINING_PARAMS['name'],
            device=0 if device == 'cuda' else 'cpu'
        )
        
        logging.info("✅ Training completed successfully!")
        best_model_path = os.path.join(results.save_dir, 'weights/best.pt')
        logging.info(f"Best model saved at: {best_model_path}")

    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)

if __name__ == '__main__':
    train_model()