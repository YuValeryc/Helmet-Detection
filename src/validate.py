import os
import glob
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_CONFIG_PATH = 'data/data.yaml'
TRAIN_RESULTS_DIR = 'runs/train'

def find_latest_run(base_dir):
    if not os.path.isdir(base_dir):
        return None
    list_of_dirs = glob.glob(os.path.join(base_dir, '*'))
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    return latest_dir

def validate_model():
    latest_run_dir = find_latest_run(TRAIN_RESULTS_DIR)
    
    if not latest_run_dir:
        logging.error(f"No training runs found in '{TRAIN_RESULTS_DIR}'.")
        return

    model_path = os.path.join(latest_run_dir, 'weights', 'best.pt')

    if not os.path.exists(model_path):
        logging.error(f"Model weights 'best.pt' not found in: {latest_run_dir}")
        return

    if not os.path.exists(DATA_CONFIG_PATH):
        logging.error(f"Data configuration file not found at '{DATA_CONFIG_PATH}'.")
        return

    try:
        logging.info(f"Loading model from: {model_path}")
        model = YOLO(model_path)

        logging.info(f"Validating on test split using '{DATA_CONFIG_PATH}'...")
        
        metrics = model.val(
            data=DATA_CONFIG_PATH,
            split='test',
            imgsz=640
        )
        
        logging.info("Validation completed.")
        logging.info("\n--- Overall Validation Metrics ---")
        logging.info(f"  - mAP50-95: {metrics.box.map:.4f}")
        logging.info(f"  - mAP50:    {metrics.box.map50:.4f}")
        logging.info(f"  - mAP75:    {metrics.box.map75:.4f}")
        
        logging.info("\n--- Per-Class Metrics ---")
        all_class_names = model.names
        
        if not metrics.ap_class_index:
            logging.warning("No classes detected in the test set.")
        else:
            for i, class_index in enumerate(metrics.ap_class_index):
                class_name = all_class_names[class_index]
                precision = metrics.box.p[i]
                recall = metrics.box.r[i]
                map50 = metrics.box.maps[i]
                
                logging.info(f"  - Class '{class_name}' (index: {class_index}):")
                logging.info(f"    - Precision: {precision:.4f}")
                logging.info(f"    - Recall:    {recall:.4f}")
                logging.info(f"    - mAP@50:    {map50:.4f}")
        
        logging.info("--------------------------------------")
        
    except Exception as e:
        logging.error(f"Validation error: {e}", exc_info=True)

if __name__ == '__main__':
    validate_model()
