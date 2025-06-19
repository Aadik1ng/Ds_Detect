import os
import yaml
from ultralytics import YOLO
from utils.data_utils import load_data

def evaluate_model(config_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load data
    images, labels = load_data(config_path)

    # Load the trained model
    model_path = os.path.join(config['training']['save_dir'], 'best.pt')  # Adjust as needed
    model = YOLO(model_path)

    # Evaluate the model
    results = model.val(data=config['dataset']['path'], imgsz=config['training']['imgsz'])

    # Print evaluation metrics
    print("Evaluation Results:")
    print(f"Precision: {results.p:.4f}")
    print(f"Recall: {results.r:.4f}")
    print(f"mAP: {results.map:.4f}")

if __name__ == "__main__":
    evaluate_model('../config/config.yaml')