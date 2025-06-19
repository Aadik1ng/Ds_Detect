import os
import glob
import cv2
import numpy as np
import yaml
from sklearn.model_selection import KFold

def check_data(config_path):
    images, labels = load_data(config_path)
    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    
    # Optionally, print the first few image and label paths
    if images:
        print("Sample images:", images[:5])
    if labels:
        print("Sample labels:", labels[:5])

def load_data(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Use absolute paths
    image_dir = os.path.abspath(os.path.join(config['dataset']['path'] ,config['dataset']['images']))
    label_dir = os.path.abspath(os.path.join(config['dataset']['path'] ,config['dataset']['labels']))
    
    print(f"Image directory: {image_dir}")
    print(f"Label directory: {label_dir}")
    
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))
            if os.path.exists(label_path):
                images.append(img_path)
                labels.append(label_path)
    
    print(f"Found {len(images)} images: {images}")
    print(f"Expected labels: {labels}")
    
    return images, labels
    

def k_fold_split(data, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    return list(kf.split(data))

def preprocess_image(image_path, target_size=(640, 640)):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    target_h, target_w = target_size

    # Calculate the ratio to maintain aspect ratio
    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h))

    # Create a new image with padding
    padded_image = np.full((target_h, target_w, 3), 128)  # Fill with a gray color
    padded_image[(target_h - new_h) // 2:(target_h - new_h) // 2 + new_h,
                 (target_w - new_w) // 2:(target_w - new_w) // 2 + new_w] = resized_image

    return padded_image

if __name__ == "__main__":
    config_path = r'config\config.yaml' 
    check_data(config_path)
