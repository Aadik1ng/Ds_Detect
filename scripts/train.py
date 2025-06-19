import os
import yaml
from ultralytics import YOLO
from utils.data_utils import load_data, k_fold_split,check_data

def train_model(config_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load data
    images, labels = load_data(config_path)
    
    # K-Fold Cross Validation
    folds = k_fold_split(images, k=config['training']['k_folds'])
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"Training on fold {fold + 1}/{config['training']['k_folds']}")

        # Create temporary directory for the current fold
        fold_dir = f'runs/fold_{fold}'
        os.makedirs(os.path.join(fold_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'labels', 'val'), exist_ok=True)

        # Prepare training and validation data paths for the current fold
        train_images = [images[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_images = [images[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # Copy/Symlink files to temporary fold directory (copy for simplicity, symlink for efficiency on non-Windows)
        for img_path, label_path in zip(train_images, train_labels):
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)
            dest_img_path = os.path.join(fold_dir, 'images', 'train', img_name)
            dest_label_path = os.path.join(fold_dir, 'labels', 'train', label_name)
            try:
                os.link(img_path, dest_img_path) # Try symlink first
            except OSError:
                import shutil # Import shutil only if needed
                shutil.copyfile(img_path, dest_img_path)
            try:
                os.link(label_path, dest_label_path) # Try symlink first
            except OSError:
                 import shutil # Import shutil only if needed
                 shutil.copyfile(label_path, dest_label_path)

        for img_path, label_path in zip(val_images, val_labels):
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)
            dest_img_path = os.path.join(fold_dir, 'images', 'val', img_name)
            dest_label_path = os.path.join(fold_dir, 'labels', 'val', label_name)
            try:
                os.link(img_path, dest_img_path) # Try symlink first
            except OSError:
                import shutil # Import shutil only if needed
                shutil.copyfile(img_path, dest_img_path)
            try:
                os.link(label_path, dest_label_path) # Try symlink first
            except OSError:
                 import shutil # Import shutil only if needed
                 shutil.copyfile(label_path, dest_label_path)

        # Create temporary dataset.yaml for the current fold
        fold_dataset_config = {
            'path': fold_dir, # Path to the temporary fold directory
            'train': 'images/train',
            'val': 'images/val',
            'nc': config['dataset']['nc'] if 'nc' in config['dataset'] else 1, # Assuming nc is in main config or default 1
            'names': config['dataset']['names']
        }
        fold_dataset_yaml_path = os.path.join(fold_dir, 'dataset.yaml')
        with open(fold_dataset_yaml_path, 'w') as f:
            yaml.dump(fold_dataset_config, f)

        # Load YOLO model (re-initialize for each fold to ensure clean state)
        model = YOLO(config['model']['name'])

        # Train the model for the current fold
        model.train(data=fold_dataset_yaml_path,
                    imgsz=config['training']['imgsz'],
                    epochs=config['training']['epochs'],
                    batch=config['training']['batch'],
                    # save_dir will be handled by ultralytics within the fold_dir
                    # project and name can be set to distinguish folds
                    project='runs/detect',
                    name=f'train_fold_{fold}')

        # Save logs and metrics for the fold (Ultralytics saves within the run directory)

        # Clean up temporary fold directory (optional, can keep for inspection)
        # import shutil
        # shutil.rmtree(fold_dir)

if __name__ == "__main__":
    config_path = r'config\config.yaml'
    check_data(config_path)
    train_model(config_path)