import streamlit as st
import os
import re
from ultralytics import YOLO # Import YOLO

def get_fold_dirs(base_dir="runs/detect"):
    """Finds directories corresponding to k-fold training runs."""
    fold_dirs = []
    if os.path.exists(base_dir):
        # Regex to match directories like train_fold_0, train_fold_1, etc.
        fold_pattern = re.compile(r'train_fold_\d+')
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and fold_pattern.match(item):
                fold_dirs.append(item_path)
    return sorted(fold_dirs)

st.title('K-Fold Model Selector')

# Get available fold directories
available_folds = get_fold_dirs()

if not available_folds:
    st.warning("No k-fold training directories found in runs/detect/. Please run the training script first.")
elif available_folds:
    # Create a dropdown to select a fold
    selected_fold_dir = st.selectbox(
        'Select a training fold:',
        available_folds
    )

    st.write(f"Selected Fold: {selected_fold_dir}")

    # Add radio buttons to select model type
    selected_model_type = st.radio(
        'Select model weights:',
        ('best.pt', 'last.pt')
    )

    # Construct paths to best.pt and last.pt
    weights_dir = os.path.join(selected_fold_dir, 'weights')
    selected_model_path = os.path.join(weights_dir, selected_model_type)

    st.subheader('Selected Model Path')
    if os.path.exists(selected_model_path):
        st.info(f"Using model: `{selected_model_path}`")

        st.subheader('Run Inference')
        input_data_path = st.text_input('Enter path to image or video file:')

        if st.button('Run Inference'):
            if input_data_path and os.path.exists(input_data_path):
                try:
                    # Load the model
                    model = YOLO(selected_model_path)

                    # Run inference
                    # Setting save=True tells ultralytics to save the results
                    results = model(input_data_path, save=True)

                    st.success("Inference completed!")

                    # Determine if the input was a video or image
                    is_video = any(input_data_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv'])

                    st.subheader("Inference Results")

                    if is_video:
                        st.write("Processing video frame by frame:")
                        total_drones_detected = 0
                        frame_count = 0

                        for r in results:
                            frame_count += 1
                            # r.plot() returns an annotated image (frame) as a numpy array
                            img_annotated = r.plot()

                            # Count drones in the current frame (assuming drone class is 0)
                            current_frame_drones = 0
                            if hasattr(r, 'boxes') and r.boxes is not None:
                                for box in r.boxes:
                                    # Check if the detected class is drone (class ID 0)
                                    if int(box.cls[0].item()) == 0:
                                        current_frame_drones += 1
                                        total_drones_detected += 1

                            st.image(img_annotated, caption=f"Frame {frame_count} - Drones detected: {current_frame_drones}", use_column_width=True)

                        st.subheader("Summary")
                        st.write(f"Total number of drones detected across all frames: {total_drones_detected}")

                    else:
                        # Input was likely an image, display annotated image(s)
                        for r in results:
                            # r.plot() returns an annotated image as a numpy array
                            img_annotated = r.plot()
                            st.image(img_annotated, caption=f"Detections for {os.path.basename(input_data_path)}", use_column_width=True)

                except Exception as e:
                    st.error(f"An error occurred during inference: {e}")
            elif not input_data_path:
                st.warning("Please enter a path to an image or video file.")
            else:
                st.warning(f"Input file not found at: {input_data_path}")

    else:
        st.warning(f"Selected model weights not found at `{selected_model_path}`. Please ensure training for this fold is complete.") 