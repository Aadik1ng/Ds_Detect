import streamlit as st
import os
import re
import tempfile
from ultralytics import YOLO

# Configure page
st.set_page_config(page_title="Drone Detection App", layout="wide", page_icon="🚁")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

def get_fold_dirs(base_dir="runs/detect"):
    """Finds directories corresponding to k-fold training runs."""
    fold_dirs = []
    if os.path.exists(base_dir):
        fold_pattern = re.compile(r'train_fold_\d+')
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and fold_pattern.match(item):
                fold_dirs.append(item_path)
    return sorted(fold_dirs)

# Main header
st.markdown('<div class="main-header"><h1>🚁 Drone Detection System</h1><p>K-Fold Model Inference Interface</p></div>', unsafe_allow_html=True)

# Sidebar for model selection
with st.sidebar:
    st.header("🎯 Model Configuration")
    
    available_folds = get_fold_dirs()
    if not available_folds:
        st.error("❌ No k-fold training directories found")
        st.stop()
    
    selected_fold_dir = st.selectbox('📁 Select Training Fold:', available_folds, format_func=lambda x: f"Fold {x.split('_')[-1]}")
    selected_model_type = st.radio('⚖️ Model Weights:', ('best.pt', 'last.pt'), help="best.pt: Best validation performance, last.pt: Latest checkpoint")
    
    weights_dir = os.path.join(selected_fold_dir, 'weights')
    selected_model_path = os.path.join(weights_dir, selected_model_type)
    
    if os.path.exists(selected_model_path):
        st.success(f"✅ Model loaded: {os.path.basename(selected_model_path)}")
        st.info(f"📂 Path: {selected_model_path}")
    else:
        st.error("❌ Model not found")
        st.stop()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Input Data")
    
    # File upload option
    uploaded_file = st.file_uploader("Choose an image or video file", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'mkv'])
    
    # Or path input
    input_data_path = st.text_input("Or enter file path:", placeholder="/path/to/your/file.jpg")
    
    # Use uploaded file or path
    final_input_path = None
    if uploaded_file is not None:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            final_input_path = tmp_file.name
        st.success(f"📁 File uploaded: {uploaded_file.name}")
    elif input_data_path and os.path.exists(input_data_path):
        final_input_path = input_data_path
        st.success(f"📁 File found: {os.path.basename(input_data_path)}")
    
    # Inference settings
    st.subheader("⚙️ Inference Settings")
    col_conf, col_iou = st.columns(2)
    with col_conf:
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    with col_iou:
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)

with col2:
    st.header("📊 Model Info")
    st.markdown(f"""
    <div class="metric-card">
        <h4>Selected Model</h4>
        <p><strong>Fold:</strong> {selected_fold_dir.split('_')[-1]}</p>
        <p><strong>Weights:</strong> {selected_model_type}</p>
        <p><strong>Status:</strong> ✅ Ready</p>
    </div>
    """, unsafe_allow_html=True)

# Run inference
if st.button("🚀 Run Detection", type="primary", use_container_width=True):
    if final_input_path:
        try:
            with st.spinner("🔄 Loading model..."):
                model = YOLO(selected_model_path)
            
            st.success("✅ Model loaded successfully!")
            
            # Determine if video or image
            is_video = any(final_input_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv'])
            
            if is_video:
                st.header("🎬 Video Processing")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process video with progress tracking
                results = model(final_input_path, conf=confidence, iou=iou_threshold, save=True)
                
                # Display results
                st.subheader("📹 Detection Results")
                total_drones = 0
                frames_with_detections = 0
                
                # Create columns for metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                for i, r in enumerate(results):
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        frame_drones = sum(1 for box in r.boxes if int(box.cls[0].item()) == 0)
                        if frame_drones > 0:
                            frames_with_detections += 1
                            total_drones += frame_drones
                    
                    # Update progress
                    progress = (i + 1) / len(results)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {i+1}/{len(results)}")
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                # Display metrics
                with metric_col1:
                    st.metric("Total Frames", len(results))
                with metric_col2:
                    st.metric("Frames with Detections", frames_with_detections)
                with metric_col3:
                    st.metric("Total Drones Detected", total_drones)
                
                # Show sample frames with detections
                st.subheader("🎯 Sample Detections")
                sample_frames = [r for r in results if hasattr(r, 'boxes') and r.boxes is not None and any(int(box.cls[0].item()) == 0 for box in r.boxes)][:5]
                
                if sample_frames:
                    cols = st.columns(min(len(sample_frames), 3))
                    for i, r in enumerate(sample_frames):
                        col_idx = i % 3
                        with cols[col_idx]:
                            img_annotated = r.plot()
                            drone_count = sum(1 for box in r.boxes if int(box.cls[0].item()) == 0)
                            st.image(img_annotated, caption=f"Drones: {drone_count}", use_column_width=True)
                else:
                    st.info("No drones detected in the video")
                    
            else:
                st.header("🖼️ Image Processing")
                with st.spinner("🔄 Running detection..."):
                    results = model(final_input_path, conf=confidence, iou=iou_threshold, save=True)
                
                st.success("✅ Detection completed!")
                
                # Display results
                for r in results:
                    img_annotated = r.plot()
                    drone_count = 0
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        drone_count = sum(1 for box in r.boxes if int(box.cls[0].item()) == 0)
                    
                    st.image(img_annotated, caption=f"Detections: {drone_count} drones", use_column_width=True)
                    
                    # Display detection details
                    if drone_count > 0:
                        st.subheader("📋 Detection Details")
                        for i, box in enumerate(r.boxes):
                            if int(box.cls[0].item()) == 0:  # Drone class
                                conf = box.conf[0].item()
                                st.write(f"Drone {i+1}: Confidence {conf:.3f}")
                
        except Exception as e:
            st.error(f"❌ Error during inference: {str(e)}")
    else:
        st.warning("⚠️ Please upload a file or provide a valid file path")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>🚁 Drone Detection System | Powered by YOLOv8</div>", unsafe_allow_html=True)