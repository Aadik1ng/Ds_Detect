import cv2
import os
import yaml
from ultralytics import YOLO
import json

def predict_drones(video_path, config_path, output_json_path="drone_detections.json"):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load the trained model
    model_path = os.path.join(config['training']['save_dir'], 'best.pt')  # Adjust as needed
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    drone_count = 0

    detection_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run inference
        results = model(frame, conf=config['inference']['conf'], iou=config['inference']['iou'])

        # Process results and collect data if drones are detected
        frame_detections = []
        total_drones_in_frame = 0

        # Assuming drone class is 0 based on config.yaml and app.py
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            for box in results[0].boxes:
                # Check if the detected class is drone (class ID 0)
                if int(box.cls[0].item()) == 0:
                    total_drones_in_frame += 1
                    # Extract bounding box (xyxy format), confidence, and class ID
                    x1, y1, x2, y2 = [round(coord.item(), 2) for coord in box.xyxy[0]]
                    confidence = round(box.conf[0].item(), 4)
                    class_id = int(box.cls[0].item())

                    frame_detections.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id # Should be 0 for drone
                    })

        # If any drones were detected in this frame, record the data
        if total_drones_in_frame > 0:
            detection_data.append({
                "frame_number": frame_count,
                "drone_count": total_drones_in_frame,
                "detections": frame_detections
            })

        # Optionally, draw bounding boxes on the frame
        for *box, conf, cls in results.xyxy[0]:
            label = f'Drone {conf:.2f}'
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame (optional)
        cv2.imshow('Drone Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save detection data to JSON file
    try:
        with open(output_json_path, 'w') as f:
            json.dump(detection_data, f, indent=4)
        print(f'Detection data saved to {output_json_path}')
    except Exception as e:
        print(f'Error saving JSON data: {e}')

    print(f'Total number of drones detected in frames with detections: {sum(d["drone_count"] for d in detection_data)}') # Updated total count

if __name__ == "__main__":
    video_path = 'videoplayback.mp4'  # Replace with your video path or use the sample video
    config_path = r'config\config.yaml'
    output_json_path = r'output\drone_detections.json' # Define output path
    # Check if the sample video exists before running
    if os.path.exists(video_path):
        predict_drones(video_path, config_path, output_json_path)
    else:
        print(f"Error: Video file not found at {video_path}")
        print("Please update the 'video_path' variable in predict.py to point to your video file.")