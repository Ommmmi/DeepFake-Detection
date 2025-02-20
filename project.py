import cv2
import os
from mtcnn import MTCNN
from tqdm import tqdm

# Define directories
input_dir = 'C:\\Users\\ritul\\Downloads\\archive (10)'  # Directory containing input videos
output_dir = 'C:\\Users\\ritul\\Downloads\\processed_video'  # Directory to save processed videos
frames_output_dir = 'C:\\Users\\ritul\\Downloads\\extracted_frames'  # Directory to save extracted frames

# Initialize MTCNN for face detection
detector = MTCNN()

# Function to detect and crop face
def detect_and_crop_face(frame):
    # Detect faces
    detections = detector.detect_faces(frame)
    if len(detections) > 0:
        # Assume the first detected face is the target
        x, y, width, height = detections[0]['box']
        x, y = max(0, x), max(0, y)  # Ensure coordinates are within bounds
        cropped_face = frame[y:y+height, x:x+width]
        return cropped_face
    return None  # No face detected

# Function to process a single video
def process_video(video_path, output_video_path, output_frames_dir):
    # Read the video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = fps * 3  # Capture every 3 seconds

    # Define the output video writer
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4 codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (224, 224))  # Resize cropped faces to 224x224
    
    # Create directory for frames
    os.makedirs(output_frames_dir, exist_ok=True)

    for frame_idx in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        # Process frames only at specified intervals
        if frame_idx % frame_interval == 0:
            # Detect and crop face
            face = detect_and_crop_face(frame)
            if face is not None:
                # Resize to 224x224 for consistency
                face = cv2.resize(face, (224, 224))
                out.write(face)

                # Save each frame as an image
                frame_filename = os.path.join(output_frames_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(frame_filename, face)

    cap.release()
    out.release()

# Main preprocessing loop
def preprocess_dataset(input_dir, output_dir, frames_output_dir):
    # Iterate over all video files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(video_path, input_dir)
                
                # Paths for processed video and frames
                output_video_path = os.path.join(output_dir, relative_path)
                output_frames_dir = os.path.join(frames_output_dir, os.path.splitext(relative_path)[0])
                
                process_video(video_path, output_video_path, output_frames_dir)

# Run preprocessing
if __name__ == "__main__":
    preprocess_dataset(input_dir, output_dir, frames_output_dir)
    print("Preprocessing complete. Processed videos and frames saved to:", output_dir, "and", frames_output_dir)


