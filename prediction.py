import torch
from torchvision import models
import cv2
import numpy as np
from PIL import Image
from ss import get_model  # Assuming get_model is implemented elsewhere

from torchvision import transforms

# Define the transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to preprocess a frame
def preprocess_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert frame to PIL Image
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Load the trained model
def load_model(model_path, model_name='resnet'):
    model = get_model(model_name)  # Choose the same model architecture as used during training
    # Load the state dict with strict=False to ignore missing/unexpected keys
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)  # Use strict=False to avoid errors for missing/extra keys
    model.eval()  # Set the model to evaluation mode
    return model

# Function to process and predict on a video
def predict_video(video_path, model, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every `frame_skip` frames
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_pos % frame_skip == 0:
            preprocessed_frame = preprocess_frame(frame)
            with torch.no_grad():
                output = model(preprocessed_frame)  # Perform forward pass
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities).item()
                predictions.append(predicted_class)

    cap.release()
    # Aggregate predictions (e.g., majority vote)
    final_prediction = max(set(predictions), key=predictions.count)
    return final_prediction, predictions

# Path to the best model
best_model_path = 'best_model.pth'  # Ensure this is the correct path to your saved model

# Load the model
model = load_model(best_model_path, model_name='resnet')  # Change 'resnet' to the architecture used in training

# Path to the video file for prediction
video_path = "C:\\Users\\ritul\\Downloads\\00288.mp4"  # Replace with a valid video path

# Predict the class of the video
predicted_class, frame_predictions = predict_video(video_path, model, frame_skip=30)

# Map the class index to the label
class_labels = {0: 'real', 1: 'fake'}
predicted_label = class_labels[predicted_class]

print(f"Final Predicted Class for Video: {predicted_label}")
print(f"Frame-level Predictions: {frame_predictions}")

