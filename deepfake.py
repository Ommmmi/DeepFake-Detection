import os
import shutil
from sklearn.model_selection import train_test_split

# Directories for extracted frames
extracted_frames_dir = 'C:\\Users\\ritul\\Downloads\\extracted_frames'  # Directory containing extracted frames
train_frames_dir = 'C:\\Users\\ritul\\Downloads\\train_frames'  # Directory to save training frames
test_frames_dir = 'C:\\Users\\ritul\\Downloads\\test_frames'  # Directory to save testing frames

# Function to split extracted frames
def split_frames_dataset(extracted_frames_dir, train_frames_dir, test_frames_dir, test_size=0.2):
    os.makedirs(train_frames_dir, exist_ok=True)
    os.makedirs(test_frames_dir, exist_ok=True)

    for label in ['real', 'fake']:
        label_dir = os.path.join(extracted_frames_dir, label)
        if not os.path.exists(label_dir):
            print(f"Directory for class '{label}' not found. Skipping.")
            continue

        # Get all subdirectories (one per video) in the label directory
        video_dirs = [os.path.join(label_dir, d) for d in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, d))]

        # Split directories into train and test
        train_dirs, test_dirs = train_test_split(video_dirs, test_size=test_size, random_state=42)

        # Copy train directories
        for video_dir in train_dirs:
            label_train_dir = os.path.join(train_frames_dir, label)
            os.makedirs(label_train_dir, exist_ok=True)
            shutil.copytree(video_dir, os.path.join(label_train_dir, os.path.basename(video_dir)))

        # Copy test directories
        for video_dir in test_dirs:
            label_test_dir = os.path.join(test_frames_dir, label)
            os.makedirs(label_test_dir, exist_ok=True)
            shutil.copytree(video_dir, os.path.join(label_test_dir, os.path.basename(video_dir)))

# Split the dataset
split_frames_dataset(extracted_frames_dir, train_frames_dir, test_frames_dir)
print("Frame dataset split complete.")
