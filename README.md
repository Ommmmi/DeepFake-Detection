# DeepFake-Detection
This project focuses on detecting deepfake videos by analyzing extracted frames using Convolutional Neural Networks (CNNs). The model classifies video frames as either real or fake and can process videos to provide an overall prediction.
Project Structure
├── extracted_frames/       # Directory containing extracted frames
│   ├── real/               # Real video frames
│   ├── fake/               # Fake video frames
├── train_frames/           # Training dataset
├── test_frames/            # Testing dataset
├── model_training.py       # Script for training CNN model
├── dataset_split.py        # Script for splitting dataset
├── predict_video.py        # Script for testing on new videos
├── requirements.txt        # Required dependencies
Features

Frame Extraction & Preprocessing: Automatically organizes video frames into real and fake categories.

Deep Learning Model: Uses ResNet18/VGG19 for classification.

Data Augmentation: Includes horizontal flipping, color jitter, and rotation to improve model generalization.

Training & Evaluation: Tracks accuracy and loss, includes confusion matrix visualization.

Real-time Video Prediction: Processes videos by analyzing frames and aggregating predictions.
Installation

Clone the repository:

git clone https://github.com/Ommmmi/deepfake-detection.git
cd deepfake-detection

Install dependencies:

pip install -r requirements.txt

Ensure you have the necessary dataset with frames stored in extracted_frames/.

Usage

Split Dataset

python dataset_split.py

Train the Model

python model_training.py

Test on a Video

python predict_video.py --video_path path/to/video.mp4

Model Performance

Training and validation metrics are logged.

Best model is saved automatically.

Confusion matrix visualization for error analysis.

Future Work

Implement attention-based CNNs for improved accuracy.

Use optical flow analysis to detect temporal inconsistencies in deepfakes.

Develop a web-based interface for real-time deepfake detection.



