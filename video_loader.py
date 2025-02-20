import torch
import os
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset, DataLoader

# Custom Dataset Class for Video Loading
class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.video_paths = []
        self.labels = []

        for label, class_name in enumerate(['real', 'fake']):
            class_dir = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.mp4'):
                    self.video_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load video
        video, _, _ = read_video(video_path, pts_unit='sec')

        # Apply transformations
        if self.transform:
            video = torch.stack([self.transform(frame) for frame in video])

        return video, label

# Define transformations
transform = Compose([
    Resize((224, 224)),  # Resize frames to 224x224
    ToTensor()           # Convert frames to tensors
])

# Create datasets
train_dataset = VideoDataset("C:\\Users\\ritul\\Downloads\\train_frames", transform=transform)
test_dataset = VideoDataset("C:\\Users\\ritul\\Downloads\\test_frames", transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

print("Data loading complete.")

