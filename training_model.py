import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Dataset class for frames
class FrameDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        print(f"Scanning directory: {data_dir}")

        # Ensure directory exists
        if not os.path.exists(data_dir):
            print(f"Directory does not exist: {data_dir}")
            return

        for label, class_name in enumerate(['real', 'fake']):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Folder does not exist: {class_dir}")
                continue

            for video_folder in os.listdir(class_dir):
                video_folder_path = os.path.join(class_dir, video_folder)
                if os.path.isdir(video_folder_path):
                    for file_name in os.listdir(video_folder_path):
                        if file_name.endswith(('.jpg', '.png')):
                            image_path = os.path.join(video_folder_path, file_name)
                            self.image_paths.append(image_path)
                            self.labels.append(label)

        print(f"Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = read_image(image_path).float() / 255.0  # Normalize to [0, 1]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

# Transformation for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization for pre-trained models
])

# Model loading function
def get_model(model_name='resnet'):
    if model_name == 'resnet':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: real and fake
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    else:
        raise ValueError("Supported models: 'resnet', 'vgg16', 'vgg19'")

    return model

# LSTM model with feature extraction from base model
class LSTMModel(nn.Module):
    def __init__(self, base_model, hidden_size=256, num_layers=1, num_classes=2):
        super(LSTMModel, self).__init__()
        self.base_model = base_model
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.base_model(x)
        x = x.view(batch_size, seq_len, -1)  # Flatten to fit into LSTM
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use the last LSTM output
        return out

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    best_acc = 0.0
    best_model_path = "best_model.pth"  # Path to save the best model

    # Lists to store the losses and accuracies for plotting
    train_losses, train_accuracies = [], []
    val_losses=[]
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
    
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Save the model if validation accuracy is improved
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch + 1} with accuracy {val_acc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['real', 'fake'], yticklabels=['real', 'fake'])
        plt.title(f'Confusion Matrix at Epoch {epoch + 1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        plt.close()  # Close the plot to avoid overlap

    # Plot training and validation loss and accuracy
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")

# Entry point of the script
if __name__ == "__main__":
    # Paths to data
    train_dir = 'C:\\Users\\ritul\\Downloads\\train_frames'
    val_dir = 'C:\\Users\\ritul\\Downloads\\test_frames'

    # Datasets and DataLoaders
    train_dataset = FrameDataset(train_dir, transform=transform)
    val_dataset = FrameDataset(val_dir, transform=transform)

    # Log dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Check if the datasets are empty
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: One or both datasets are empty.")
    else:
        # Proceed with DataLoader initialization
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

        # Initialize Model, Loss, and Optimizer
        model_name = 'vgg19'  # Choose: 'resnet', 'vgg16', 'vgg19'
        base_model = get_model(model_name)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(base_model.parameters(), lr=0.001)

        # Train the Model
        train_model(base_model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu')

