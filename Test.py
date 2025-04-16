import os
import subprocess
import threading
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from tqdm import tqdm


# ===== NVIDIA GPU MONITORING =====
class GPUMonitor:
    def __init__(self, interval=2):
        self.interval = interval
        self.stop_signal = False

    def monitor_gpu(self):
        """Run nvidia-smi in a loop"""
        while not self.stop_signal:
            try:
                result = subprocess.run(['nvidia-smi'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)
                os.system('cls' if os.name == 'nt' else 'clear')
                print("=== NVIDIA GPU Monitoring ===")
                print(result.stdout)
                if result.stderr:
                    print("Error:", result.stderr)
            except Exception as e:
                print(f"Monitoring error: {str(e)}")
            time.sleep(self.interval)

    def start(self):
        """Start monitoring in background thread"""
        self.thread = threading.Thread(target=self.monitor_gpu, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring"""
        self.stop_signal = True
        self.thread.join()


# ===== MODEL ARCHITECTURE =====
class CNNClassifier(nn.Module):
    def __init__(self, img_size):
        super(CNNClassifier, self).__init__()
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        # Calculate the flattened size after the feature extractor
        # Each MaxPool2d(2) halves the spatial dimensions.
        # There are 4 pooling layers so the scaling factor is 2^4 = 16.
        flattened_size = 256 * (img_size // 16) * (img_size // 16)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.SiLU(),  # Swish activation
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        print(f"[DEBUG] Training batch {batch_idx+1}/{len(loader)}")
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.float32).unsqueeze(1)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, all_preds, all_labels




def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, all_preds, all_labels


if __name__ == '__main__':
    # ===== GPU CONFIGURATION =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using GPU!")
        #monitor = GPUMonitor()
        #monitor.start()
    else:
        print("Using CPU!")

    # ===== DATA PREPARATION =====
    IMG_SIZE = 250
    BATCH_SIZE = 32

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_dir = "dataset"  # Ensure dataset/chihuahua and dataset/muffin exist
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    split = int(0.2 * num_samples)
    random.seed(42)
    random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    from torch.utils.data.sampler import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_ds = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(root=data_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4, pin_memory=True)

    # ===== MODEL SETUP =====
    model = CNNClassifier(IMG_SIZE).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Initialize GradScaler only if CUDA is available
    if device.type == 'cuda':
        # Use the recommended signature for CUDA devices.
        scaler = torch.amp.GradScaler(device='cuda')
    else:
        scaler = None

    num_epochs = 25

    # ===== TRAINING =====
    print("\n[INFO] Starting training with GPU acceleration using PyTorch...")
    for epoch in range(num_epochs):
        print(f"\n🚀 Epoch {epoch+1}/{num_epochs}")
        train_loss, train_preds, train_labels = train_epoch(model, train_loader, optimizer, scaler, criterion, device)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        print(f"📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


    # ===== EVALUATION =====
    print("\n[INFO] Evaluating model...")
    val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    binary_preds = [1 if p >= 0.5 else 0 for p in val_preds]
    print(f"Test Loss: {val_loss:.4f}")
    print("\n📋 Classification Report:\n")
    print(classification_report(val_labels, binary_preds))

    # ===== TEST SAMPLE IMAGES =====
    print("\n[INFO] Testing sample images...")
    from PIL import Image

    def predict_image(model, image_path, transform):
        model.eval()
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            output = model(image)
            prob = torch.sigmoid(output)
        return prob.item()

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_images = {
        "Chihuahua": "dataset/chihuahua/chihuahua_25.JPG",
        "Muffin": "dataset/muffin/img_0_74.jpg"
    }

    for name, path in test_images.items():
        prob = predict_image(model, path, test_transform)
        result = "Muffin" if prob >= 0.5 else "Chihuahua"
        confidence = prob if prob >= 0.5 else 1 - prob
        print(f"→ Prediction for '{name}': {result} (confidence: {confidence*100:.2f}%)")

    # ===== SAVE THE MODEL =====
    model_save_path = 'models/chihuahua_muffin_modelV4.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print("\n[SUCCESS] Model saved!")

    # Stop GPU monitoring if active
    if device.type == "cuda":
        monitor.stop()
