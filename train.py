import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kagglehub

# 1. Custom Dataset for MRL Eye Dataset (Updated for Kagglehub & Subfolders)
class MRLEyeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        
        # os.walk safely looks through all subfolders for .png files
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith('.png'):
                    self.image_paths.append(os.path.join(dirpath, f))

        print(f"Found {len(self.image_paths)} images in the dataset.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L') # Convert to Grayscale
        
        # Extract just the filename from the full path
        img_name = os.path.basename(img_path)
        
        # MRL format: subjectID_imageID_gender_glasses_eyeState_reflections_lighting_sensor.png
        # eyeState is at index 4 (0 = closed, 1 = open)
        label = int(img_name.split('_')[4])
        label = torch.tensor([label], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, label

# 2. Define a Lightweight CNN
class EyeClassifier(nn.Module):
    def __init__(self):
        super(EyeClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 128), nn.ReLU(), # 24x24 input becomes 3x3
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid() # Output: probability of being OPEN
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    # Fetch dataset using kagglehub
    print("Fetching dataset via kagglehub...")
    dataset_path = kagglehub.dataset_download("akashshingha850/mrl-eye-dataset")
    print("Path to dataset files:", dataset_path)

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((24, 24)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load data using the kagglehub path
    dataset = MRLEyeDataset(root_dir=dataset_path, transform=transform)
    
    # We use num_workers=0 here for broader compatibility on Mac, 
    # but you can bump it up to 2 or 4 to speed up data loading if desired.
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    # Initialize model, loss, and optimizer
    # Use MPS (Metal Performance Shaders) if available on Mac (Apple Silicon/newer Intel)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = EyeClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Simple Training Loop
    print("Starting training...")
    for epoch in range(5): # Train for 5 epochs
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

    # Save the trained model weights
    torch.save(model.state_dict(), 'eye_model.pth')
    print("Model saved as eye_model.pth")