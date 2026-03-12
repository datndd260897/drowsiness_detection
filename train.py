import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
import kagglehub

# 1. Download the Dataset
print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("serenaraju/yawn-eye-dataset-new")
print("Dataset downloaded to:", dataset_path)

# Look for the exact folder containing the class subdirectories
# Often Kaggle extracts them into a subfolder like 'dataset_new/train'
base_dir = dataset_path
for root, dirs, files in os.walk(dataset_path):
    if 'Open' in dirs and 'Closed' in dirs:
        base_dir = root
        break
print(f"Using image directory: {base_dir}")

# 2. Define the Model Architecture (Same as Inference)
class EyeClassifier(nn.Module):
    def __init__(self):
        super(EyeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 18 * 18, 128)
        self.fc2 = nn.Linear(128, 2) # 2 Outputs: Closed(0), Open(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Custom Dataset to Filter Only 'Open' and 'Closed' Classes
class OpenClosedEyeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # We only care about these two folders. Closed = 0, Open = 1
        class_mapping = {'Closed': 0, 'Open': 1}
        
        for class_name, label in class_mapping.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                # Account for potential lowercase/different naming conventions in the dataset
                class_dir = os.path.join(root_dir, class_name.lower()) 
                
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB') # Convert to RGB first to ensure consistency
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 4. Data Preparation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Convert to 1-channel Grayscale
    transforms.Resize((24, 24)),                 # Resize to 24x24 to match our CNN
    transforms.ToTensor()                        # Converts to Tensor and scales pixels to [0.0, 1.0]
])

full_dataset = OpenClosedEyeDataset(root_dir=base_dir, transform=transform)
print(f"Found {len(full_dataset)} eye images (Open/Closed).")

# Split into 80% training and 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = EyeClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop
epochs = 15

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + Backward + Optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Track accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Validation Phase
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {100 * correct_train / total_train:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100 * correct_val / total_val:.2f}%")

# 7. Save the Model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/model.pth')
print("Training complete! Model weights saved to 'models/model.pth'.")