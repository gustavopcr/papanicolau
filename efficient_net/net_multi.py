import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
from efficientnet_pytorch import EfficientNet

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        #img_path = img_path.replace('\\', '/')

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f'Image {img_name} not found in {self.root_dir}')

        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx, -1])  # Assuming label is already encoded as integers
        haralick_features = self.data.iloc[idx, 1:-1].values.astype(float)

        if self.transform:
            image = self.transform(image)

        haralick_features = torch.tensor(haralick_features, dtype=torch.float32)  # Ensure the correct dtype

        return image, haralick_features, label

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prepare dataset
dataset = CustomDataset(csv_file='csv/haralick_multi.csv', root_dir='output', transform=transform)

# Split the dataset into training (80%) and testing (20%) sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Prepare data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model with an additional input for Haralick features
class EfficientNetWithHaralick(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetWithHaralick, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, 128)  # Intermediate layer
        self.fc1 = nn.Linear(128 + 18, 64)  # 128 from EfficientNet, 18 from Haralick features
        self.fc2 = nn.Linear(64, num_classes)  # Output layer with num_classes units

    def forward(self, x, haralick_features):
        x = self.efficientnet(x)
        x = torch.cat((x, haralick_features), dim=1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Instantiate the model with the number of classes
num_classes = 6  # Number of classes
model = EfficientNetWithHaralick(num_classes=num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define accuracy calculation function
def calculate_accuracy(loader):
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, haralick_features, labels in loader:
            images = images.to(device)
            haralick_features = haralick_features.to(device)
            labels = labels.to(device)

            outputs = model(images, haralick_features)
            _, predicted = torch.max(outputs, 1)
            
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    return accuracy

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, haralick_features, labels) in enumerate(train_loader):
        images = images.to(device)
        haralick_features = haralick_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, haralick_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    # Calculate and print accuracy for the training and testing datasets
    training_accuracy = calculate_accuracy(train_loader)
    testing_accuracy = calculate_accuracy(test_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {100.0 * training_accuracy:.2f}%, Testing Accuracy: {100.0 * testing_accuracy:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'efficientnet_with_haralick_multiclass.pth')
print("Model saved to efficientnet_with_haralick_multiclass.pth")

# Load the model
model = EfficientNetWithHaralick(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('efficientnet_with_haralick_multiclass.pth'))
model.eval()  # Set the model to evaluation mode
print("Model loaded from efficientnet_with_haralick_multiclass.pth")

# Calculate final accuracy on the training and testing sets
final_training_accuracy = calculate_accuracy(train_loader)
final_testing_accuracy = calculate_accuracy(test_loader)
print(f'Final Training Accuracy: {100.0 * final_training_accuracy:.2f}%')
print(f'Final Testing Accuracy: {100.0 * final_testing_accuracy:.2f}%')
