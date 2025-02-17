import os
import json
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Define paths
image_dir = "C:/Users/edayu/PycharmProjects/Yapayzeka/dental/Diş eti akademik yapay zeka/gum2/gum/images"
json_dir = "C:/Users/edayu/PycharmProjects/Yapayzeka/dental/Diş eti akademik yapay zeka/gum2/gum/labels"

import os
import json

# Define paths
json_dir = "C:/Users/edayu/PycharmProjects/Yapayzeka/dental/Diş eti akademik yapay zeka/gum2/gum/labels"
import json
import os

# Data transforms
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
}


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Load image paths and corresponding labels
image_paths = []
labels = []

for img_file in os.listdir(image_dir):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        img_path = os.path.join(image_dir, img_file)
        json_path = os.path.join(json_dir, img_file.replace(".jpg", ".json").replace(".png", ".json"))

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                # Assuming the label is stored under a key "label" in the JSON file
                label = data["label"]
                image_paths.append(img_path)
                labels.append(label)

# Split into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2,
                                                                        random_state=42)

# Create datasets
train_dataset = CustomDataset(train_images, train_labels, transform=data_transforms["train"])
test_dataset = CustomDataset(test_images, test_labels, transform=data_transforms["test"])

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataloaders = {
    "train": train_loader,
    "test": test_loader
}


# Define the model
class ResNetModified(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(512, 2)  # Assuming binary classification
        )

    def forward(self, inputs):
        x = self.model(inputs)
        return x


# Function to train the model
def train_model(dataloaders, model, loss_fn, optimizer, device, num_epochs=30):
    accuracies = {"train": [], "test": []}
    losses = {"train": [], "test": []}
    for epoch in range(num_epochs):
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc.item())

            print(f"Epoch {epoch}/{num_epochs - 1}, {phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    return accuracies, losses


# Function to run the model on the test set
def run_phase(dataloaders, model, loss_fn, optimizer, device, phase="test"):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    all_preds = []
    all_labels = []

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

    print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    return all_preds, all_labels


# Set random seed
SEED = 101
torch.manual_seed(SEED)
np.random.seed(SEED)

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Initialize model
model = ResNetModified().to(device)

# Loss function & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
accuracies, losses = train_model(dataloaders, model, loss_fn, optimizer, device, num_epochs=30)

# Test the model
print("Testing model...")
all_preds, all_labels = run_phase(dataloaders, model, loss_fn, optimizer, device, "test")

# Plot confusion matrix
confusion_mtx = np.zeros((2, 2))
for p, l in zip(all_preds, all_labels):
    confusion_mtx[l, p] += 1

ConfusionMatrixDisplay(confusion_mtx, display_labels=[0, 1]).plot()
plt.show()


# Save model and data
def save_model(model, path, name, accuracies, losses):
    torch.save(model.state_dict(), os.path.join(path, f"{name}.pt"))
    data = {"accuracies": accuracies, "losses": losses}
    with open(os.path.join(path, f"{name}.json"), "w") as f:
        json.dump(data, f)


save_path = "C:/Users/edayu/PycharmProjects/Yapayzeka/dental/Diş eti akademik yapay zeka/models"
model_name = "resnet_modified"
save_model(model, save_path, model_name, accuracies, losses)
