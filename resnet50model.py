import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import json
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


json_dir = r'C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum\gum\gum\labels'
image_dir = r'C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum\gum\gum\images'

images = []
masks = []

print(f"JSON dosyaları aranıyor: {json_dir}")  # Check if JSON files are found
for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        with open(os.path.join(json_dir, json_file)) as f:
            data = json.load(f)

        image_info = data['images'][0]
        image_path = os.path.join(image_dir, image_info['file_name'])

        if not os.path.exists(image_path):
            print(f"Dosya bulunamadı: {image_path}")
            continue

        print(f"İşleniyor: {image_path}")  # Print the image being processed
        image = Image.open(image_path).convert('RGB')
        images.append(image)

        annotation = data['annotations'][0]
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        segmentation = np.array(annotation['segmentation'][0]).reshape(-1, 2)
        for point in segmentation:
            x, y = point.astype(int)
            if 0 <= x < image_info['width'] and 0 <= y < image_info['height']:
                mask[y, x] = 1
        masks.append(mask)

print(f"Toplam resim sayısı: {len(images)}")  # Print the number of images and masks
print(f"Toplam maske sayısı: {len(masks)}")

# Eğitim ve test setlerine ayır
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)


# Veri seti oluşturma
class DentalDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            image = self.transform(image)
            # Maskeyi yeniden boyutlandırma ve sınıf indekslerine dönüştürme
            mask = Image.fromarray(mask)
            mask = self.transform(mask)
            mask = mask.long()  # Maskelerin uzun tam sayı formatında olması gerekiyor
        return image, mask


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


train_dataset = DentalDataset(train_images, train_masks, transform)
test_dataset = DentalDataset(test_images, test_masks, transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

import torchvision.models as models
import torch.nn as nn

# ResNet modelini yükle
resnet = models.resnet18(weights='DEFAULT')  # weights parametresi güncel kullanımı


# ResNet18'den çıkan özellik haritasının boyutları için uygun katman
import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Segmentation, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # Remove the original fully connected layer

        # Custom layers
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Adjust the target size as needed

    def forward(self, x):
        # Extract features
        x = self.resnet(x)
        print(f'After ResNet: {x.shape}')  # Debug print

        # If the output from resnet needs to be reshaped
        x = x.view(x.size(0), 2048, 1, 1)

        # Apply adaptive pooling to reshape
        x = self.adaptive_pool(x)
        print(f'After adaptive pooling: {x.shape}')  # Debug print

        # Apply custom layers
        x = self.conv1(x)
        print(f'After conv1: {x.shape}')  # Debug print
        x = self.conv2(x)
        print(f'After conv2: {x.shape}')  # Debug print

        # Upsample to the original image size
        x = self.upsample(x)
        print(f'After upsample: {x.shape}')  # Debug print

        return x



# Example usage
model = ResNet50Segmentation(num_classes=2)
input_tensor = torch.randn(8, 3, 224, 224)  # Example input tensor with batch_size=8, channels=3, height=224, width=224
output = model(input_tensor)
print(f'Output shape: {output.shape}')


import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


# Dummy dataset for illustration
class DummyDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random input image and segmentation mask
        image = torch.randn(3, 224, 224)
        mask = torch.randint(0, 2, (224, 224), dtype=torch.long)  # Random binary mask
        return image, mask


# Create dataset and dataloader
dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model, loss function, and optimizer
model = ResNet50Segmentation(num_classes=2)
criterion = nn.CrossEntropyLoss()  # Use cross entropy loss for segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)
'''
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Model çıktıları
        outputs = model(inputs)

        # Çıktıyı [batch, height, width, num_classes] formatına dönüştürme
        outputs = outputs.permute(0, 2, 3, 1)  # [batch, height, width, num_classes]

        # `labels` tensor'unun boyutunu [batch, height, width] formatına getir
        labels = labels.squeeze(1)  # Eğer `labels` tensor'u [batch, 1, height, width] formatında ise

        # Kayıp hesaplama
        loss = criterion(outputs, labels)

        # Geri yayılım ve optimizasyon
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}')

# Example usage
model = ResNet50Segmentation(num_classes=2)
input_tensor = torch.randn(8, 3, 224, 224)  # Example input tensor with batch_size=8, channels=3, height=224, width=224
output = model(input_tensor)
print(f'Output shape: {output.shape}')

# Modeli cihaza gönder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50Segmentation(num_classes=2).to(device)

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torchmetrics import Metric
import torch
import numpy as np

from torchmetrics import Metric

class RMSE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        # Boyutları eşitle
        if preds.shape != target.shape:
            preds = torch.nn.functional.interpolate(preds, size=target.shape[1:], mode='bilinear', align_corners=True)
        squared_error = torch.square(preds - target)
        self.sum_squared_error += torch.sum(squared_error)
        self.count += torch.numel(target)

    def compute(self):
        return torch.sqrt(self.sum_squared_error / self.count)



# Eğitim döngüsü
model.train()
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Model çıktıları
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Geri yayılım ve optimizasyon
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')

# RMSE hesapla
rmse = RMSE()
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        rmse.update(outputs, labels)
print(f'RMSE: {rmse.compute().item()}')
'''