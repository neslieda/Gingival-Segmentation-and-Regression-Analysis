import json
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# JSON dosyalarının bulunduğu klasör
json_dir = r'C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum\gum\gum\labels'

# Görsellerin bulunduğu klasör
image_dir = r'C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum\gum\gum\images'

# Görselleri ve etiketleri yükle
images = []
masks = []

for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        with open(os.path.join(json_dir, json_file)) as f:
            data = json.load(f)

        # Görsel dosyasının yolunu belirle
        image_info = data['images'][0]
        image_path = os.path.join(image_dir, image_info['file_name'])

        if not os.path.exists(image_path):
            print(f"Dosya bulunamadı: {image_path}")
            continue

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
            # Maskeyi de aynı boyuta getirmek için yeniden boyutlandır
            mask = Image.fromarray(mask)
            mask = self.transform(mask)
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
class ResNet18Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Segmentation, self).__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        self.resnet.fc = nn.Identity()  # Fully connected katmanı çıkart

        # Ekstra katmanlar: özellik haritalarını al ve yeniden boyutlandır
        self.upsample = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.upsample(x)
        return x


# Modeli cihaza gönder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18Segmentation(num_classes=2).to(device)

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}/{num_epochs} Loss: {epoch_loss:.4f}')
    scheduler.step()

print('Eğitim tamamlandı.')
