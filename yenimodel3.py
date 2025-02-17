import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image

# Set paths
json_folder_path = r"C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum2\gum\labels"
image_folder_path = r"C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum2\gum\images"
mask_folder_path = r"C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum2\gum\masks"

# Load JSON data
json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
annotations = []
image_files = {}

for json_file in json_files:
    with open(os.path.join(json_folder_path, json_file)) as f:
        data = json.load(f)
        for image in data['images']:
            image_files[image['id']] = image['file_name']
        for annotation in data['annotations']:
            annotation['file_name'] = image_files[annotation['image_id']]
            annotations.append(annotation)

# Define transformations
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

# Custom dataset
class DentalDataset(Dataset):
    def __init__(self, annotations, image_folder, mask_folder, transform=None):
        self.annotations = annotations
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_name = os.path.join(self.image_folder, annotation['file_name'])
        image = Image.open(img_name).convert("RGB")

        mask_name = os.path.join(self.mask_folder, f"mask_{annotation['file_name']}")
        mask = Image.open(mask_name).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Create datasets
image_datasets = {
    "train": DentalDataset(annotations, image_folder_path, mask_folder_path, transform=data_transforms["train"]),
    "validation": DentalDataset(annotations, image_folder_path, mask_folder_path, transform=data_transforms["test"]),
    "test": DentalDataset(annotations, image_folder_path, mask_folder_path, transform=data_transforms["test"]),
}

def create_dataloaders(batch_size):
    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True),
        "validation": DataLoader(image_datasets["validation"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(image_datasets["test"], batch_size=batch_size, shuffle=False)
    }
    return dataloaders

# Define the ResNet-UNet model
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet34(weights='IMAGENET1K_V1')
        self.base_layers = list(base_model.children())

        # Encoder (ResNet)
        self.encoder1 = nn.Sequential(*self.base_layers[:3])  # Conv1, BN, ReLU
        self.encoder2 = nn.Sequential(*self.base_layers[3:5])  # Layer1
        self.encoder3 = self.base_layers[5]  # Layer2
        self.encoder4 = self.base_layers[6]  # Layer3
        self.encoder5 = self.base_layers[7]  # Layer4

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self._block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self._block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = self._block(128, 64)

        # Final conv layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # -> [64, 112, 112]
        enc2 = self.encoder2(enc1)  # -> [64, 56, 56]
        enc3 = self.encoder3(enc2)  # -> [128, 28, 28]
        enc4 = self.encoder4(enc3)  # -> [256, 14, 14]
        enc5 = self.encoder5(enc4)  # -> [512, 7, 7]

        # Decoder with skip connections
        dec4 = self.upconv4(enc5)  # -> [256, 14, 14]
        dec4 = torch.cat((dec4, enc4), dim=1)  # -> [512, 14, 14]
        dec4 = self.decoder4(dec4)  # -> [256, 14, 14]

        dec3 = self.upconv3(dec4)  # -> [128, 28, 28]
        dec3 = torch.cat((dec3, enc3), dim=1)  # -> [256, 28, 28]
        dec3 = self.decoder3(dec3)  # -> [128, 28, 28]

        dec2 = self.upconv2(dec3)  # -> [64, 56, 56]
        dec2 = torch.cat((dec2, enc2), dim=1)  # -> [128, 56, 56]
        dec2 = self.decoder2(dec2)  # -> [64, 56, 56]

        dec1 = self.upconv1(dec2)  # -> [64, 112, 112]
        dec1 = torch.cat((dec1, enc1), dim=1)  # -> [128, 112, 112]
        dec1 = self.decoder1(dec1)  # -> [64, 112, 112]

        output = torch.sigmoid(self.final(dec1))  # -> [1, 112, 112]
        return output

# Modeli tanımlama
model = ResNetUNet()

# Kayıp fonksiyonu ve optimizasyon
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Eğitim için gerekli ayarlar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Model configuration
model_data = {
    "resnet_unet": {
        "model": ResNetUNet,
        "batch_size": 8,
        "epochs": 10
    }
}

def train_model(dataloaders, model, loss_fn, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0

            for inputs, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, masks)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'Epoch {epoch}/{num_epochs - 1} {phase} Loss: {epoch_loss:.4f}')

def visualize_predictions(model, dataloaders, device, num_images=6):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))

    with torch.no_grad():
        for inputs, masks in dataloaders['test']:
            inputs = inputs.to(device)
            masks = masks.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 3, images_so_far)
                ax.axis('off')
                ax.set_title(f'Input Image')
                plt.imshow(inputs.cpu().data[j].permute(1, 2, 0))

                ax = plt.subplot(num_images // 2, 3, images_so_far + num_images)
                ax.axis('off')
                ax.set_title(f'Ground Truth')
                plt.imshow(masks.cpu().data[j].squeeze(), cmap='gray')

                ax = plt.subplot(num_images // 2, 3, images_so_far + num_images * 2)
                ax.axis('off')
                ax.set_title(f'Predicted Mask')
                plt.imshow(preds[j].cpu().squeeze(), cmap='gray')

                if images_so_far == num_images:
                    return

# Main script
SEED = 101
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load model
model_name = "resnet_unet"
model_class = model_data[model_name]["model"]
model = model_class().to(device)

# Set parameters
batch_size = model_data[model_name]["batch_size"]
epochs = model_data[model_name]["epochs"]
dataloaders = create_dataloaders(batch_size)

# Set loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the model
train_model(dataloaders, model, loss_fn, optimizer, device, epochs)

# Visualize predictions
visualize_predictions(model, dataloaders, device)
