import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet34, mobilenet_v3_small
from sklearn.metrics import ConfusionMatrixDisplay
from ptflops import get_model_complexity_info
from torch.utils.mobile_optimizer import optimize_for_mobile
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

        label = annotation['category_id']
        return image, mask, label


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


def get_macs(model):
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False)
    return macs


# Define models
class MobileNetModified(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v3_small(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[0].in_features, 2)
        )

    def forward(self, x):
        return self.model(x)


class ResNetModified(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet34(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 2)
        )

    def forward(self, x):
        return self.model(x)


models = {
    "mobilenet": {
        "model": MobileNetModified,
        "batch_size": 32,
        "epochs": 5
    },
    "resnet": {
        "model": ResNetModified,
        "batch_size": 128,
        "epochs": 5
    }
}


def train_model(dataloaders, model, loss_fn, optimizer, device, num_epochs):
    accuracies = {"train": [], "validation": []}
    losses = {"train": [], "validation": []}
    for epoch in range(num_epochs):
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, masks, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            accuracies[phase].append(epoch_acc)
            losses[phase].append(epoch_loss)
            print(f'Epoch {epoch}/{num_epochs - 1} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return accuracies, losses


def run_phase(dataloaders, model, loss_fn, optimizer, device, phase):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for inputs, masks, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


def load_model(model, path, name, device):
    model.load_state_dict(torch.load(os.path.join(path, name + ".pt"), map_location=device))
    with open(os.path.join(path, name + ".json"), "r") as fin:
        data = json.load(fin)
    return data


def save_model(model, path, name, accuracies, losses):
    name = os.path.join(path, name)
    torch.save(model.state_dict(), name + ".pt")
    acc_fixed = {
        item: torch.stack(accuracies[item]).tolist() if not isinstance(accuracies[item][0], float) else accuracies[item]
        for item in accuracies.keys()}
    loss_fixed = losses
    with open(name + ".json", "w") as fout:
        json.dump({"accuracies": acc_fixed, "losses": loss_fixed}, fout)


def optimize_model(model):
    example = torch.rand(1, 3, 224, 224)
    model = model.cpu()
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    return traced_script_module_optimized


def save_mobile_model(traced_script_module_optimized, path):
    traced_script_module_optimized._save_for_lite_interpreter(path)


def visualize_predictions(model, dataloaders, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))

    with torch.no_grad():
        for inputs, masks, _ in dataloaders['test']:
            inputs = inputs.to(device)
            masks = masks.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 6, images_so_far)
                ax.axis('off')
                ax.set_title(f'Prediction: {preds[j]}')
                plt.imshow(inputs.cpu().data[j].permute(1, 2, 0))
                ax = plt.subplot(num_images // 2, 6, images_so_far + num_images)
                ax.axis('off')
                ax.set_title(f'Mask')
                plt.imshow(masks.cpu().data[j].squeeze(), cmap='gray')
                ax = plt.subplot(num_images // 2, 6, images_so_far + num_images * 2)
                ax.axis('off')
                ax.set_title(f'Predicted Mask')
                plt.imshow(preds[j].cpu().numpy(), cmap='gray')

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

    model.train(mode=was_training)


# Main script
SEED = 101
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Convert user input to lowercase for model selection
model_name = input(f"Which model? ({', '.join(list(models.keys()))}) > ").lower()

if model_name not in models:
    raise ValueError(f"Invalid model name. Please choose from: {', '.join(list(models.keys()))}")

model_data = models[model_name]

model = model_data["model"]().to(device)
dataloaders = create_dataloaders(model_data["batch_size"])

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

should_train = input("Train or load model? (train, load) > ").lower() == "train"

if should_train:
    accuracies, losses = train_model(dataloaders, model, loss_fn, optimizer, device, num_epochs=model_data["epochs"])
else:
    path = input("Path of model to load > ")
    name = input("Name of model to load > ")
    data = load_model(model, path, name, device)
    accuracies = data["accuracies"]
    losses = data["losses"]

print("Testing model...")
run_phase(dataloaders, model, loss_fn, optimizer, device, "test")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(accuracies["train"], label="Train")
plt.plot(accuracies["validation"], label="Validation")
plt.legend()
plt.title("Accuracies")
plt.subplot(1, 2, 2)
plt.plot(losses["train"], label="Train")
plt.plot(losses["validation"], label="Validation")
plt.legend()
plt.title("Losses")
plt.show()

visualize_predictions(model, dataloaders, device, num_images=6)

if input("Optimize model? (y/n) > ").lower() == "y":
    print("Optimizing model...")
    traced_script_module_optimized = optimize_model(model)
    path = input("Path to save optimized model > ")
    save_mobile_model(traced_script_module_optimized, path)
    print("Optimized model saved.")
else:
    if should_train:
        if input("Save trained model? (y/n) > ").lower() == "y":
            path = input("Path to save > ")
            name = input("Name of model > ")
            save_model(model, path, name, accuracies, losses)
            print("Model saved.")
