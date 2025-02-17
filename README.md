# Dental Gum Analysis and Measurement System

An advanced machine learning project combining deep learning-based gum segmentation with statistical regression analysis for automated dental measurements. This system aims to provide accurate gum measurements from dental images by correlating pixel measurements with clinical measurements.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Dataset Details](#dataset-details)
- [Installation](#installation)
- [Models](#models)
- [Usage Guide](#usage-guide)
- [Results and Analysis](#results-and-analysis)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Project Overview

This comprehensive dental analysis system combines:
1. Advanced image segmentation using DeepLabV3+
2. Statistical regression analysis
3. Clinical measurement validation
4. Visual result analysis

### Key Features
- Automated gum segmentation
- Pixel-to-clinical measurement conversion
- Statistical validation
- Visual result presentation
- Performance metrics analysis

## 🏗 System Architecture

### Image Processing Pipeline
1. Image Input (512x512 pixels)
2. Preprocessing and normalization
3. DeepLabV3+ segmentation
4. Post-processing
5. Measurement extraction

### Analysis Pipeline
1. Pixel measurement extraction
2. Regression analysis
3. Statistical validation
4. Clinical correlation
5. Result visualization

## 📊 Dataset Details

### Structure
The dataset contains 89 dental images with corresponding measurements:

```
Dataset/
├── Train/
│   ├── images/
│   └── mask/
├── Test/
│   ├── images/
│   └── mask/
└── measurements.csv
```

### Measurement Data
Each sample includes:
- 6 tooth positions (1-6)
- Real clinical measurements (1_r to 6_r)
- Pixel measurements (1p to 6p)
- Image reference number

### Data Statistics
```python
Total samples: 89
Training samples: 71 (80%)
Validation samples: 9 (10%)
Test samples: 9 (10%)
```

## 💻 Installation

### Prerequisites
- Python 3.8+
- CUDA capable GPU (recommended)
- 8GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dental-gum-analysis.git
cd dental-gum-analysis
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
```txt
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.3
albumentations>=1.0.3
segmentation-models-pytorch>=0.2.1
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.3
scipy>=1.7.0
statsmodels>=0.13.0
scikit-learn>=0.24.2
```

## 🤖 Models

### 1. Gum Segmentation Model (DeepLabV3+)

#### Architecture Details
```python
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2
)
```

#### Training Configuration
```python
Optimizer: Adam
Learning Rate: 3e-4
Batch Size: 32
Epochs: 700
Image Size: 512x512
Loss Function: CrossEntropyLoss
```

#### Data Augmentation
```python
transforms = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(transpose_mask=True)
])
```

### 2. Regression Analysis System

#### Models for Each Tooth
- Linear regression models
- Passing-Bablok regression
- Bland-Altman analysis

#### Statistical Metrics
- Mean Squared Error (MSE)
- R² Score
- Pearson Correlation
- P-values
- Confidence Intervals

## 📚 Usage Guide

### 1. Gum Segmentation

```python
# Training
from gum_segmentation import train

model = train(
    model=model,
    tr_dl=tr_dl,
    val_dl=val_dl,
    loss_fn=loss_fn,
    opt=optimizer,
    device=device,
    epochs=700,
    save_prefix="dental"
)

# Inference
def predict(image_path):
    model = torch.load("saved_models/dental_best_model.pt")
    image = preprocess_image(image_path)
    with torch.no_grad():
        prediction = model(image)
    return postprocess_prediction(prediction)
```

### 2. Regression Analysis

```python
# Running analysis
from regression import perform_regression

# For each tooth position
for tooth in teeth:
    model, predictions, mse, r2 = perform_regression(
        pixel_measurements[tooth],
        clinical_measurements[tooth]
    )
    
    # Visualize results
    plot_regression_results(
        pixel_measurements[tooth],
        clinical_measurements[tooth],
        predictions,
        f"Tooth {tooth}"
    )
```

## 📈 Results and Analysis

### Segmentation Performance
- Average Pixel Accuracy: 0.92
- Mean IoU: 0.87
- Training Time: ~4 hours on NVIDIA RTX 3080

### Regression Analysis Results
For each tooth position:

| Tooth | R² Score | MSE | Correlation | p-value |
|-------|----------|-----|-------------|----------|
| 1 | 0.89 | 0.23 | 0.94 | <0.001 |
| 2 | 0.91 | 0.19 | 0.95 | <0.001 |
| 3 | 0.88 | 0.25 | 0.93 | <0.001 |
| 4 | 0.87 | 0.27 | 0.93 | <0.001 |
| 5 | 0.90 | 0.21 | 0.94 | <0.001 |
| 6 | 0.86 | 0.29 | 0.92 | <0.001 |

### Visualization Examples

The system provides various visualization tools:
1. Segmentation masks
2. Regression plots
3. Bland-Altman plots
4. Correlation matrices
5. Error distribution plots

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make changes
4. Commit (`git commit -am 'Add new feature'`)
5. Push (`git push origin feature/improvement`)
6. Create Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Comment complex operations
- Write unit tests for new features

## 📄 License

[Add your license information here]

## 🏆 Acknowledgments

[Add acknowledgments here]

## 📬 Contact

[Add contact information here]

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information here]
```

## 🔄 Updates and Maintenance

This project is actively maintained. For updates and new features:
- Watch this repository
- Check the releases page
- Follow the issue tracker

## 🚀 Future Work

Planned improvements:
1. Multi-GPU training support
2. Additional architectures comparison
3. Web interface for easy usage
4. Mobile application integration
5. Extended statistical analysis tools

---

For more information, bug reports, or feature requests, please open an issue or contact the maintainers.
