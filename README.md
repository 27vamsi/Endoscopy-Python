# 🏥 Medical Image Classification System

<div align="center">

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Medical AI](https://img.shields.io/badge/Medical-AI-green.svg)
![Binary Classification](https://img.shields.io/badge/Classification-Binary-purple.svg)

*Advanced medical image classification for gastrointestinal conditions* 🔬

</div>

## ✨ Features

- 🎯 **Multi-Class Classification** - 10 medical conditions detection
- ⚖️ **Dataset Balancing** - Automatic upsampling/downsampling for equal class distribution
- 🧠 **Individual Binary Models** - Separate CNN for each condition (One-vs-All approach)
- 📊 **Real-time Prediction** - Classify new medical images instantly
- 📈 **Training Visualization** - Accuracy and loss plots for each model
- 🗂️ **Organized Output** - Structured file management and model saving

## 🏥 Medical Conditions Detected

The system can classify the following gastrointestinal conditions:

| Condition | Description | Model Performance |
|-----------|-------------|-------------------|
| **Angioectasia** | Vascular malformations | ~50% accuracy* |
| **Bleeding** | Active bleeding detection | ~84% accuracy |
| **Erosion** | Mucosal erosions | ~78% accuracy |
| **Erythema** | Inflammation/redness | ~74% accuracy |
| **Foreign Body** | Foreign object detection | ~79% accuracy |
| **Lymphangiectasia** | Lymphatic vessel dilation | ~57% accuracy* |
| **Normal** | Healthy tissue | ~76% accuracy |
| **Polyp** | Tissue growths | ~84% accuracy |
| **Ulcer** | Mucosal ulcerations | ~93% accuracy |
| **Worms** | Parasitic infections | ~95% accuracy |

*\*Some conditions show lower accuracy due to subtle visual features*

## 🚀 Quick Start

### Installation
```bash
pip install tensorflow scikit-learn matplotlib pandas
```

### Basic Usage
```python
# The system automatically:
# 1. Balances your dataset (500 samples per class)
# 2. Trains individual binary models for each condition
# 3. Classifies new images using ensemble approach

# Just run the script with your data structure:
python medical_classifier.py
```

## 📁 Data Structure

### Input Format
```
training/
├── Angioectasia/
│   ├── image1.jpg
│   └── image2.jpg
├── Bleeding/
├── Erosion/
├── Erythema/
├── Foreign Body/
├── Lymphangiectasia/
├── Normal/
├── Polyp/
├── Ulcer/
└── Worms/

Images/                    # Test images for classification
├── test1.jpg
└── test2.jpg
```

### Output Structure
```
Balanced_Data__is/
├── Angioectasia/
│   ├── positive/         # 500 balanced positive samples
│   └── negative/         # 500 balanced negative samples
├── Bleeding/
└── [other conditions...]

Models/
├── Angioectasia_model.keras
├── Bleeding_model.keras
└── [other models...]
```

## ⚙️ Configuration

### Model Architecture
```python
# CNN Architecture per condition
model = Sequential([
    Input(shape=(224, 224, 3)),
    Rescaling(1./255),
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])
```

### Training Parameters
```python
IMAGE_SIZE = 224           # Input image dimensions
BATCH_SIZE = 32           # Training batch size
EPOCHS = 10               # Training epochs
target_count = 500        # Samples per class after balancing
```

## 📊 Results Summary

### Training Performance
- **Angioectasia**: Early stopping due to poor performance (~50% accuracy)
- **Bleeding**: Strong performance with 84% validation accuracy  
- **Erosion**: Stable training reaching 78% accuracy
- **Erythema**: Gradual improvement to 74% accuracy
- **Foreign Body**: Progressive learning achieving 79% accuracy
- **Lymphangiectasia**: Challenging class with 57% accuracy
- **Normal**: Good baseline performance at 76% accuracy
- **Polyp**: Excellent results with 84% accuracy
- **Ulcer**: Outstanding performance at 93% accuracy  
- **Worms**: Best performance with 95%+ accuracy

### Classification Strategy
The system uses a **One-vs-All ensemble approach**:
1. Each condition gets its own binary classifier
2. For new images, all 10 models make predictions
3. The condition with highest confidence score wins
4. Images are automatically sorted into predicted folders

## 🎯 Key Features

### Dataset Balancing
- **Upsampling**: Classes with <500 images are upsampled with replacement
- **Downsampling**: Classes with >500 images are randomly sampled  
- **Binary Format**: Each class vs. all others for focused learning

### Data Augmentation
- Horizontal and vertical flipping
- Random rotation (±20%)
- Automatic rescaling to [0,1] range

### Training Optimizations
- **Early Stopping**: Prevents overfitting (patience=5)
- **Learning Rate Reduction**: Adaptive LR with ReduceLROnPlateau
- **Best Weights Restoration**: Saves optimal model state

## 🔬 Medical Applications

### Clinical Use Cases
- **Endoscopy Screening**: Automated analysis of endoscopic images
- **Diagnostic Support**: AI-assisted diagnosis for gastroenterologists  
- **Research Tool**: Large-scale medical image analysis
- **Training Aid**: Educational tool for medical students

### Performance Notes
- **High Accuracy Classes**: Worms, Ulcer, Polyp show excellent detection
- **Challenging Classes**: Angioectasia, Lymphangiectasia need improvement
- **Balanced Performance**: Most conditions achieve 70%+ accuracy

## ⚠️ Important Notes

### Medical Disclaimer
- This system is for **research and educational purposes only**
- **Not intended for clinical diagnosis** without expert validation
- Results should always be reviewed by qualified medical professionals
- Regulatory approval required for clinical deployment

### Technical Limitations
- Performance varies significantly by condition type
- Some classes may need larger/better quality datasets
- Binary approach may miss complex multi-condition cases

## 🛠️ Customization

### Adjusting Balance Target
```python
# Change target samples per class
prepare_balanced_dataset_per_class(dataset_dir, output_dir, target_count=1000)
```

### Model Architecture Modifications
```python
# Add more layers or change architecture
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))  # Add regularization
```

### Training Parameters
```python
EPOCHS = 20              # Longer training
BATCH_SIZE = 16          # Smaller batches for limited memory
IMAGE_SIZE = 299         # Higher resolution
```

---

**🩺 Advancing Medical AI through Computer Vision** • **📧 Contact for Research Collaboration** • **⭐ Star if this helps your medical research!**
