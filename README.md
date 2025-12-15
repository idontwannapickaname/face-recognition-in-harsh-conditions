# Face Recognition in Harsh Conditions

A deep learning project for face recognition under challenging lighting conditions, including low-light and darkened environments.

## 📋 Overview

This project implements a contrastive learning approach for face recognition that performs robustly under harsh lighting conditions. The system uses pretrained EdgeFace models and fine-tunes them with a contrastive learning objective to handle normal, low-light, and darkened facial images.

## 🎯 Features

- **Contrastive Learning**: Pretraining with contrastive loss for robust feature extraction
- **Multi-condition Support**: Handles normal, low-light, and darkened facial images
- **EdgeFace Integration**: Leverages pretrained EdgeFace models (from XXS to Base variants)
- **MTCNN Face Alignment**: Automatic face detection and alignment
- **Comprehensive Evaluation**: Multiple evaluation seeds for statistical reliability
- **PCA Visualization**: Tools for embedding visualization

## 📁 Project Structure

```
.
├── main.py                 # Main training and evaluation script
├── baseline.py             # Baseline model implementation
├── hubconf.py              # PyTorch Hub configuration
├── requirements.txt        # Python dependencies
├── dependencies.sh         # System dependencies
├── backbones/              # Model backbone implementations
│   └── timmfr.py           # TIMM-based face recognition backbone
├── checkpoints/            # Pretrained EdgeFace model checkpoints
│   ├── edgeface_xxs.pt
│   ├── edgeface_xs_q.pt
│   ├── edgeface_base.pt
│   └── ...
├── data/                   # Dataset directory
│   ├── normal/             # Normal lighting conditions
│   ├── low_light/          # Low-light conditions
│   └── darken_normal/      # Darkened images
├── eval_models/            # Saved evaluation models
├── face_alignment/         # MTCNN-based face alignment
│   ├── align.py
│   └── mtcnn_pytorch/
├── module/                 # Core modules
│   ├── augmentations.py    # Data augmentation
│   ├── data_utils.py       # Data loading utilities
│   ├── datasets.py         # Dataset classes
│   ├── evaluation.py       # Evaluation metrics
│   ├── models.py           # Model definitions
│   └── training.py         # Training loops
├── models/                 # Saved model checkpoints
└── utils/                  # Utility scripts
    ├── draw_pca.py         # PCA visualization
    └── get_embedding.py    # Feature extraction
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/idontwannapickaname/face-recognition-in-harsh-conditions.git
cd face-recognition-in-harsh-conditions
```

2. Install system dependencies:

```bash
bash dependencies.sh
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## 💾 Dataset Structure

Organize your dataset in the following structure:

```
data/
├── normal/           # Normal lighting conditions
│   ├── person1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── person2/
│       └── ...
├── low_light/        # Low-light conditions
│   └── ...
└── darken_normal/    # Darkened images
    └── ...
```

Each person should have their own folder containing their facial images under different conditions.

## 🎓 Training

### Configuration

Edit the hyperparameters in [main.py](main.py):

```python
# Data
DATA_ROOT = "data"
TRN_SPLITS = ['normal']           # Training data conditions
TST_SPLITS = ['low_light']        # Testing data conditions
BATCH_SIZE = 8

# Pre-training
PRETRAIN_EPOCHS = 50
PRETRAIN_LR = 1e-3
PRETRAIN_BATCH_SIZE = 8192

# Fine-tuning
FINETUNE_EPOCHS = 60
FINETUNE_LR = 1e-3
EMBED_DIM = 512
```

### Run Training

```bash
python main.py
```

The training process includes:

1. **Pretrain Phase**: Contrastive learning on the backbone
2. **Fine-tune Phase**: Task-specific training with unfreezing strategy

## 📊 Evaluation

The project includes evaluation scripts with multiple random seeds for statistical reliability:

- Baseline models: `baseline_model_seed_{1,10,20,100,200}_state_dict.pt`
- Proposed models: `ours_model_seed_{1,10,20,100,200}_state_dict.pt`

Evaluation metrics include:

- **Accuracy**: Classification accuracy
- **Precision, Recall, F1**: Per-class and macro-averaged metrics
- **Embedding Visualization**: PCA plots

## 🔧 Models

### Pretrained Checkpoints

The project uses EdgeFace pretrained models:

- `edgeface_xxs.pt` - Extra-extra-small variant
- `edgeface_xxs_q.pt` - Quantized XXS variant
- `edgeface_xs_q.pt` - Quantized XS variant
- `edgeface_xs_gamma_06.pt` - XS with gamma=0.6
- `edgeface_s_gamma_05.pt` - Small with gamma=0.5
- `edgeface_base.pt` - Base variant

### Custom Models

- **ContrastiveBackbone**: Backbone with contrastive learning
- **ContrastiveModel**: Full model with classification head
- **BaselineModel**: Baseline implementation for comparison

## 📈 Visualization

Generate PCA plots of embeddings:

```bash
python utils/draw_pca.py
```

Extract embeddings:

```bash
python utils/get_embedding.py
```

## 🛠️ Face Alignment

The project includes MTCNN-based face alignment:

```python
from face_alignment.align import align_face

aligned_face = align_face(image_path)
```

## 👥 Contributors

This project is developed for face recognition research under challenging lighting conditions.

## 🙏 Acknowledgments

- EdgeFace for pretrained models
- MTCNN for face detection and alignment
- TIMM library for vision transformers

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is for research and educational purposes.
