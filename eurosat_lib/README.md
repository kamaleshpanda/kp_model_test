# EuroSAT Multimodal Classification Model

This library provides an easy-to-use interface for the EuroSAT Multimodal Classification model. It abstracts away the complex preprocessing steps (including morphological and graph-based features) and provides a simple API for running inference on Multispectral (.tif) and RGB (.jpg) image pairs.

## Installation

```bash
git clone <your-repo-url>
cd eurosat_lib
# Make sure you fetch the git lfs objects (the weights)
git lfs pull
pip install -e .
```

## Usage

```python
from eurosat_model import EuroSATPredictor

# Initialize the predictor
# This automatically loads the PyTorch model and pre-trained weights
predictor = EuroSATPredictor()

# Run prediction
predicted_class = predictor.predict(
    ms_image_path="path/to/image.tif",
    rgb_image_path="path/to/image.jpg"
)

print(f"Predicted class: {predicted_class}")
```
