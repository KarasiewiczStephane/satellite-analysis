# Satellite Image Analysis Platform

> Deep learning system for land use classification and change detection from satellite imagery

[![CI](https://github.com/KarasiewiczStephane/satellite-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/KarasiewiczStephane/satellite-analysis/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A production-ready deep learning platform for analyzing satellite imagery, featuring:

- **Land Use Classification** - 10-class classification using ResNet-50 / EfficientNet-B0 on EuroSAT
- **Change Detection** - Multi-temporal analysis to detect land use transitions
- **Geospatial Export** - GeoJSON vector polygons and GeoTIFF rasters with proper CRS
- **Interactive Dashboard** - Streamlit + Folium web interface with Plotly charts
- **Area Statistics** - Per-class area calculation in m2, temporal change tracking

## Architecture

```
+-------------------------------------------------------------------+
|                   Satellite Analysis Platform                      |
+-------------------------------------------------------------------+
|                                                                    |
|  Data Pipeline        Models             Dashboard                 |
|  +------------+    +-------------+    +--------------------+       |
|  | Download   |    | ResNet-50   |    | Streamlit          |       |
|  | Preprocess |    | EfficientNet|    | + Folium Map       |       |
|  | Augment    |    | Change Det. |    | + Plotly Charts    |       |
|  | Split      |    | Evaluator   |    +--------------------+       |
|  +------------+    +-------------+                                 |
|                                       Export                       |
|  Geospatial                          +--------------------+        |
|  (rasterio, geopandas)              | GeoJSON / GeoTIFF  |        |
|                                      | Statistics / CSV   |        |
|                                      +--------------------+        |
+-------------------------------------------------------------------+
```

## Results

### Classification Performance

| Model | Accuracy | F1 (Macro) | Parameters |
|-------|----------|------------|------------|
| ResNet-50 | ~94% | ~0.94 | 23.5M |
| EfficientNet-B0 | ~93% | ~0.93 | 4.0M |

*Results on EuroSAT RGB dataset with default hyperparameters.*

## Quick Start

### 1. Install

```bash
git clone https://github.com/KarasiewiczStephane/satellite-analysis.git
cd satellite-analysis

python -m venv venv
source venv/bin/activate

make install
# or: pip install -r requirements.txt
```

> **System dependencies (Linux):** geospatial libraries require `libgdal-dev libgeos-dev libproj-dev`.

### 2. Download the EuroSAT Dataset

A handful of sample images ship in `data/sample/` so you can explore the dashboard
immediately. To download the full EuroSAT dataset (~90 MB) for training:

```bash
python -m src.main download
```

This fetches the EuroSAT zip into `data/raw/` and extracts it automatically.

### 3. Launch the Dashboard

```bash
make dashboard
# or: streamlit run src/dashboard/app.py --server.port 8501
```

Open **http://localhost:8501** in your browser. The dashboard supports:
- Single image classification with probability charts
- Change detection between two temporal images
- Time series analysis across multiple images
- Image upload directly from the sidebar

### 4. Train a Model (optional)

```bash
python -m src.main train --architecture resnet50 --epochs 50
```

Training saves checkpoints to `checkpoints/`. The dashboard automatically loads
`checkpoints/best_model.pt` when available.

### 5. CLI Commands

```bash
python -m src.main --help        # Show all commands
python -m src.main download      # Download EuroSAT dataset
python -m src.main train         # Train classifier (supports --architecture, --epochs)
python -m src.main evaluate      # Evaluate a trained model
python -m src.main predict       # Run prediction on an image (supports --image)
```

### Docker

```bash
make docker-compose-up
# Dashboard at http://localhost:8501

# GPU support (requires nvidia-docker):
make docker-gpu
```

## Usage Examples

### Classify a Single Image

```python
from src.models.classifier import LandUseClassifier
import torch

model = LandUseClassifier(num_classes=10)
checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

image_tensor = ...  # (1, 3, 64, 64)
with torch.no_grad():
    output = model(image_tensor)
    predicted_class = output.argmax(dim=1).item()
```

### Export Classification as GeoJSON

```python
from src.export.geojson_writer import export_geojson

export_geojson(
    classification_mask,
    transform,
    "EPSG:4326",
    "output/classification.geojson",
)
```

### Calculate Land Use Statistics

```python
from src.analysis.statistics import calculate_statistics

stats = calculate_statistics(mask, transform, crs)
print(f"Forest: {stats.class_distribution['Forest']:.1f}%")
```

## Project Structure

```
satellite-analysis/
├── src/
│   ├── data/           # Data loading, preprocessing, augmentation
│   ├── models/         # Classifier, trainer, evaluator, change detector
│   ├── export/         # GeoJSON and GeoTIFF writers
│   ├── analysis/       # Area statistics and temporal tracking
│   ├── dashboard/      # Streamlit web application
│   ├── utils/          # Config loader, structured logging
│   └── main.py         # CLI entry point
├── tests/              # Comprehensive test suite (100+ tests)
├── configs/            # YAML configuration
├── .github/workflows/  # CI pipeline
├── Dockerfile          # Multi-stage Docker build
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml      # Ruff, pytest, coverage config
```

## Tech Stack

- **Deep Learning**: PyTorch, torchvision, timm
- **Geospatial**: rasterio, geopandas, shapely, pyproj
- **Dashboard**: Streamlit, Folium, Plotly
- **Testing**: pytest, pytest-cov (>90% coverage)
- **Quality**: ruff, pre-commit
- **CI/CD**: GitHub Actions, Docker

## Development

```bash
make test             # pytest tests/ -v --tb=short --cov=src
make lint             # ruff check + ruff format on src/ and tests/
make clean            # Remove __pycache__ and .pyc files
make run              # Show CLI help (python -m src.main)

# Full quality check
pre-commit run --all-files
```


## Author

**Stéphane Karasiewicz** — [skarazdata.com](https://skarazdata.com) | [LinkedIn](https://www.linkedin.com/in/stephane-karasiewicz/)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [EuroSAT Dataset](https://github.com/phelber/eurosat) by Helber et al.
- Sentinel-2 satellite imagery from ESA Copernicus programme
