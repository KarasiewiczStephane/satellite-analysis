# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Satellite Image Analysis Platform — a PyTorch deep learning system for land use classification (10-class EuroSAT) and temporal change detection from satellite imagery. Includes geospatial export (GeoJSON/GeoTIFF), area statistics, and a Streamlit dashboard with Folium maps.

## Commands

```bash
make install          # pip install -r requirements.txt
make test             # pytest tests/ -v --tb=short --cov=src
make lint             # ruff check + ruff format on src/ and tests/
make run              # python -m src.main
make dashboard        # streamlit run src/dashboard/app.py --server.port 8501
make docker-compose-up  # docker-compose up -d (dashboard on port 8501)
```

Run a single test file: `pytest tests/test_models/test_classifier.py -v`
Run a single test: `pytest tests/test_models/test_classifier.py::test_name -v`

System deps for geospatial (needed on bare metal / CI): `libgdal-dev libgeos-dev libproj-dev`

## Architecture

**Data pipeline** (`src/data/`): `downloader.py` fetches EuroSAT zip → `preprocessor.py` provides `EuroSATDataset` (PyTorch Dataset supporting GeoTIFF and JPEG) with stratified train/val/test splits → `geospatial.py` wraps rasterio loading into `SatelliteImage` dataclass with CRS/transform metadata → `augmentation.py` for transforms.

**Models** (`src/models/`): `LandUseClassifier` wraps ResNet-50 (torchvision) or EfficientNet-B0 (timm/torchvision) with configurable input channels (RGB or 4-band). `Trainer` handles AMP training, early stopping, checkpointing to `checkpoints/`. `ChangeDetector` computes difference maps between temporal image pairs and uses `ChangeClassifier` (binary CNN) for change detection. `evaluator.py` for metrics.

**Export** (`src/export/`): `geojson_writer.py` vectorizes classification masks into GeoDataFrame polygons via `rasterio.features.shapes`. `geotiff_writer.py` writes raster outputs.

**Analysis** (`src/analysis/statistics.py`): Computes per-class areas in m² from classification masks + affine transforms, tracks temporal changes as DataFrames, exports CSV/JSON reports.

**Dashboard** (`src/dashboard/app.py`): Streamlit app with Folium map integration and Plotly charts.

**CLI** (`src/main.py`): argparse with subcommands: `train`, `evaluate`, `predict`, `download`.

## Configuration

Singleton `Config` class in `src/utils/config.py` loads `configs/config.yaml` with dot-notation access: `config.get("model.learning_rate")`. All modules import the shared `config` instance. For tests, use the `mock_config` fixture from `tests/conftest.py` which calls `config.reset()` + `config.load(tmp_path)`.

## Key Conventions

- **Python 3.11+**, ruff with line-length 100 (not 88), rules: E/W/F/I/B/C4/UP, E501 ignored
- `pythonpath = ["."]` is NOT set in pyproject.toml — imports use `from src.xxx import yyy`
- Coverage omits `src/dashboard/*`
- EuroSAT classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- Image tensors are `(C, H, W)` format throughout; default image size is 64×64
- Checkpoints saved as dicts with keys: `model_state_dict`, `optimizer_state_dict`, `best_val_loss`, `history`
- Geospatial operations use rasterio `Affine` transforms and `CRS` objects; geographic CRS uses 111,320 m/degree approximation for area calculations
