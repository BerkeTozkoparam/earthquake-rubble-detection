# Earthquake Rubble Detection Decision Support System

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-earthquake--rubble--detection-blue?logo=github)](https://github.com/BerkeTozkoparam/earthquake-rubble-detection)

A production-ready decision support prototype for earthquake rubble detection in wide-area aerial/satellite imagery using YOLO11 and EfficientNet-B0 CNN fusion with comprehensive audit trail logging.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Details](#model-details)
- [Audit Trail System](#audit-trail-system)
- [API Reference](#api-reference)
- [Performance Metrics](#performance-metrics)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

This system combines two complementary deep learning models to detect earthquake rubble in wide-area imagery:

- **YOLO11s**: Real-time object detection for rubble bounding boxes
- **EfficientNet-B0 CNN**: Fine-grained tile classification for missed detections

The dual-model approach achieves **98% coverage** on test data (207/211 labeled rubble areas) by leveraging each model's strengths. A comprehensive audit trail system tracks all operations for compliance and reproducibility.

**Disclaimer**: This is a decision support prototype for human expert review. It is **not** an automated operational decision system and requires human validation before any response actions.

---

## Features

### Core Detection
- ✅ YOLO11s bounding box detection (single-class rubble)
- ✅ EfficientNet-B0 tile classification (128×128 px, 64 px stride)
- ✅ Intelligent fusion: both models + CNN-only + YOLO-only categories
- ✅ Raw tile preservation for decision logic (NMS display-only)

### Risk Assessment
- ✅ Automatic severity level assignment (🔴 CRITICAL / 🟠 MEDIUM / 🟡 LOW)
- ✅ Confidence-based thresholding (YOLO 0.25, CNN 0.80)
- ✅ Operational action recommendations per alert

### Audit & Compliance
- ✅ Comprehensive event logging (analysis start, inference, fusion, alerts)
- ✅ Session tracking with timestamps (ISO 8601)
- ✅ Master audit log (append-only JSONL format)
- ✅ Streamlit audit trail viewer with filtering and export

### Data Export
- ✅ CSV export (coordinates, sources, scores, actions)
- ✅ JSON export (full metadata, detection details)
- ✅ Audit log export (session-level tracing)

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│  Image Upload → Analysis → Risk Assessment → Export  │
└──────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    ┌───────────┐   ┌──────────┐   ┌──────────────┐
    │   YOLO    │   │   CNN    │   │ Alert Fusion │
    │ Detector  │   │  Tiler   │   │   Manager    │
    │  (640×640)│   │(128×128) │   │              │
    └───────────┘   └──────────┘   └──────────────┘
        │                │                 │
        └─────────────────┼─────────────────┘
                          │
                  ┌───────────────┐
                  │ Audit Logger  │
                  │  (all events) │
                  └───────────────┘
```

### Model Pipeline

**YOLO Detection** → **CNN Scanning** → **Fusion** → **Risk Assessment** → **Audit Log**

1. **YOLO Inference**: Input image (any size) → Rubble bounding boxes
2. **CNN Tile Scanning**: Sliding window (128×128, stride 64) → Positive tiles
3. **Fusion Logic**: IoU matching → Alert categories (both/CNN-only/YOLO-only)
4. **NMS Display**: IoU 0.30 filter for UI visualization (raw tiles preserved)
5. **Risk Levels**: Confidence averaging → Severity assignment
6. **Audit Logging**: All events logged with timestamp and metadata

---

## Installation

### Requirements

- Python 3.9+
- 8GB RAM (16GB recommended for batch processing)
- Optional: CUDA 11.8+ for GPU acceleration

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BerkeTozkoparam/earthquake-rubble-detection.git
   cd earthquake-rubble-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model files** (not included in repo):
   - `yolo_rubble_best.pt` (18 MB)
   - `earthquake_tile_cnn_state_dict.pt` (16 MB)
   
   Place in the project root directory.

4. **Verify configuration**:
   ```bash
   python -c "from earthquake_detector import ModelManager; m = ModelManager('inference_config.json'); print('✓ Models loaded')"
   ```

---

## Quick Start

```bash
# Start the application
streamlit run app.py

# Open browser to http://localhost:8501
# Upload an image → View results → Export data
```

### Demo Image

A sample `test_earthquake.jpg` (768×1024 px) is included for testing.

---

## Usage

### Web Interface (Streamlit)

1. **Load Image**: Upload JPG, PNG, or TIFF (up to 200 MB)
2. **Auto-Analysis**:
   - YOLO detects rubble boxes
   - CNN scans 128×128 tiles
   - Fusion merges results
3. **Review Results**:
   - Risk assessment (severity level)
   - Alert table (coordinates, scores, recommended actions)
   - Colored visualization (Blue=YOLO, Orange=CNN, Green=Both)
4. **Export**:
   - CSV/JSON with full metadata
   - Audit log for compliance

### Sidebar Configuration

- **CNN Probability Threshold** (0.5–1.0, default 0.80)
- **NMS IoU Threshold** (0.1–0.5, default 0.30) — *Display only*
- **CNN Batch Size** (16–64, default 32) — *Memory management*

### Audit Trail Viewer

Open **"📋 Denetim İzi (Audit Trail)"** expander to:
- View event summary (total events, breakdown)
- Filter by event type
- See JSON details for each event
- Download session audit log

---

## Model Details

### YOLO11s Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | YOLO11 small |
| Input Size | 640×640 |
| Confidence Threshold | 0.25 |
| Classes | 1 (Collapsed building / rubble) |
| Framework | Ultralytics 8.4.163 |

### EfficientNet-B0 Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | EfficientNet-B0 |
| Classes | 2 (clear_candidate, rubble) |
| Tile Size | 128×128 px |
| Stride | 64 px |
| Probability Threshold | 0.80 (rubble class) |
| Preprocessing | Resize to 224×224 + ImageNet normalization |
| Framework | PyTorch 2.11.0, torchvision 0.26.0 |

### Preprocessing

```python
Transform = Compose([
    Resize(size=(224, 224), interpolation=bilinear),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], 
              std=[0.229, 0.224, 0.225])  # ImageNet
])
```

---

## Audit Trail System

### Logged Events

| Event | Details |
|-------|---------|
| `ANALYSIS_START` | Image metadata (size, filename) |
| `YOLO_INFERENCE` | Detections count, confidence threshold, duration |
| `CNN_SCANNING` | Total tiles, positive tiles, threshold, duration |
| `FUSION_COMPLETE` | Alert counts by category, NMS parameters |
| `ALERT_GENERATED` | Coordinates, severity, scores, recommended action |
| `DATA_EXPORT` | Format, record count, filename |
| `ERROR` | Exception type, message, context |

### File Structure

```
audit_logs/
├── session_20260926_144530.json      # Formatted session log
├── session_20260926_144545.json
└── master_audit.jsonl                # Append-only master log
```

### Example Audit Entry

```json
{
  "timestamp": "2026-09-26T14:45:30.123456",
  "session_id": "20260926_144530",
  "event_type": "ALERT_GENERATED",
  "level": "INFO",
  "details": {
    "description": "Alert: 🔴 CRITICAL - Both",
    "alert_id": "alert_001",
    "category": "both",
    "coordinates": {"x": 150, "y": 200},
    "severity": "🔴 CRITICAL",
    "yolo_score": 0.8234,
    "cnn_score": 0.8521,
    "recommended_action": "Damaged building support required; rapid evacuation"
  }
}
```

---

## API Reference

### `ModelManager`

```python
from earthquake_detector import ModelManager

manager = ModelManager("inference_config.json")
manager.load_yolo("yolo_rubble_best.pt")
manager.load_cnn("earthquake_tile_cnn_state_dict.pt")
```

### `YOLODetector`

```python
detector = YOLODetector(manager)
detections = detector.detect(image_array)
# Returns: [{x, y, w, h, confidence, class_name}, ...]
```

### `CNNTiler`

```python
tiler = CNNTiler(manager)
tiles = tiler.scan(image_array, batch_size=32)
# Returns: [{x, y, tile_size, rubble_probability, class_name}, ...]
```

### `AlertManager`

```python
alert_manager = AlertManager()
fusion = alert_manager.fuse(yolo_dets, cnn_dets, iou_threshold=0.3)
# Returns: {both, cnn_only, yolo_only, cnn_raw, cnn_nms}
```

### `AuditLogger`

```python
from audit_logger import AuditLogger

audit = AuditLogger()
audit.log_analysis_start("image.jpg", (640, 480))
audit.log_yolo_inference("image.jpg", 3, 0.061, 0.25)
session_file = audit.save_session()
```

---

## Performance Metrics

### Test Dataset (44 positive images, 211 labeled rubble areas)

| Model | Detection Rate | Notes |
|-------|-----------------|-------|
| YOLO11s | 175/211 (83%) | Fast, real-time capable |
| EfficientNet-B0 CNN | 199/211 (94%) | Comprehensive, slower |
| **Fusion (NMS pre)** | **207/211 (98%)** | Best coverage |
| Fusion (NMS post) | 206/211 (98%) | Display visualization |

### Inference Time

| Component | Time | Device |
|-----------|------|--------|
| YOLO (640×640) | ~61 ms | CPU |
| CNN (full image scan) | ~0.5–2 s | CPU |
| Fusion | <50 ms | CPU |

**Note**: Performance on new data may differ. Test metrics are reference only.

---

## Limitations

1. **Minimum Image Size**: Images <128 px cannot be CNN-scanned (warning issued)
2. **Edge Tiles**: Partial tiles at image boundaries are not processed
3. **Performance Variance**: Test metrics do not transfer to new domains
4. **Human Review Required**: All alerts require expert validation
5. **Not Operational**: System is **not** approved for automated response dispatch
6. **GPU Optional**: Inference works on CPU but is slower (~1–2 sec per image)

---

## Severity Levels & Recommended Actions

### 🔴 CRITICAL
- **Trigger**: Both YOLO + CNN detect (avg confidence ≥0.85)
- **Action**: Damaged building support required; rapid evacuation
- **Priority**: Immediate response

### 🟠 MEDIUM
- **Trigger**: CNN detects, YOLO misses (prob ≥0.85) OR both models <0.85
- **Action**: Detailed investigation; operational readiness
- **Priority**: High priority review

### 🟡 LOW
- **Trigger**: YOLO only (confidence <0.75)
- **Action**: Verification after detailed review
- **Priority**: Routine inspection

---

## License

MIT License — See LICENSE file for details.

---

## Citation

If you use this system in research or production, please cite:

```bibtex
@software{earthquake_rubble_2026,
  author = {Tozkoparam, Berke Baran},
  title = {Earthquake Rubble Detection Decision Support System},
  year = {2026},
  url = {https://github.com/BerkeTozkoparam/earthquake-rubble-detection}
}
```

---

## Support & Issues

For questions, bug reports, or contributions:
- GitHub Issues: [earthquake-rubble-detection/issues](https://github.com/BerkeTozkoparam/earthquake-rubble-detection/issues)
- Email: berkebaran00@gmail.com

---

**Last Updated**: September 26, 2026  
**Status**: Production-Ready (Prototype Phase)
