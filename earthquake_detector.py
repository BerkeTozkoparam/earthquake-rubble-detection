import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("ultralytics not installed")
    YOLO = None

try:
    import torchvision.models as models
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
except ImportError:
    logger.warning("torchvision not installed")


class ModelManager:
    """Load and cache YOLO and CNN models."""

    def __init__(self, config_path: str, device: str = "auto"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.device = self._set_device(device)
        self.yolo_model = None
        self.cnn_model = None
        self.cnn_transform = None

    def _load_config(self) -> dict:
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _set_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_yolo(self, model_path: str):
        """Load YOLO model."""
        if self.yolo_model is not None:
            return
        try:
            self.yolo_model = YOLO(model_path)
            self.yolo_model.to(self.device)
            logger.info(f"YOLO model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            raise

    def load_cnn(self, model_path: str):
        """Load EfficientNet-B0 CNN model."""
        if self.cnn_model is not None:
            return
        try:
            state_dict = torch.load(model_path, map_location=self.device)

            # Create EfficientNet-B0 with 2 classes
            self.cnn_model = models.efficientnet_b0(weights=None)
            num_features = self.cnn_model.classifier[1].in_features
            self.cnn_model.classifier[1] = torch.nn.Linear(num_features, 2)

            # Load state dict
            self.cnn_model.load_state_dict(state_dict, strict=True)
            self.cnn_model.to(self.device)
            self.cnn_model.eval()

            # Create transform
            self.cnn_transform = Compose([
                Resize(size=(224, 224), interpolation=2, antialias=True),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
            ])

            logger.info(f"CNN model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load CNN: {e}")
            raise


class YOLODetector:
    """YOLO rubble detection."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.config = model_manager.config['yolo']

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run YOLO inference.
        Returns list of {x, y, w, h, confidence, class_name}
        """
        if self.model_manager.yolo_model is None:
            return []

        results = self.model_manager.yolo_model(image, conf=self.config['confidence_threshold'])
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                detections.append({
                    'x': float((x1 + x2) / 2),
                    'y': float((y1 + y2) / 2),
                    'w': float(x2 - x1),
                    'h': float(y2 - y1),
                    'confidence': conf,
                    'class_name': self.config['class_names'][0]
                })

        return detections


class CNNTiler:
    """CNN tile scanning for rubble classification."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.config = model_manager.config['cnn']
        self.tile_size = self.config['tile_size']
        self.stride = self.config['stride']
        self.threshold = self.config['rubble_probability_threshold']

    def scan(self, image: np.ndarray, batch_size: int = 32) -> List[Dict]:
        """
        Scan image with sliding window tiles.
        Returns list of {x, y, tile_size, rubble_probability, class_name}
        """
        if self.model_manager.cnn_model is None:
            return []

        if image.shape[0] < self.tile_size or image.shape[1] < self.tile_size:
            logger.warning(f"Image size {image.shape} < tile size {self.tile_size}")
            return []

        detections = []
        tiles_batch = []
        coords_batch = []

        # Collect all tiles
        for y in range(0, image.shape[0] - self.tile_size + 1, self.stride):
            for x in range(0, image.shape[1] - self.tile_size + 1, self.stride):
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                tiles_batch.append(tile)
                coords_batch.append((x, y))

        # Process in batches
        with torch.no_grad():
            for i in range(0, len(tiles_batch), batch_size):
                batch_tiles = tiles_batch[i:i+batch_size]
                batch_coords = coords_batch[i:i+batch_size]

                # Transform tiles
                batch_tensors = []
                for tile in batch_tiles:
                    from PIL import Image
                    pil_tile = Image.fromarray(tile)
                    transformed = self.model_manager.cnn_transform(pil_tile)
                    batch_tensors.append(transformed)

                batch_input = torch.stack(batch_tensors).to(self.model_manager.device)
                outputs = self.model_manager.cnn_model(batch_input)
                probs = F.softmax(outputs, dim=1)
                rubble_probs = probs[:, 1].cpu().numpy()

                # Filter by threshold
                for (x, y), prob in zip(batch_coords, rubble_probs):
                    if prob >= self.threshold:
                        detections.append({
                            'x': float(x),
                            'y': float(y),
                            'tile_size': self.tile_size,
                            'rubble_probability': float(prob),
                            'class_name': 'rubble'
                        })

        return detections


class AlertManager:
    """Combine YOLO and CNN detections into alerts."""

    def __init__(self):
        self.alerts = []

    def fuse(self, yolo_dets: List[Dict], cnn_dets: List[Dict],
             iou_threshold: float = 0.3) -> Dict:
        """
        Fuse YOLO and CNN detections.
        Groups into: both, cnn_only, yolo_only
        """

        # Apply NMS to CNN detections for visualization only
        cnn_nms = self._nms_cnn_tiles(cnn_dets, iou_threshold)

        # But keep raw CNN detections for decision logic
        cnn_raw = cnn_dets

        # Match YOLO boxes to CNN raw detections
        yolo_matched = set()
        cnn_matched = set()
        both_alerts = []

        for yolo_idx, yolo_box in enumerate(yolo_dets):
            for cnn_idx, cnn_tile in enumerate(cnn_raw):
                if self._iou(yolo_box, cnn_tile) > 0.1:
                    both_alerts.append({
                        'source': 'both',
                        'yolo': yolo_box,
                        'cnn': cnn_tile,
                        'x': (yolo_box['x'] + cnn_tile['x'] + cnn_tile['tile_size']/2) / 2,
                        'y': (yolo_box['y'] + cnn_tile['y'] + cnn_tile['tile_size']/2) / 2
                    })
                    yolo_matched.add(yolo_idx)
                    cnn_matched.add(cnn_idx)
                    break

        # Unmatched CNN detections (use NMS version for display)
        cnn_only_alerts = []
        for cnn_idx, cnn_tile in enumerate(cnn_nms):
            if cnn_idx not in cnn_matched:
                # Find raw CNN equivalent
                raw_tile = next((t for t in cnn_raw if t['x'] == cnn_tile['x'] and t['y'] == cnn_tile['y']), None)
                if raw_tile:
                    cnn_only_alerts.append({
                        'source': 'cnn_only',
                        'cnn': raw_tile,
                        'x': cnn_tile['x'] + cnn_tile['tile_size'] / 2,
                        'y': cnn_tile['y'] + cnn_tile['tile_size'] / 2
                    })

        # Unmatched YOLO detections
        yolo_only_alerts = []
        for yolo_idx, yolo_box in enumerate(yolo_dets):
            if yolo_idx not in yolo_matched:
                yolo_only_alerts.append({
                    'source': 'yolo_only',
                    'yolo': yolo_box,
                    'x': yolo_box['x'],
                    'y': yolo_box['y']
                })

        return {
            'both': both_alerts,
            'cnn_only': cnn_only_alerts,
            'yolo_only': yolo_only_alerts,
            'cnn_nms': cnn_nms,  # For visualization
            'cnn_raw': cnn_raw  # Keep raw for decision
        }

    def _iou(self, yolo_box: Dict, cnn_tile: Dict) -> float:
        """IoU between YOLO box and CNN tile."""
        yolo_x1 = yolo_box['x'] - yolo_box['w'] / 2
        yolo_y1 = yolo_box['y'] - yolo_box['h'] / 2
        yolo_x2 = yolo_box['x'] + yolo_box['w'] / 2
        yolo_y2 = yolo_box['y'] + yolo_box['h'] / 2

        cnn_x1 = cnn_tile['x']
        cnn_y1 = cnn_tile['y']
        cnn_x2 = cnn_tile['x'] + cnn_tile['tile_size']
        cnn_y2 = cnn_tile['y'] + cnn_tile['tile_size']

        inter_x1 = max(yolo_x1, cnn_x1)
        inter_y1 = max(yolo_y1, cnn_y1)
        inter_x2 = min(yolo_x2, cnn_x2)
        inter_y2 = min(yolo_y2, cnn_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        yolo_area = yolo_box['w'] * yolo_box['h']
        cnn_area = cnn_tile['tile_size'] ** 2
        union_area = yolo_area + cnn_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _nms_cnn_tiles(self, cnn_dets: List[Dict], iou_threshold: float) -> List[Dict]:
        """NMS for CNN detections (display only)."""
        if not cnn_dets:
            return []

        sorted_dets = sorted(cnn_dets, key=lambda x: x['rubble_probability'], reverse=True)
        keep = []

        for det in sorted_dets:
            keep_this = True
            for kept in keep:
                iou = self._tile_iou(det, kept)
                if iou > iou_threshold:
                    keep_this = False
                    break
            if keep_this:
                keep.append(det)

        return keep

    def _tile_iou(self, tile1: Dict, tile2: Dict) -> float:
        """IoU between two tiles."""
        t1_x1, t1_y1 = tile1['x'], tile1['y']
        t1_x2 = t1_x1 + tile1['tile_size']
        t1_y2 = t1_y1 + tile1['tile_size']

        t2_x1, t2_y1 = tile2['x'], tile2['y']
        t2_x2 = t2_x1 + tile2['tile_size']
        t2_y2 = t2_y1 + tile2['tile_size']

        inter_x1 = max(t1_x1, t2_x1)
        inter_y1 = max(t1_y1, t2_y1)
        inter_x2 = min(t1_x2, t2_x2)
        inter_y2 = min(t1_y2, t2_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = tile1['tile_size']**2 + tile2['tile_size']**2 - inter_area

        return inter_area / union_area if union_area > 0 else 0
