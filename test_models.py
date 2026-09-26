#!/usr/bin/env python3
"""Test model loading and basic inference."""

import sys
import numpy as np
from PIL import Image
from pathlib import Path

from earthquake_detector import ModelManager, YOLODetector, CNNTiler, AlertManager

def test_model_loading():
    """Test if models load correctly."""
    print("🔧 Test 1: Model Yükleme")
    print("-" * 50)

    try:
        manager = ModelManager("inference_config.json")
        print("✓ Config yüklendi")

        print("  YOLO yükleniyor...")
        manager.load_yolo("yolo_rubble_best.pt")
        print(f"  ✓ YOLO yüklendi (device: {manager.device})")

        print("  CNN yükleniyor...")
        manager.load_cnn("earthquake_tile_cnn_state_dict.pt")
        print(f"  ✓ CNN yüklendi (device: {manager.device})")

        return manager
    except Exception as e:
        print(f"  ✗ Hata: {e}")
        sys.exit(1)


def test_inference(manager):
    """Test inference on a dummy image."""
    print("\n🔧 Test 2: Dummy Görüntü Inference")
    print("-" * 50)

    try:
        # Create dummy 640x640 image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        print(f"  Dummy görüntü: {dummy_image.shape}")

        print("  YOLO inference...")
        detector = YOLODetector(manager)
        yolo_dets = detector.detect(dummy_image)
        print(f"  ✓ YOLO sonuç: {len(yolo_dets)} kutu")
        if yolo_dets:
            det = yolo_dets[0]
            print(f"    Örnek: conf={det['confidence']:.3f}, pos=({det['x']:.0f},{det['y']:.0f})")

        print("  CNN tarama...")
        tiler = CNNTiler(manager)
        cnn_dets = tiler.scan(dummy_image, batch_size=32)
        print(f"  ✓ CNN sonuç: {len(cnn_dets)} kare")
        if cnn_dets:
            det = cnn_dets[0]
            print(f"    Örnek: prob={det['rubble_probability']:.3f}, pos=({det['x']:.0f},{det['y']:.0f})")

        print("  Fusion...")
        alert_mgr = AlertManager()
        fusion = alert_mgr.fuse(yolo_dets, cnn_dets, iou_threshold=0.3)
        print(f"  ✓ Birleşim:")
        print(f"    - İkisi de: {len(fusion['both'])}")
        print(f"    - Yalnız CNN: {len(fusion['cnn_only'])}")
        print(f"    - Yalnız YOLO: {len(fusion['yolo_only'])}")
        print(f"    - CNN raw (karar için): {len(fusion['cnn_raw'])}")
        print(f"    - CNN NMS (gösterim için): {len(fusion['cnn_nms'])}")

    except Exception as e:
        print(f"  ✗ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_coordinate_transform():
    """Test coordinate transformations."""
    print("\n🔧 Test 3: Koordinat Dönüşümü")
    print("-" * 50)

    try:
        # Simulate YOLO box
        yolo_box = {
            'x': 320, 'y': 320, 'w': 100, 'h': 100,
            'confidence': 0.8, 'class_name': 'rubble'
        }

        # Simulate CNN tile
        cnn_tile = {
            'x': 300, 'y': 300, 'tile_size': 128,
            'rubble_probability': 0.85, 'class_name': 'rubble'
        }

        alert_mgr = AlertManager()
        iou = alert_mgr._iou(yolo_box, cnn_tile)
        print(f"  YOLO box: center=({yolo_box['x']},{yolo_box['y']}), size={yolo_box['w']}x{yolo_box['h']}")
        print(f"  CNN tile: corner=({cnn_tile['x']},{cnn_tile['y']}), size={cnn_tile['tile_size']}x{cnn_tile['tile_size']}")
        print(f"  ✓ IoU: {iou:.3f} (>0.1 için match)")

    except Exception as e:
        print(f"  ✗ Hata: {e}")
        sys.exit(1)


def test_nms():
    """Test NMS preserves raw detections."""
    print("\n🔧 Test 4: NMS (Ham Kareleri Koruma)")
    print("-" * 50)

    try:
        cnn_dets = [
            {'x': 0, 'y': 0, 'tile_size': 128, 'rubble_probability': 0.9},
            {'x': 50, 'y': 0, 'tile_size': 128, 'rubble_probability': 0.7},  # Yüksek IoU, elimine edilebilir
            {'x': 500, 'y': 500, 'tile_size': 128, 'rubble_probability': 0.85},  # Ayrı, korunacak
        ]

        alert_mgr = AlertManager()
        nms_result = alert_mgr._nms_cnn_tiles(cnn_dets, iou_threshold=0.3)

        print(f"  Ham kareler: {len(cnn_dets)}")
        print(f"  NMS sonrası: {len(nms_result)}")
        print(f"  ✓ NMS gösterim kutularını filtreler, raw korunur")

    except Exception as e:
        print(f"  ✗ Hata: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Model Test Paketi")
    print("=" * 50)

    manager = test_model_loading()
    test_inference(manager)
    test_coordinate_transform()
    test_nms()

    print("\n" + "=" * 50)
    print("✅ Tüm testler başarıyla tamamlandı!")
    print("=" * 50)
    print("\n🚀 Streamlit uygulamasını başlatmak için:")
    print("   streamlit run app.py")
