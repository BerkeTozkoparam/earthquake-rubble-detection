"""Audit trail logging for earthquake rubble detection system."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class AuditLogger:
    """Log all system operations for audit trail."""

    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Current session log
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_dir / f"session_{self.session_id}.json"
        self.entries = []

        # Master log (all operations)
        self.master_log_file = self.log_dir / "master_audit.jsonl"

        logger.info(f"Audit logger initialized: {self.log_dir}")

    def log_event(self, event_type: str, details: Dict[str, Any],
                  level: str = "INFO") -> None:
        """Log an event to audit trail."""

        entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'event_type': event_type,
            'level': level,
            'details': details
        }

        # Add to session memory
        self.entries.append(entry)

        # Write to master log (append)
        with open(self.master_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"Audit: {event_type} - {details.get('description', '')}")

    def log_analysis_start(self, image_name: str, image_size: tuple) -> None:
        """Log start of image analysis."""
        self.log_event('ANALYSIS_START', {
            'description': 'Image analysis started',
            'image_name': image_name,
            'image_size': f"{image_size[1]}×{image_size[0]}",
            'timestamp_iso': datetime.now().isoformat()
        })

    def log_model_loading(self, model_name: str, model_type: str,
                         device: str, success: bool, duration: float = 0.0) -> None:
        """Log model loading."""
        self.log_event('MODEL_LOAD', {
            'description': f'{model_name} loaded',
            'model_name': model_name,
            'model_type': model_type,
            'device': device,
            'success': success,
            'duration_ms': round(duration * 1000, 2)
        })

    def log_yolo_inference(self, image_name: str, num_detections: int,
                          duration: float, confidence_threshold: float) -> None:
        """Log YOLO inference."""
        self.log_event('YOLO_INFERENCE', {
            'description': f'YOLO detected {num_detections} boxes',
            'image_name': image_name,
            'detections': num_detections,
            'confidence_threshold': confidence_threshold,
            'duration_ms': round(duration * 1000, 2)
        })

    def log_cnn_scanning(self, image_name: str, num_tiles: int,
                        num_positive_tiles: int, duration: float,
                        probability_threshold: float) -> None:
        """Log CNN tile scanning."""
        self.log_event('CNN_SCANNING', {
            'description': f'CNN scanned {num_tiles} tiles, found {num_positive_tiles}',
            'image_name': image_name,
            'total_tiles': num_tiles,
            'positive_tiles': num_positive_tiles,
            'probability_threshold': probability_threshold,
            'duration_ms': round(duration * 1000, 2)
        })

    def log_fusion(self, image_name: str, num_both: int, num_cnn_only: int,
                  num_yolo_only: int, nms_iou: float) -> None:
        """Log fusion results."""
        self.log_event('FUSION_COMPLETE', {
            'description': f'Fusion complete: {num_both} both, {num_cnn_only} CNN, {num_yolo_only} YOLO',
            'image_name': image_name,
            'alerts_both': num_both,
            'alerts_cnn_only': num_cnn_only,
            'alerts_yolo_only': num_yolo_only,
            'nms_iou_threshold': nms_iou,
            'total_alerts': num_both + num_cnn_only + num_yolo_only
        })

    def log_alert(self, alert_id: str, category: str, coordinates: tuple,
                 severity: str, yolo_score: Optional[float] = None,
                 cnn_score: Optional[float] = None, recommended_action: str = '') -> None:
        """Log individual alert."""
        self.log_event('ALERT_GENERATED', {
            'description': f'Alert: {severity} - {category}',
            'alert_id': alert_id,
            'category': category,
            'coordinates': {'x': int(coordinates[0]), 'y': int(coordinates[1])},
            'severity': severity,
            'yolo_score': round(yolo_score, 4) if yolo_score else None,
            'cnn_score': round(cnn_score, 4) if cnn_score else None,
            'recommended_action': recommended_action
        })

    def log_export(self, export_format: str, num_records: int, filename: str) -> None:
        """Log data export."""
        self.log_event('DATA_EXPORT', {
            'description': f'Exported {num_records} records as {export_format}',
            'format': export_format,
            'num_records': num_records,
            'filename': filename
        })

    def log_risk_assessment(self, risk_level: str, num_high: int, num_medium: int,
                           num_low: int, recommendation: str) -> None:
        """Log risk assessment."""
        self.log_event('RISK_ASSESSMENT', {
            'description': f'Risk level: {risk_level}',
            'risk_level': risk_level,
            'high_severity_alerts': num_high,
            'medium_severity_alerts': num_medium,
            'low_severity_alerts': num_low,
            'recommendation': recommendation
        })

    def log_error(self, error_type: str, error_message: str,
                 context: Dict[str, Any] = None) -> None:
        """Log error."""
        self.log_event('ERROR', {
            'description': f'{error_type}: {error_message}',
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {}
        }, level='ERROR')

    def save_session(self) -> str:
        """Save session log to file."""
        with open(self.session_log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'session_id': self.session_id,
                'start_time': self.entries[0]['timestamp'] if self.entries else None,
                'end_time': datetime.now().isoformat(),
                'num_events': len(self.entries),
                'events': self.entries
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Session saved: {self.session_log_file}")
        return str(self.session_log_file)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        if not self.entries:
            return {}

        event_types = {}
        alerts = []
        errors = []

        for entry in self.entries:
            event_type = entry['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1

            if event_type == 'ALERT_GENERATED':
                alerts.append(entry)
            elif entry['level'] == 'ERROR':
                errors.append(entry)

        return {
            'session_id': self.session_id,
            'total_events': len(self.entries),
            'event_breakdown': event_types,
            'num_alerts': len(alerts),
            'num_errors': len(errors),
            'start_time': self.entries[0]['timestamp'] if self.entries else None,
            'end_time': datetime.now().isoformat()
        }

    def get_audit_history(self, limit: int = 100) -> list:
        """Get recent audit entries."""
        return self.entries[-limit:]

    def read_master_log(self, limit: int = 100) -> list:
        """Read master audit log."""
        if not self.master_log_file.exists():
            return []

        entries = []
        with open(self.master_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass

        return entries[-limit:]
