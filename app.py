import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
import io
import csv
from datetime import datetime
import logging

from earthquake_detector import ModelManager, YOLODetector, CNNTiler, AlertManager
from audit_logger import AuditLogger
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize audit logger
if 'audit_logger' not in st.session_state:
    st.session_state.audit_logger = AuditLogger()

# ============================================================================
# HELPER FUNCTIONS (defined first for Streamlit)
# ============================================================================

def visualize_detections(image: np.ndarray, yolo_dets: list, cnn_nms: list, fusion: dict) -> np.ndarray:
    """Draw YOLO boxes and CNN tiles on image."""
    img_copy = Image.fromarray(image).copy()
    draw = ImageDraw.Draw(img_copy, 'RGBA')

    # Draw YOLO boxes (blue)
    for det in yolo_dets:
        x, y, w, h = det['x'], det['y'], det['w'], det['h']
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 100, 255, 200), width=3)
        draw.text((x1, y1-20), f"YOLO {det['confidence']:.2f}", fill=(0, 100, 255, 255))

    # Draw CNN tiles NMS (orange/yellow)
    for tile in cnn_nms:
        x, y, size = tile['x'], tile['y'], tile['tile_size']
        draw.rectangle([int(x), int(y), int(x+size), int(y+size)],
                      outline=(255, 165, 0, 200), width=2)
        draw.text((int(x), int(y)-15), f"CNN {tile['rubble_probability']:.2f}",
                 fill=(255, 165, 0, 255))

    # Draw "both" alerts (green cross)
    for alert in fusion['both']:
        x, y = int(alert['x']), int(alert['y'])
        size = 10
        draw.line([x-size, y, x+size, y], fill=(0, 255, 0, 255), width=3)
        draw.line([x, y-size, x, y+size], fill=(0, 255, 0, 255), width=3)

    return np.array(img_copy)


def build_alerts_table(fusion: dict, yolo_dets: list) -> pd.DataFrame:
    """Build alerts dataframe with severity and recommended actions."""
    rows = []

    # Both detections (HIGH severity)
    for alert in fusion['both']:
        yolo_conf = alert['yolo']['confidence']
        cnn_prob = alert['cnn']['rubble_probability']
        avg_confidence = (yolo_conf + cnn_prob) / 2

        if avg_confidence >= 0.85:
            severity = '🔴 CİDDİ'
            action = 'Hasarlı operasyon desteği gerekli; hızlı evakuasyon önerilir'
        elif avg_confidence >= 0.75:
            severity = '🟠 ORTA'
            action = 'Detaylı araştırma gerekli; operasyon hazırlığı başlat'
        else:
            severity = '🟡 DÜŞÜK'
            action = 'Doğrulama gerekli; rutin kontrol'

        rows.append({
            'Ciddiyet': severity,
            'Kaynak': 'İkisi de ✓✓',
            'X': int(alert['yolo']['x']),
            'Y': int(alert['yolo']['y']),
            'YOLO': f"{yolo_conf:.3f}",
            'CNN': f"{cnn_prob:.3f}",
            'Ort.': f"{avg_confidence:.3f}",
            'Önerilen Aksiyon': action
        })

    # CNN only (emphasized - HIGH priority)
    for alert in fusion['cnn_only']:
        cnn_prob = alert['cnn']['rubble_probability']

        if cnn_prob >= 0.90:
            severity = '🔴 CİDDİ'
            action = 'CNN yakalama (YOLO kaçırdı): Hızlı doğrulama + operasyon hazırlığı'
        else:
            severity = '🟠 ORTA'
            action = 'CNN tespit (YOLO kaçırdı): Uzman doğrulaması gerekli'

        rows.append({
            'Ciddiyet': severity,
            'Kaynak': 'Yalnız CNN ⚠️',
            'X': int(alert['cnn']['x']),
            'Y': int(alert['cnn']['y']),
            'YOLO': '—',
            'CNN': f"{cnn_prob:.3f}",
            'Ort.': f"{cnn_prob:.3f}",
            'Önerilen Aksiyon': action
        })

    # YOLO only
    for alert in fusion['yolo_only']:
        yolo_conf = alert['yolo']['confidence']

        if yolo_conf >= 0.75:
            severity = '🟠 ORTA'
            action = 'YOLO tespit (CNN desteklemedi): Doğrulama sonrası karar ver'
        else:
            severity = '🟡 DÜŞÜK'
            action = 'YOLO taraması: Detay incelemesi önerilir'

        rows.append({
            'Ciddiyet': severity,
            'Kaynak': 'Yalnız YOLO',
            'X': int(alert['yolo']['x']),
            'Y': int(alert['yolo']['y']),
            'YOLO': f"{yolo_conf:.3f}",
            'CNN': '—',
            'Ort.': f"{yolo_conf:.3f}",
            'Önerilen Aksiyon': action
        })

    return pd.DataFrame(rows)


def generate_csv(alerts_df, image_name: str) -> bytes:
    """Generate CSV export."""
    output = io.StringIO()
    alerts_df.to_csv(output, index=False)
    csv_bytes = output.getvalue().encode('utf-8-sig')
    return csv_bytes


def generate_json(fusion: dict, yolo_dets: list, image_name: str) -> str:
    """Generate JSON export."""
    export_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'image_name': image_name,
            'note': 'Prototip - insan incelemesi gerektirir'
        },
        'summary': {
            'yolo_detections': len(yolo_dets),
            'cnn_raw_tiles': len(fusion['cnn_raw']),
            'cnn_nms_tiles': len(fusion['cnn_nms']),
            'alerts_both': len(fusion['both']),
            'alerts_cnn_only': len(fusion['cnn_only']),
            'alerts_yolo_only': len(fusion['yolo_only'])
        },
        'detections': {
            'yolo': yolo_dets,
            'cnn_raw': fusion['cnn_raw'],
            'cnn_nms': fusion['cnn_nms']
        },
        'alerts': {
            'both': fusion['both'],
            'cnn_only': fusion['cnn_only'],
            'yolo_only': fusion['yolo_only']
        }
    }

    return json.dumps(export_data, indent=2, default=str)


# ============================================================================
# PAGE CONFIG & INITIALIZATION
# ============================================================================

st.set_page_config(
    page_title="Deprem Enkaz Karar Destek",
    page_icon="🏚️",
    layout="wide"
)

st.title("🏚️ Deprem Enkaz Görüntü Analiz - Karar Destek Prototipi")
st.markdown("**Not:** Bu prototip insan incelemesine yardımcı olmak için tasarlanmıştır. Otomatik saha sevki kararı değildir.")

# Initialize session state
if 'model_manager' not in st.session_state:
    try:
        config_path = Path("inference_config.json")
        st.session_state.model_manager = ModelManager(str(config_path))
        st.session_state.model_manager.load_yolo("yolo_rubble_best.pt")
        st.session_state.model_manager.load_cnn("earthquake_tile_cnn_state_dict.pt")
        st.session_state.models_loaded = True
    except Exception as e:
        st.error(f"❌ Model yükleme hatası: {e}")
        st.session_state.models_loaded = False
        st.stop()

# Sidebar controls
st.sidebar.header("⚙️ Ayarlar")
confidence_thresh = st.sidebar.slider("CNN Olasılık Eşiği", 0.5, 1.0, 0.8, 0.05)
nms_iou = st.sidebar.slider("NMS IoU Eşiği (Gösterim)", 0.1, 0.5, 0.3, 0.05)
batch_size = st.sidebar.slider("CNN Batch Boyutu", 16, 64, 32)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Main interface
uploaded_file = st.file_uploader("Geniş alan hava/uydu görüntüsü yükle",
                                  type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.subheader("📸 Yüklenen Görüntü")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(image, use_container_width=True)
    with col2:
        st.metric("Görüntü Boyutu", f"{image_array.shape[1]}×{image_array.shape[0]}")

    # Run detection
    st.session_state.audit_logger.log_analysis_start(uploaded_file.name, image_array.shape)

    st.subheader("🔍 Analiz Yapılıyor...")
    progress = st.progress(0)

    with st.spinner("YOLO enkaz tespit ediliyor..."):
        t_start = time.time()
        detector = YOLODetector(st.session_state.model_manager)
        yolo_dets = detector.detect(image_array)
        st.session_state.audit_logger.log_yolo_inference(
            uploaded_file.name, len(yolo_dets), time.time() - t_start,
            st.session_state.model_manager.config['yolo']['confidence_threshold']
        )
        progress.progress(33)

    with st.spinner("CNN taraması yapılıyor..."):
        t_start = time.time()
        tiler = CNNTiler(st.session_state.model_manager)
        cnn_dets = tiler.scan(image_array, batch_size=batch_size)
        st.session_state.audit_logger.log_cnn_scanning(
            uploaded_file.name,
            (image_array.shape[0] // 64) * (image_array.shape[1] // 64),
            len(cnn_dets), time.time() - t_start,
            st.session_state.model_manager.config['cnn']['rubble_probability_threshold']
        )
        progress.progress(66)

    with st.spinner("Uyarılar birleştiriliyor..."):
        alert_manager = AlertManager()
        fusion = alert_manager.fuse(yolo_dets, cnn_dets, iou_threshold=nms_iou)
        progress.progress(100)
        st.session_state.audit_logger.log_fusion(
            uploaded_file.name, len(fusion['both']), len(fusion['cnn_only']),
            len(fusion['yolo_only']), nms_iou
        )

    # Summary statistics
    st.subheader("📊 Analiz Özeti")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("YOLO Tespit", len(yolo_dets))
    with col2:
        st.metric("CNN Tarama", len(fusion['cnn_raw']))
    with col3:
        st.metric("Toplam Uyarı", len(fusion['both']) + len(fusion['cnn_only']) + len(fusion['yolo_only']))
    with col4:
        coverage = (len(fusion['cnn_raw']) / max(1, (image_array.shape[0] // 64) * (image_array.shape[1] // 64))) * 100
        st.metric("Kapsamı", f"{min(100, int(coverage))}%")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 İkisi de", len(fusion['both']))
    with col2:
        st.metric("🟠 Yalnız CNN", len(fusion['cnn_only']))
    with col3:
        st.metric("🟡 Yalnız YOLO", len(fusion['yolo_only']))
    
    st.subheader("⚠️ Risk Değerlendirmesi")
    total_alerts = len(fusion['both']) + len(fusion['cnn_only']) + len(fusion['yolo_only'])
    
    if total_alerts == 0:
        st.success("✅ **BÖLGE GÜVENLİ** — Enkaz tespit edilmedi")
        st.caption("(%98 başarı oranı; uzman doğrulaması önerilir)")
    else:
        if len(fusion['both']) > 0:
            st.error(f"🔴 **CİDDİ RİSK** — {len(fusion['both'])} alan her iki model tarafından tespit")
            st.markdown("→ Aksiyon: **Hasarlı operasyon desteği gerekli; hızlı evakuasyon**")
        if len(fusion['cnn_only']) > 0:
            st.warning(f"🟠 **ORTA RİSK** — {len(fusion['cnn_only'])} alan CNN tespit (YOLO kaçırdı)")
            st.markdown("→ Aksiyon: **Detaylı araştırma; operasyon hazırlığı**")
        if len(fusion['yolo_only']) > 0:
            st.info(f"🟡 **DÜŞÜK RİSK** — {len(fusion['yolo_only'])} alan sadece YOLO tespit")
            st.markdown("→ Aksiyon: **Doğrulama sonrası karar ver**")

    st.subheader("🎨 Görselleştirme")
    fig_image = visualize_detections(image_array, yolo_dets, fusion['cnn_nms'], fusion)
    st.image(fig_image, use_container_width=True)

    # Alerts table
    st.subheader("⚠️ İnceleme Kuyruğu")

    alerts_df = build_alerts_table(fusion, yolo_dets)

    if not alerts_df.empty:
        # Color coding
        def highlight_row(row):
            if row['Kaynak'] == 'İkisi de':
                return ['background-color: #e8f4f8'] * len(row)
            elif row['Kaynak'] == 'Yalnız CNN (ÖNEMLİ)':
                return ['background-color: #fff3cd'] * len(row)
            elif row['Kaynak'] == 'Yalnız YOLO':
                return ['background-color: #f0f0f0'] * len(row)
            return [''] * len(row)

        styled_df = alerts_df.style.apply(highlight_row, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
    else:
        st.success("✅ Enkaz tespit edilmedi. Görüntü bölgesi güvenli olarak değerlendirilmiştir.")
        st.caption("(Not: Bu, insan uzmanı tarafından doğrulanması gereken bir yardımcı tavsiyedir.)")

    # Export
    st.subheader("💾 Dışa Aktar")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 CSV İndir"):
            csv_bytes = generate_csv(alerts_df, uploaded_file.name)
            st.download_button(
                label="CSV Dosyasını İndir",
                data=csv_bytes,
                file_name=f"enkaz_analiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("📥 JSON İndir"):
            json_str = generate_json(fusion, yolo_dets, uploaded_file.name)
            st.download_button(
                label="JSON Dosyasını İndir",
                data=json_str,
                file_name=f"enkaz_analiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )



# ============================================================================
# AUDIT TRAIL VIEWER
# ============================================================================

with st.expander("📋 Denetim İzi (Audit Trail) — Tüm İşlemler", expanded=False):
    st.subheader("Analiz Geçmişi")
    
    if st.session_state.audit_logger.entries:
        summary = st.session_state.audit_logger.get_session_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Toplam Event", summary.get('total_events', 0))
        with col2:
            st.metric("Toplam Uyarı", summary.get('num_alerts', 0))
        with col3:
            st.metric("Hata Sayısı", summary.get('num_errors', 0))
        with col4:
            st.metric("Oturum Kimliği", summary.get('session_id', 'N/A')[:12])
        
        st.markdown("**Event Dağılımı:**")
        event_breakdown = summary.get('event_breakdown', {})
        for event_type, count in sorted(event_breakdown.items()):
            st.caption(f"  • {event_type}: {count}")
        
        st.markdown("---")
        st.markdown("**İşlem Detayları:**")
        
        # Filter options
        col1, col2 = st.columns([3, 1])
        with col1:
            filter_type = st.selectbox(
                "Event Türü ile Filtrele",
                ["Tümü"] + sorted(set([e['event_type'] for e in st.session_state.audit_logger.entries]))
            )
        with col2:
            show_limit = st.number_input("Göster (son N)", 1, 100, 20)
        
        # Display audit log
        audit_entries = st.session_state.audit_logger.get_audit_history(show_limit)
        
        if filter_type != "Tümü":
            audit_entries = [e for e in audit_entries if e['event_type'] == filter_type]
        
        for entry in reversed(audit_entries):
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 2, 4])

                with col1:
                    if entry['level'] == 'ERROR':
                        st.error(entry['event_type'])
                    else:
                        st.info(entry['event_type'])

                with col2:
                    st.caption(entry['timestamp'])

                with col3:
                    st.caption(entry['details'].get('description', ''))

                st.json(entry['details'], expanded=False)
        
        st.markdown("---")
        st.markdown("**Oturum Kaydet**")
        
        if st.button("💾 Audit Log'u İndir (JSON)"):
            log_file = st.session_state.audit_logger.save_session()
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            st.download_button(
                label="Audit Log Dosyasını İndir",
                data=log_content,
                file_name=f"audit_log_{st.session_state.audit_logger.session_id}.json",
                mime="application/json"
            )
    else:
        st.info("Henüz herhangi bir işlem kaydı yok.")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Deprem Enkaz Analiz Sistemi")
