# 🏚️ Deprem Enkaz Görüntü Analiz — Karar Destek Prototipi

Bu prototip, deprem enkazı tespiti için **YOLO** (kutu tespit) ve **CNN** (sınıflandırma) modellerini birleştiren interaktif bir karar destek aracıdır.

## ⚠️ Önemli Not

**Bu sistem insan incelemesine yardımcı olmak için tasarlanmıştır. Otomatik saha sevki veya operasyonel karar kriter DEĞILDIR.**

## 🎯 Amacı

- Geniş alan hava/uydu görüntülerinde enkaz bölgelerini tespit
- YOLO ve CNN'nin tamamlayıcı güçlerini kullanarak yüksek kapsamlılık
- İnsan operatörlere görsel ve veri desteği sunma
- Bir model tarafından kaçırılan alanları diğerinin yakalaması

## 📋 Gereksinimler

- Python 3.9+
- CUDA (opsiyonel, GPU hızlandırması için)

## 🚀 Kurulum

### 1. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 2. Model Dosyalarını Kontrol Et

Klasörde şu dosyaların olması gerekir:
- `yolo_rubble_best.pt` — YOLO enkaz tespit modeli
- `earthquake_tile_cnn_state_dict.pt` — CNN sınıflandırıcı
- `inference_config.json` — Konfigürasyon (eşikler, ön işleme)

### 3. Uygulamayı Çalıştır

```bash
streamlit run app.py
```

Tarayıcı otomatik olarak açılacak: `http://localhost:8501`

## 📖 Kullanım

### Temel Adımlar

1. **Görüntü Yükle**: Hava/uydu görüntüsünü yükle (JPG, PNG, TIFF)
2. **Otomatik Analiz**: YOLO + CNN otomatik olarak çalışır
3. **Sonuçları Gözden Geçir**: 
   - Görsel harita: Mavi (YOLO), Turuncu (CNN), Yeşil çapraz (Her İkisi)
   - İnceleme kuyruğu: Uyarıları öncelik sırasına göre listele
4. **Dışa Aktar**: CSV/JSON olarak rapor indir

### Arayüz Kontrolleri

**Sidebar Ayarları:**
- **CNN Olasılık Eşiği** (0.5–1.0): CNN rubble kesinliği eşiği
- **NMS IoU Eşiği**: Gösterimde CNN kutularını basitleştirme (karar hesabına etki etmez)
- **CNN Batch Boyutu**: Bellek vs. hız dengeleme

## 🔍 Algılama Mantığı

### YOLO
- Tek sınıflı kutu tespit: "Collapsed building / rubble"
- Kutuları pixel koordinatlarında döndürür
- Güven eşiği: config.json'dan (varsayılan 0.25)

### CNN
- 128×128 piksellik kareler, 64 piksel adımla tarama
- EfficientNet-B0: 2 sınıf [0]="clear_candidate", [1]="rubble"
- Ön işleme: 224×224 resize + ImageNet normalizasyon
- Eşik: CNN olasılığı ≥ 0.80 (config.json)

### Fusion (Birleştirme)
1. **İkisi de**: YOLO kutusu + CNN kareleri çakışıyorsa
2. **Yalnız CNN (ÖNEMLİ)**: CNN yakaladı, YOLO kaçırdı
   - CNN'nin temel işlevi YOLO kaçırmalarını yakalamak
   - Sarı vurgulanmış, ilk dikkat edilecek
3. **Yalnız YOLO**: YOLO yakaladı, CNN doğrulamamadı

### Görselleştirme
- **Mavi kutular**: YOLO tespit
- **Turuncu/sarı kareler**: CNN tespit (NMS uygulanmış)
- **Yeşil çapraz**: İkisi de algıladı

**Not**: NMS (Non-Maximum Suppression, IoU 0.30) **yalnızca gösterim kutularına** uygulanır. Karar hesabında **ham CNN kareleri** kullanılır.

## 📊 Çıktılar

### CSV Export
Her uyarı için:
- Kaynak (İkisi de / Yalnız CNN / Yalnız YOLO)
- X, Y koordinatları
- YOLO skoru (varsa)
- CNN olasılığı (varsa)
- İnceleme durumu

### JSON Export
- Meta veriler (timestamp, görüntü adı)
- Özet (sayılar)
- Ham deteksiyonlar (YOLO, CNN)
- Fusion uyarıları

## ⚡ Performans Notu

### Test Sonuçları (44 pozitif görüntü, 211 etiketli enkaz)
- YOLO: 175/211 algıla (83%)
- CNN: 199/211 algıla (94%)
- Birleşim (NMS öncesi): 207/211 (98%)
- Birleşim (NMS sonrası): 206/211 (98%)

**Bu rakamlar test seti için elde edilmiştir. Yeni görüntülerdeki performans farklı olabilir.**

## 🛠️ Teknik Detaylar

### Model Yükleme
- `ModelManager`: YOLO ve CNN modelleri cache'e alır
- GPU otomatik kullanılır (varsa), yoksa CPU'ya geri döner
- State dict strict mode: Uyumsuzluk hata verir (tasarım gereği)

### CNN Scanning
- Batch işleme: Bellek taşmasını önlemek için (varsayılan batch_size=32)
- Görüntü < 128px: Uyarı verir, tarama yapmaz
- Transform: inference_config.json'dan tam uyum

### NMS
```
NMS (İoU 0.30) → Gösterim kutularını filtreler
Raw detections → Karar hesabında kullanılır (koruma)
```

## 🐛 Bilinen Kısıtlamalar

1. **Küçük görüntüler**: ≤128 px CNN taraması yapamaz
2. **Kenarlı kareler**: Görüntü kenarındaki eksik kareler işlenmez
3. **Performans garantisi yok**: Test sonuçları yeni veriye taşınmaz
4. **İnsan incelemesi gerekli**: Sistem tüm uyarıları doğrula

## 📞 Destek

Sorular ve hata raporları: `berkebaran00@gmail.com`

---

**© 2026 | Deprem Enkaz Analiz Sistemi | Karar Destek Prototipi**
