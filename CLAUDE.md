# Trade Advisor — Claude Rehberi

## Proje Özeti

Streamlit tabanlı candlestick + teknik indikatör analiz uygulaması.
GitHub: `GurselKona/Advisor` — `main` branch'e push = otomatik Streamlit Community Cloud deploy.

## Dosya Rolleri

| Dosya | Rol |
|---|---|
| `trade_advisor.py` | Streamlit UI, grafik, rapor |
| `candlestick_analyzer.py` | Kural tabanlı motor (EMA/RSI/WaveTrend + 17 formasyon) |
| `ml_analyzer.py` | ML motoru — `analyze(df)` aynı arayüz |
| `data_providers.py` | Veri kaynağı soyutlaması (Yahoo Finance + CSV) |
| `train_model.py` | Offline RandomForest eğitim scripti |
| `model.joblib` | Pre-trained model — git'e eklenmez (.gitignore'da) |

## Kritik Teknik Kararlar

### Plotly 6.x — Datetime
`df.index` (pandas DatetimeIndex, nanosaniye) doğrudan Plotly trace'e **geçirilmez** — grafik yanlış konumda render olur (2000 yılı gibi).
Her zaman dönüştür:
```python
x_vals = df.index.to_pydatetime().tolist()   # index için
y_vals = series.tolist()                       # Series için
```

### Rangebreak — Sadece Hafta Sonu
Saat bazlı rangebreak (`bounds=[16, 9.5], pattern="hour"`) **eklenmez** — tüm veri UTC'ye normalize edildiğinden yerel işlem saatleri (ET 9:30–16:00 = UTC 14:30–21:00) çoğu çubuğu gizler.
Sadece hafta sonu gizleme aktif:
```python
rangebreaks = []
if not has_weekend:
    rangebreaks.append(dict(bounds=["sat", "mon"]))
```

### 2h / 4h Interval
Yahoo Finance bu intervalları doğrudan desteklemez. `60m` veri çekilir, ardından resample edilir:
```python
_RESAMPLE = {"2h": "2h", "4h": "4h"}
fetch_iv = "60m" if interval in self._RESAMPLE else interval
```

### ML Model Önceliği
1. `model.joblib` varsa yükle (hızlı, kaliteli)
2. Yoksa mevcut veriyle anlık eğit — minimum 120 bar gerekli

## Streamlit — Yerel Geliştirme

Watchdog kurulu değil → dosya değişikliği otomatik algılanmaz.
Kod değiştirdikten sonra:
```bash
pkill -f "streamlit run trade_advisor"
streamlit run trade_advisor.py
```
Yeniden başlatma sonrası `session_state` sıfırlanır — kullanıcının "Analiz Et"e tekrar basması gerekir.

## Bekleyen İyileştirmeler

- **Piyasa tatil günleri:** Grafikteki küçük boşluklar (Noel, Şükran Günü vb.) hâlâ görünüyor.
  Plan: veri boşluklarından dinamik rangebreak hesabı — takvim kütüphanesi gerektirmez.

## Konvansiyonlar

- UI dili: Türkçe
- Kod yorumları: Türkçe (teknik terimler İngilizce)
- Tüm zaman damgaları UTC'ye normalize edilir, tz-naive saklanır
- Her analiz sonucu `trade_report_YYYYMMDD_HHMMSS.txt` olarak diske kaydedilir
