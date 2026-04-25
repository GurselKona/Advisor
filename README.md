# Trade Advisor

Otomatik candlestick ve teknik indikatör analizi yapan Streamlit tabanlı web uygulaması. Aynı anda en fazla 5 ticker analiz edebilir, kural tabanlı veya makine öğrenmesi motoru ile sinyal üretir.

**Canlı uygulama:** Streamlit Community Cloud — `git push` sonrası otomatik deploy olur.

---

## İçindekiler

- [Özellikler](#özellikler)
- [Mimari](#mimari)
- [Kurulum](#kurulum)
- [Uygulamayı Çalıştırma](#uygulamayı-çalıştırma)
- [Parametreler ve Seçenekler](#parametreler-ve-seçenekler)
- [Analiz Motorları](#analiz-motorları)
- [Veri Kaynakları](#veri-kaynakları)
- [Teknik İndikatörler](#teknik-i̇ndikatörler)
- [Mum Formasyonları](#mum-formasyonları)
- [ML Modeli Eğitme](#ml-modeli-eğitme)
- [Bağımlılıklar](#bağımlılıklar)
- [Proje Yapısı](#proje-yapısı)

---

## Özellikler

- **Çoklu ticker:** Tek analizde 1–5 sembol karşılaştırması
- **Ticker arama:** Yahoo Finance'ın fuzzy search API'si ile otomatik tamamlama
- **İki analiz motoru:** Kural tabanlı (EMA/RSI/WaveTrend + formasyonlar) ve Makine Öğrenmesi (RandomForest)
- **İki veri kaynağı:** Yahoo Finance (gerçek zamanlı) ve yerel CSV dosyası
- **11 interval:** 1 dakikadan 1 aya kadar; her interval için Yahoo Finance limitlerine göre dinamik dönem presetleri
- **İnteraktif grafik:** Plotly candlestick + EMA çizgileri + hacim + crosshair; hafta sonu boşluğu gizleme
- **Giriş/Stop/TP çizgileri:** Sinyal BUY/SELL ise grafik üzerinde yatay çizgi gösterimi
- **Özet tablo:** Tüm tickerlar için sinyal, güç, güven, RSI, EMA hizalama tek görünümde
- **Rapor:** Her analizde `trade_report_YYYYMMDD_HHMMSS.txt` otomatik kaydedilir; `.txt` indirme butonu
- **Pre-trained ML modeli:** `model.joblib` varsa yüklenir — anlık eğitime gerek kalmaz

---

## Mimari

```
trade_advisor.py        ← Streamlit UI + grafik + rapor
│
├── candlestick_analyzer.py   ← Kural tabanlı motor
│     EMA, RSI, WaveTrend hesabı
│     17 mum formasyonu algılayıcı
│     Skor tabanlı sinyal sentezi
│
├── ml_analyzer.py            ← ML motoru (aynı arayüz)
│     20 özellik üretimi
│     Pre-trained model önce denenir
│     Yoksa mevcut veriyle fallback eğitim
│
├── data_providers.py         ← Veri kaynağı soyutlama
│     YahooFinanceProvider
│     CsvFileProvider
│     REGISTRY — UI'ya açılan kayıt defteri
│
└── train_model.py            ← Offline ML eğitim scripti
      32 ticker, 5 yıl günlük veri
      500 ağaçlı RandomForest → model.joblib
```

Her iki motor da aynı `analyze(df) → TradeRecommendation` arayüzünü uygular; UI motoru bilmez.

---

## Kurulum

```bash
# Repo klonla
git clone https://github.com/GurselKona/Advisor.git
cd Advisor

# Bağımlılıkları kur
pip install -r requirements.txt
```

> **Not:** Streamlit'in otomatik dosya yenileme özelliği (Watchdog) için
> `pip install watchdog` gerekir. Kurulu değilse kod değişikliklerinden sonra
> Streamlit'i manuel yeniden başlatın.

---

## Uygulamayı Çalıştırma

```bash
streamlit run trade_advisor.py
```

Tarayıcıda `http://localhost:8501` açılır.

Eski örnekleri durdurmak için:

```bash
pkill -f "streamlit run trade_advisor"
```

---

## Parametreler ve Seçenekler

### Veri Kaynağı

| Seçenek | Açıklama |
|---|---|
| **Yahoo Finance** | Hisse, ETF, kripto, döviz, endeks — canlı veri |
| **CSV Dosyası** | Yerel CSV — `date/datetime`, `Open`, `High`, `Low`, `Close`[, `Volume`] sütunları |

### Analiz Motoru

| Seçenek | Açıklama |
|---|---|
| **Kural Tabanlı** | EMA/RSI/WaveTrend + 17 mum formasyonu; skor tabanlı sentez |
| **Makine Öğrenmesi** | RandomForest; minimum 120 bar gerekli |

### Interval (Granülerite)

| Kod | Etiket | Yahoo Finance Limiti |
|---|---|---|
| `1m` | 1 Dakika | Son 7 gün |
| `2m` | 2 Dakika | Son 60 gün |
| `5m` | 5 Dakika | Son 60 gün |
| `15m` | 15 Dakika | Son 60 gün |
| `30m` | 30 Dakika | Son 60 gün |
| `60m` | 1 Saat | Son 730 gün |
| `2h` | 2 Saat | Son 730 gün (60m'den resample) |
| `4h` | 4 Saat | Son 730 gün (60m'den resample) |
| `1d` | 1 Gün | Limitsiz |
| `1wk` | 1 Hafta | Limitsiz |
| `1mo` | 1 Ay | Limitsiz |

### Dönem Presetleri

1 gün / 3 gün / 7 gün / 1 ay / 3 ay / 6 ay / 1 yıl / 2 yıl / 3 yıl / 4 yıl / 5 yıl — seçili interval'in Yahoo Finance limitini aşan presetler otomatik gizlenir.

**Özel tarih** seçeneği ile başlangıç/bitiş tarihleri manuel girilebilir.

---

## Analiz Motorları

### Kural Tabanlı (`candlestick_analyzer.py`)

İki bileşenin net skorunu toplar:

**1. Mum Formasyon Skoru**
Son 5 bar'da tespit edilen formasyonlara güce göre puan atanır:
- STRONG formasyon: ±3 puan
- MODERATE formasyon: ±2 puan
- WEAK formasyon: ±1 puan

**2. İndikatör Skoru**
EMA hizalama, RSI seviyesi, WaveTrend bölgesi ve kesişimleri +/− puan verir (detay: [Teknik İndikatörler](#teknik-i̇ndikatörler)).

**Sinyal kararı:**
- `toplam_net > 0` → BUY
- `toplam_net < 0` → SELL
- `toplam_net == 0` → NEUTRAL

**Güç:**
- net ≥ 8 → STRONG
- net ≥ 4 → MODERATE
- net < 4 → WEAK

**Stop Loss & Take Profit:** ATR(14) × 1.5 ve risk/ödül = 2.0

### Makine Öğrenmesi (`ml_analyzer.py`)

**Özellikler (20 adet):**
- EMA20/50/100/200 mesafeleri (kapanışa göre yüzde)
- EMA çiftleri arası fark (20/50, 50/100, 100/200)
- RSI(14), WaveTrend WT1/WT2 ve farkları
- Mum body oranı, üst/alt fitil oranları
- Getiri: 1/3/5 bar geriye
- Volatilite: 5/10 bar rolling std
- ATR oranı (14 bar)

**Etiketleme:** `forward_n=5` bar ilerideki kapanışa göre:
- `+%0.5`'ten fazla → BUY
- `−%0.5`'ten fazla → SELL
- Arada → NEUTRAL

**Model önceliği:**
1. `model.joblib` varsa yükle (hız + kalite)
2. Yoksa mevcut veriyle anlık eğit (min 120 bar gerekli)

**Güven eşikleri:**
- ≥ 0.65 → STRONG
- ≥ 0.50 → MODERATE
- < 0.50 → WEAK

---

## Veri Kaynakları

### Yahoo Finance

```
Ticker formatları:
  Hisse          : AAPL, TSLA, MSFT
  ETF            : SPY, QQQ
  Kripto         : BTC-USD, ETH-USD
  Döviz          : EURUSD=X, GBPUSD=X
  Endeks         : ^GSPC (S&P 500), ^IXIC (NASDAQ)
```

- Arama kutusu fuzzy search destekler (sembol veya şirket adı)
- 2h/4h intervallar 60m veriden resample edilir
- Tüm zaman dilimleri UTC'ye normalize edilir

### CSV Dosyası

Uygulama dizinindeki `.csv` dosyaları otomatik listelenir.

**Gerekli sütunlar:**
```
date (veya datetime/timestamp/time), Open, High, Low, Close
```

**İsteğe bağlı:** `Volume`

Sütun adları büyük/küçük harf duyarsız kabul edilir.

---

## Teknik İndikatörler

### EMA (Exponential Moving Average)

Periyotlar: 20, 50, 100, 200

**Hizalama sınıflandırması:**
| Durum | Değer |
|---|---|
| Kapanış > EMA20 > EMA50 > EMA100 > EMA200 | BULL_FULL (+3) |
| Kapanış ≥ 3 EMA'nın üzerinde | BULL_PARTIAL (+1) |
| Kapanış < EMA20 < EMA50 < EMA100 < EMA200 | BEAR_FULL (−3) |
| Kapanış ≤ 1 EMA'nın üzerinde | BEAR_PARTIAL (−1) |
| Diğer | MIXED (0) |

### RSI (Relative Strength Index, periyot: 14)

| Bölge | Değer | Puan |
|---|---|---|
| ≤ 20 | Aşırı satım | +3 |
| ≤ 30 | Aşırı satım | +2 |
| ≤ 40 | Satım baskısı | +1 |
| 40–60 | Nötr | 0 |
| ≥ 60 | Alım baskısı | −1 |
| ≥ 70 | Aşırı alım | −2 |
| ≥ 80 | Aşırı alım | −3 |

### WaveTrend Osilatör (n1=10, n2=21)

WT1 bölgesi (±53 eşik), WT2 ile kesişim son 1 barda kontrol edilir.

| Durum | Puan |
|---|---|
| WT1 ≤ −60 | +3 |
| WT1 ≤ −53 | +2 |
| WT1 ≥ 53 | −2 |
| WT1 ≥ 60 | −3 |
| Bull cross (< −40 bölgesinde) | +3 |
| Bull cross (nötr bölgede) | +1 |
| Bear cross (> +40 bölgesinde) | −3 |
| Bear cross (nötr bölgede) | −1 |

---

## Mum Formasyonları

Son 5 bar taranır. Tespit edilen 17 formasyon:

### Tek Mum

| Formasyon | Sinyal | Güç |
|---|---|---|
| Doji | NEUTRAL | WEAK |
| Hammer | BUY | MODERATE |
| Inverted Hammer | BUY | WEAK |
| Shooting Star | SELL | MODERATE |
| Bullish Marubozu | BUY | STRONG |
| Bearish Marubozu | SELL | STRONG |
| Spinning Top | NEUTRAL | WEAK |

### İki Mum

| Formasyon | Sinyal | Güç |
|---|---|---|
| Bullish Engulfing | BUY | STRONG |
| Bearish Engulfing | SELL | STRONG |
| Tweezer Bottom | BUY | MODERATE |
| Tweezer Top | SELL | MODERATE |
| Piercing Line | BUY | MODERATE |
| Dark Cloud Cover | SELL | MODERATE |

### Üç Mum

| Formasyon | Sinyal | Güç |
|---|---|---|
| Morning Star | BUY | STRONG |
| Evening Star | SELL | STRONG |
| Three White Soldiers | BUY | STRONG |
| Three Black Crows | SELL | STRONG |

---

## ML Modeli Eğitme

Pre-trained model yoksa uygulama mevcut veriyle anlık eğitim yapar (yavaş, düşük kalite). Kalıcı model oluşturmak için:

```bash
python3 train_model.py
```

**Eğitim kapsamı:**
- 32 ticker: ABD büyük cap, ETF, volatil teknoloji, kripto
- Her ticker için 5 yıl günlük veri (Yahoo Finance)
- 500 ağaçlı RandomForest — %80 eğitim / %20 test
- Çıktı: `model.joblib` + `model_meta.json`

**Süre:** ~3–10 dakika (internet hızına bağlı)

Model bir kez eğitildikten sonra tüm analizlerde tekrar yüklenir.

---

## Bağımlılıklar

```
yfinance==1.3.0          # Yahoo Finance veri çekme + arama
pandas==2.3.3            # DataFrame işlemleri
numpy==2.3.5             # Sayısal hesaplamalar
streamlit==1.52.1        # Web UI
streamlit-searchbox==0.1.24  # Ticker arama kutusu
plotly==6.7.0            # İnteraktif grafik
scikit-learn==1.8.0      # RandomForestClassifier
joblib==1.5.3            # Model kaydetme/yükleme
tqdm==4.67.1             # Eğitim ilerleme çubuğu
```

Kurulum: `pip install -r requirements.txt`

---

## Proje Yapısı

```
Advisor/
├── trade_advisor.py          # Ana uygulama — Streamlit UI
├── candlestick_analyzer.py   # Kural tabanlı analiz motoru
├── ml_analyzer.py            # ML analiz motoru
├── data_providers.py         # Veri kaynağı soyutlaması
├── train_model.py            # Offline model eğitim scripti
├── requirements.txt          # Python bağımlılıkları
├── model.joblib              # Pre-trained model (git'e eklenmez*)
├── model_meta.json           # Model metadata (git'e eklenmez*)
└── trade_report_*.txt        # Otomatik oluşturulan analizler
```

> \* `.gitignore` ile hariç tutulmuştur.

---

## Deploy

Streamlit Community Cloud kullanılır. `main` branch'e her `git push` sonrası otomatik yeniden deploy olur — manual republish gerekmez.
