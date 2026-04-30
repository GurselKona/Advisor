#!/usr/bin/env python3
"""
Veri kaynağı soyutlama katmanı.

Yeni kaynak eklemek için:
  1. DataProvider'dan miras alan bir sınıf yaz
  2. fetch() ve isteğe bağlı search() / detect_interval() metodlarını uygula
  3. REGISTRY sözlüğüne ekle
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import pandas as pd
from curl_cffi.requests import Session as CurlSession

# Kurumsal SSL proxy (MITM) senaryolarında curl_cffi'nin gömülü libcurl'ü
# şirket sertifikasını doğrulayamaz. Uygulama yerel masaüstü aracı olduğundan
# verify=False güvenli kabul edilir; gelen veriyi değil bağlantı kimliğini etkiler.
_SESSION = CurlSession(impersonate="chrome", verify=False)


# ── Yardımcı: kapalı gün temizliği ───────────────────────────────────────────

def _drop_closed_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Piyasanın kapalı olduğu günlere ait satırları çıkarır.
    Kriter: Volume == 0 VEYA (High == Low == Open == Close) olan satırlar.
    7/24 açık piyasalarda (kripto/forex) Volume hiç 0 olmaz, bu fonksiyon etkisizdir.
    """
    mask = pd.Series(True, index=df.index)

    if "Volume" in df.columns:
        mask &= df["Volume"] > 0

    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        flat = (df["High"] == df["Low"]) & (df["Open"] == df["Close"]) & (df["High"] == df["Open"])
        mask &= ~flat

    return df[mask]


# ── Yardımcı: interval tespiti ────────────────────────────────────────────────

def _infer_interval(df: pd.DataFrame) -> str | None:
    """DataFrame index'inden granülariteyi tahmin eder."""
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return None
    diffs = pd.Series(df.index).diff().dropna()
    median_sec = diffs.dt.total_seconds().median()
    thresholds = [
        (90,           "1m"),
        (150,          "2m"),
        (450,          "5m"),
        (1_200,        "15m"),
        (2_700,        "30m"),
        (5_400,        "60m"),
        (10_800,       "2h"),
        (21_600,       "4h"),
        (129_600,      "1d"),   # < 1.5 gün
        (864_000,      "1wk"),  # < 10 gün
        (float("inf"), "1mo"),
    ]
    for threshold, iv in thresholds:
        if median_sec <= threshold:
            return iv
    return "1mo"


# ── Temel arayüz ──────────────────────────────────────────────────────────────

class DataProvider(ABC):
    name: str = ""
    description: str = ""
    ticker_label: str = "Ticker"
    ticker_placeholder: str = ""

    @abstractmethod
    def fetch(self, ticker: str, interval: str, start: str, end: str) -> pd.DataFrame:
        """
        OHLCV verisi döndürür.

        Parametreler
        ------------
        ticker   : enstrüman kodu (provider'a özgü format)
        interval : standart interval ("1m","5m","15m","30m","60m","2h","4h","1d","1wk","1mo")
        start    : "YYYY-MM-DD"
        end      : "YYYY-MM-DD"

        Döndürür
        --------
        DatetimeIndex (UTC, tz-naive) + Open, High, Low, Close[, Volume] sütunlu DataFrame
        """

    def search(self, query: str) -> list[tuple[str, str]]:
        """Sorguya uyan önerileri döndürür: [(görünen_etiket, sembol), ...]"""
        return []

    def max_days_for_interval(self, interval: str) -> int | None:
        """Kısa vadeli interval için maksimum geçmiş gün sayısı. Sınır yoksa None."""
        return None

    def detect_interval(self, df: pd.DataFrame) -> str | None:
        """
        Çekilen veriden granülariteyi tespit eder.
        Tespit desteklenmiyorsa None döner; UI seçili interval'ı kullanır.
        """
        return None


# ── Yahoo Finance ─────────────────────────────────────────────────────────────

class YahooFinanceProvider(DataProvider):
    name = "Yahoo Finance"
    description = "Hisse, ETF, kripto, döviz, endeks"
    ticker_label = "Ticker"
    ticker_placeholder = "AAPL, TSLA, BTC-USD, EURUSD=X, GC=F (Altın), SI=F (Gümüş)…"

    _INTRADAY_LIMITS: dict[str, int] = {
        "1m": 7, "2m": 60, "5m": 60,
        "15m": 60, "30m": 60, "60m": 730,
        "2h": 730, "4h": 730,
    }
    # 2h/4h: 60m veri çek, sonra resample et
    _RESAMPLE: dict[str, str] = {"2h": "2h", "4h": "4h"}

    def fetch(self, ticker: str, interval: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        fetch_iv = "60m" if interval in self._RESAMPLE else interval
        df = yf.Ticker(ticker, session=_SESSION).history(start=start, end=end, interval=fetch_iv)
        if df.empty:
            return df

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        df = df[keep]

        if interval in self._RESAMPLE:
            agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            if "Volume" in df.columns:
                agg["Volume"] = "sum"
            df = df.resample(self._RESAMPLE[interval]).agg(agg).dropna()

        return _drop_closed_days(df).round(4)

    def search(self, query: str) -> list[tuple[str, str]]:
        if not query:
            return []
        try:
            import yfinance as yf
            quotes = [
                q for q in yf.Search(query, max_results=8, enable_fuzzy_query=True, session=_SESSION).quotes
                if q.get("symbol")
            ]
            return [
                (
                    f"{q['symbol']}  —  "
                    f"{(q.get('shortname') or q.get('longname') or '')[:30]}"
                    f"  [{q.get('exchDisp') or q.get('exchange') or ''}]",
                    q["symbol"],
                )
                for q in quotes
            ]
        except Exception:
            return []

    def max_days_for_interval(self, interval: str) -> int | None:
        return self._INTRADAY_LIMITS.get(interval)


# ── CSV Dosyası ───────────────────────────────────────────────────────────────

class CsvFileProvider(DataProvider):
    name = "CSV Dosyası"
    description = "Yerel CSV — sütunlar: date/datetime, Open, High, Low, Close[, Volume]"
    ticker_label = "CSV Dosyası"
    ticker_placeholder = "Dosya adı yazın veya listeden seçin…"

    _DEFAULT_DIR: str = os.path.dirname(os.path.abspath(__file__))

    def fetch(self, ticker: str, interval: str, start: str, end: str) -> pd.DataFrame:
        """ticker = CSV dosyasının tam yolu"""
        df = pd.read_csv(ticker)

        col_map = {c: c.capitalize() for c in df.columns
                   if c.lower() in {"open", "high", "low", "close", "volume"}}
        df = df.rename(columns=col_map)

        date_col = next(
            (c for c in df.columns
             if c.lower() in {"date", "datetime", "timestamp", "time"}),
            None,
        )
        if date_col:
            df.index = pd.to_datetime(df[date_col])
            df = df.drop(columns=[date_col])
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            if start and end:
                df = df.loc[start:end]

        keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        if not keep:
            raise ValueError("CSV'de Open/High/Low/Close sütunları bulunamadı.")
        return _drop_closed_days(df[keep]).round(4)

    def search(self, query: str) -> list[tuple[str, str]]:
        """Varsayılan dizindeki CSV dosyalarını listeler; query ile ada göre filtreler."""
        import glob
        all_csv = sorted(glob.glob(os.path.join(self._DEFAULT_DIR, "*.csv")))
        if query:
            q = query.lower()
            all_csv = [f for f in all_csv if q in os.path.basename(f).lower()]
        return [(os.path.basename(f), f) for f in all_csv]



# ── Kayıt defteri ─────────────────────────────────────────────────────────────

REGISTRY: dict[str, DataProvider] = {
    YahooFinanceProvider.name: YahooFinanceProvider(),
    CsvFileProvider.name:      CsvFileProvider(),
}
