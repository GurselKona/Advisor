#!/usr/bin/env python3
"""
Trade Advisor — Model Eğitimi

Çalıştır : python3 train_model.py
Çıktı    : model.joblib + model_meta.json (aynı klasöre)

Yahoo Finance'dan çok sayıda ticker için uzun vadeli günlük veri çeker,
özellikler üretir, etiketler ve RandomForest modeli eğitir.
Sonraki analizlerde ml_analyzer.py bu modeli yükleyip kullanır.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm

from ml_analyzer import _build_features, _make_labels, FORWARD_N, THRESHOLD, MODEL_PATH, META_PATH

# ── Eğitim evreni ─────────────────────────────────────────────────────────────

TICKERS = [
    # ABD büyük cap
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
    "JPM", "JNJ", "V", "UNH", "XOM", "WMT", "PG", "HD",
    # ETF
    "SPY", "QQQ", "IWM", "GLD", "TLT", "XLF", "XLE","SIL",
    # Volatil / teknoloji
    "NFLX", "AMD", "INTC", "CRM", "BABA",
    # Kripto
    "BTC-USD", "ETH-USD",
]

PERIOD_YEARS  = 5    # her ticker için kaç yıl geçmiş veri
N_TREES       = 500  # toplam ağaç sayısı
BATCH_SIZE    = 50   # ilerleme çubuğu için kaçar kaçar eğitilir

# ── Yardımcılar ───────────────────────────────────────────────────────────────

def fetch_ticker(ticker: str, years: int) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=years * 365)
    df = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
    )
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    return df[keep].round(4)


def ticker_to_xy(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = raw.copy().reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    feats  = _build_features(df)
    labels = _make_labels(df["close"], FORWARD_N, THRESHOLD)
    valid  = feats.notna().all(axis=1).copy()
    valid.iloc[-FORWARD_N:] = False
    return feats[valid].values, labels[valid].values


def train_with_progress(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """warm_start ile batch batch eğitir; her batch sonrası ilerleme çubuğu güncellenir."""
    clf = RandomForestClassifier(
        n_estimators=0,
        warm_start=True,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    batches = range(0, N_TREES, BATCH_SIZE)
    with tqdm(total=N_TREES, desc="  Ağaç eğitimi",
              unit="ağaç", ncols=65, colour="green") as pbar:
        for _ in batches:
            clf.n_estimators = min(clf.n_estimators + BATCH_SIZE, N_TREES)
            clf.fit(X_train, y_train)
            pbar.update(BATCH_SIZE)
    return clf


# ── Ana akış ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print(f"\n{'='*60}")
    print("  Trade Advisor — Model Eğitimi")
    print(f"  {len(TICKERS)} ticker · {PERIOD_YEARS} yıl · granülerite: 1d")
    print(f"  Hedef: {N_TREES} ağaç · forward_n={FORWARD_N} · threshold={THRESHOLD:.1%}")
    print(f"{'='*60}\n")

    # ── 1. Veri indirme ───────────────────────────────────────────────────────
    print("[ 1 / 3 ]  Veri indiriliyor…\n")

    all_X, all_y, ok_tickers, fail_tickers = [], [], [], []

    with tqdm(TICKERS, desc="  Ticker", unit="ticker", ncols=65, colour="cyan") as pbar:
        for ticker in pbar:
            pbar.set_postfix_str(ticker)
            try:
                raw = fetch_ticker(ticker, PERIOD_YEARS)
                if len(raw) < 120:
                    fail_tickers.append(f"{ticker}(az veri)")
                    continue
                X, y = ticker_to_xy(raw)
                all_X.append(X)
                all_y.append(y)
                ok_tickers.append(ticker)
            except Exception as exc:
                fail_tickers.append(f"{ticker}(hata)")
                tqdm.write(f"  ✗ {ticker}: {exc}")

    if not all_X:
        print("\nHiçbir tickerdan veri toplanamadı.")
        sys.exit(1)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    label_dist = {lbl: int((y == lbl).sum()) for lbl in ["BUY", "SELL", "NEUTRAL"]}

    print(f"\n  ✓ {len(ok_tickers)} ticker  ·  {len(X):,} toplam satır")
    print(f"  Dağılım — BUY: {label_dist['BUY']:,}  "
          f"SELL: {label_dist['SELL']:,}  "
          f"NEUTRAL: {label_dist['NEUTRAL']:,}")
    if fail_tickers:
        print(f"  Atlanan: {', '.join(fail_tickers)}")

    # ── 2. Eğitim ─────────────────────────────────────────────────────────────
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n[ 2 / 3 ]  Model eğitiliyor…")
    print(f"  Eğitim: {len(X_train):,} satır  ·  Test: {len(X_test):,} satır\n")

    clf = train_with_progress(X_train, y_train)

    # ── 3. Değerlendirme & kaydetme ───────────────────────────────────────────
    print(f"\n[ 3 / 3 ]  Değerlendirme & kaydetme…\n")

    report = classification_report(y_test, clf.predict(X_test), digits=3)
    print("  Test sonuçları:")
    for line in report.splitlines():
        print(f"  {line}")

    joblib.dump(clf, MODEL_PATH)
    meta = {
        "trained_at":    datetime.now().isoformat(),
        "period_years":  PERIOD_YEARS,
        "n_trees":       N_TREES,
        "tickers_ok":    ok_tickers,
        "tickers_fail":  fail_tickers,
        "total_samples": int(len(X)),
        "label_dist":    label_dist,
        "forward_n":     FORWARD_N,
        "threshold":     THRESHOLD,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    elapsed = time.time() - t0
    print(f"\n  ✓ Model    → {MODEL_PATH}")
    print(f"  ✓ Meta     → {META_PATH}")
    print(f"  ✓ Süre     → {elapsed/60:.1f} dakika")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
