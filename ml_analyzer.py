"""
ML tabanlı trade sinyal üreteci.
Arayüz: analyze(df) -> TradeRecommendation  (candlestick_analyzer ile aynı)

Öncelik sırası:
  1. model.joblib varsa yükle → sadece tahmin yap (hız + kalite)
  2. Yoksa mevcut veriyle eğit (fallback)

Model eğitmek için: python3 train_model.py
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from candlestick_analyzer import (
    TradeRecommendation, Signal, Strength,
    _ema, _rsi, _wavetrend, _compute_indicators, _indicator_score,
)

# ── Sabitler ──────────────────────────────────────────────────────────────────

MIN_BARS   = 120
FORWARD_N  = 5
THRESHOLD  = 0.005
MODEL_PATH = Path(__file__).parent / "model.joblib"
META_PATH  = Path(__file__).parent / "model_meta.json"


# ── Özellik üretimi ───────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    open_ = df["open"]
    high  = df["high"]
    low   = df["low"]

    ema20  = _ema(close, 20)
    ema50  = _ema(close, 50)
    ema100 = _ema(close, 100)
    ema200 = _ema(close, 200)
    rsi    = _rsi(close)
    wt1, wt2 = _wavetrend(high, low, close)

    rng = (high - low).replace(0, np.nan)
    hi  = pd.concat([open_, close], axis=1).max(axis=1)
    lo  = pd.concat([open_, close], axis=1).min(axis=1)

    return pd.DataFrame({
        "ema20_dist":  (close - ema20)  / ema20,
        "ema50_dist":  (close - ema50)  / ema50,
        "ema100_dist": (close - ema100) / ema100,
        "ema200_dist": (close - ema200) / ema200,
        "ema20_50":    (ema20  - ema50)  / ema50,
        "ema50_100":   (ema50  - ema100) / ema100,
        "ema100_200":  (ema100 - ema200) / ema200,
        "rsi":    rsi,
        "wt1":    wt1,
        "wt2":    wt2,
        "wt_diff": wt1 - wt2,
        "body_ratio":  (close - open_) / rng,
        "upper_wick":  (high - hi) / rng,
        "lower_wick":  (lo - low)  / rng,
        "ret1":  close.pct_change(1),
        "ret3":  close.pct_change(3),
        "ret5":  close.pct_change(5),
        "vol5":  close.pct_change().rolling(5).std(),
        "vol10": close.pct_change().rolling(10).std(),
        "atr_ratio": (high - low).rolling(14).mean() / close,
    }, index=df.index)


def _make_labels(close: pd.Series, forward_n: int, threshold: float) -> pd.Series:
    fwd = close.shift(-forward_n) / close - 1
    labels = pd.Series("NEUTRAL", index=close.index, dtype=object)
    labels[fwd >  threshold] = "BUY"
    labels[fwd < -threshold] = "SELL"
    return labels


# ── Model yükleme ─────────────────────────────────────────────────────────────

def _load_pretrained() -> RandomForestClassifier | None:
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def _load_meta() -> dict:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text())
        except Exception:
            pass
    return {}


# ── ATR + giriş seviyeleri ────────────────────────────────────────────────────

def _price_levels(df: pd.DataFrame, signal: Signal,
                  atr_multiplier: float, risk_reward: float):
    close = df["close"]
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - close.shift()).abs(),
        (df["low"]  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_val = float(tr.rolling(14).mean().iloc[-1])
    if math.isnan(atr_val):
        atr_val = float(df["high"].iloc[-1] - df["low"].iloc[-1])

    entry = float(close.iloc[-1])
    if signal == Signal.BUY:
        stop = entry - atr_val * atr_multiplier
        tp   = entry + (entry - stop) * risk_reward
    elif signal == Signal.SELL:
        stop = entry + atr_val * atr_multiplier
        tp   = entry - (stop - entry) * risk_reward
    else:
        entry = stop = tp = None
    return entry, stop, tp


# ── Ortak tahmin sonucu üretme ────────────────────────────────────────────────

def _build_rec(clf, last_x, df, risk_reward, atr_multiplier,
               model_source, training_bars, label_dist) -> TradeRecommendation:
    pred_label = clf.predict(last_x)[0]
    pred_proba = clf.predict_proba(last_x)[0]
    classes    = list(clf.classes_)
    confidence = float(pred_proba[classes.index(pred_label)])

    importances = sorted(
        zip(clf.feature_importances_,
            _build_features(df.iloc[:1]).columns),   # sütun adları için
        reverse=True,
    )[:5]
    top_feats = [(name, imp) for imp, name in importances]

    signal = Signal[pred_label]
    if confidence >= 0.65:
        strength = Strength.STRONG
    elif confidence >= 0.50:
        strength = Strength.MODERATE
    else:
        strength = Strength.WEAK

    inds = _compute_indicators(df)
    snap, ind_score = _indicator_score(inds, df)

    entry, stop, tp = _price_levels(df, signal, atr_multiplier, risk_reward)

    rec = TradeRecommendation(
        signal=signal, strength=strength,
        patterns=[], indicators=snap,
        entry_price=entry, stop_loss=stop, take_profit=tp,
        confidence=confidence, pattern_score=0, indicator_score=ind_score,
    )
    rec._feature_importances = top_feats        # type: ignore[attr-defined]
    rec._training_bars        = training_bars    # type: ignore[attr-defined]
    rec._label_dist           = label_dist       # type: ignore[attr-defined]
    rec._model_source         = model_source     # type: ignore[attr-defined]
    return rec


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def analyze(
    df: pd.DataFrame,
    forward_n: int = FORWARD_N,
    threshold: float = THRESHOLD,
    risk_reward: float = 2.0,
    atr_multiplier: float = 1.5,
) -> TradeRecommendation:
    df = df.copy().reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]

    if not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("DataFrame: open, high, low, close sütunları gerekli.")

    feats  = _build_features(df)
    last_x = feats.iloc[[-1]].values
    if np.isnan(last_x).any():
        last_x = feats.dropna().iloc[[-1]].values

    # ── 1. Pre-trained model ──────────────────────────────────────────────────
    clf = _load_pretrained()
    if clf is not None:
        meta = _load_meta()
        label_dist = meta.get("label_dist", {})
        return _build_rec(
            clf, last_x, df,
            risk_reward, atr_multiplier,
            model_source="pretrained",
            training_bars=meta.get("total_samples"),
            label_dist=label_dist,
        )

    # ── 2. Fallback: mevcut veriyle eğit ─────────────────────────────────────
    n = len(df)
    if n < MIN_BARS:
        raise ValueError(
            f"ML analizi için en az {MIN_BARS} bar gerekli — mevcut: {n}. "
            "Daha uzun bir dönem seçin ya da 'python3 train_model.py' ile "
            "pre-trained model oluşturun."
        )

    labels = _make_labels(df["close"], forward_n, threshold)
    valid  = feats.notna().all(axis=1).copy()
    valid.iloc[-forward_n:] = False

    X, y = feats[valid].values, labels[valid].values
    if len(X) < 60:
        raise ValueError("Geçerli satır sayısı yetersiz (min 60).")

    clf2 = RandomForestClassifier(
        n_estimators=300, max_depth=6,
        min_samples_leaf=5, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf2.fit(X, y)

    label_dist = {lbl: int((y == lbl).sum()) for lbl in ["BUY", "SELL", "NEUTRAL"]}
    return _build_rec(
        clf2, last_x, df,
        risk_reward, atr_multiplier,
        model_source="onthefly",
        training_bars=int(valid.sum()),
        label_dist=label_dist,
    )
