"""
Gaussian HMM ile piyasa rejimi tespiti.

detect_regime(df) → RegimeInfo | None
apply_regime_filter(rec, regime) → rec  (çelişen sinyali downgrade eder)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MIN_BARS = 60
N_STATES = 3

REGIME_ICON = {"Yükseliş": "🟢", "Düşüş": "🔴", "Yatay": "🟡"}


@dataclass
class RegimeInfo:
    label: str           # "Yükseliş" | "Düşüş" | "Yatay"
    state: int           # ham HMM state indeksi
    state_series: pd.Series  # index=df.index subset, değer=label (grafik/debug için)
    filtered: bool = False   # sinyal downgrade yapıldı mı


def detect_regime(df: pd.DataFrame, n_states: int = N_STATES) -> RegimeInfo | None:
    """
    Log getiri + 20-bar volatilite üzerinde Gaussian HMM fit eder.
    Yeterli bar yoksa None döner.
    """
    from hmmlearn.hmm import GaussianHMM

    close = df["Close"] if "Close" in df.columns else df["close"]

    log_ret = np.log(close / close.shift(1)).dropna()
    vol     = log_ret.rolling(20).std().dropna()

    common  = log_ret.index.intersection(vol.index)
    X       = np.column_stack([log_ret[common].values, vol[common].values])

    if len(X) < MIN_BARS:
        return None

    try:
        import warnings
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=300,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
        states = model.predict(X)
    except Exception:
        return None

    # State'leri ortalama getiriye göre etiketle
    means = [
        float(X[states == i, 0].mean()) if (states == i).any() else 0.0
        for i in range(n_states)
    ]
    ranked = sorted(range(n_states), key=lambda i: means[i])
    label_map: dict[int, str] = {}
    label_map[ranked[0]]  = "Düşüş"
    label_map[ranked[-1]] = "Yükseliş"
    for s in range(n_states):
        if s not in label_map:
            label_map[s] = "Yatay"

    current_state = int(states[-1])
    state_series  = pd.Series(
        [label_map[s] for s in states], index=common, name="regime"
    )

    return RegimeInfo(
        label=label_map[current_state],
        state=current_state,
        state_series=state_series,
    )


def apply_regime_filter(rec, regime: RegimeInfo):
    """
    Rejim ile sinyal çelişiyorsa STRONG → MODERATE'e düşürür.
    rec nesnesi yerinde değiştirilir; aynı nesne döner.
    """
    from candlestick_analyzer import Signal, Strength

    conflict = (
        (regime.label == "Yükseliş" and rec.signal == Signal.SELL) or
        (regime.label == "Düşüş"    and rec.signal == Signal.BUY)
    )

    if conflict and rec.strength == Strength.STRONG:
        rec.strength   = Strength.MODERATE
        regime.filtered = True

    return rec
