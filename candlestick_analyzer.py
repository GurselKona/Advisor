"""
Candlestick pattern detector + technical indicator trade signal generator.
Indicators: EMA 20/50/100/200, RSI 14, WaveTrend Oscillator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class Strength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


@dataclass
class PatternMatch:
    name: str
    signal: Signal
    strength: Strength
    index: int
    description: str
    bars_ago: int = 0  # 0 = son mum, 1 = bir önceki, 2 = iki önceki…


@dataclass
class IndicatorSnapshot:
    ema20: float
    ema50: float
    ema100: float
    ema200: float
    rsi: float
    wt1: float
    wt2: float
    close: float

    # derived
    ema_alignment: str = ""   # e.g. "BULL_FULL", "BEAR_PARTIAL", "MIXED"
    rsi_zone: str = ""        # OVERSOLD / NEUTRAL / OVERBOUGHT
    wt_zone: str = ""         # OVERSOLD / NEUTRAL / OVERBOUGHT
    wt_cross: str = ""        # BULL_CROSS / BEAR_CROSS / NONE

    def __post_init__(self) -> None:
        self.ema_alignment = self._ema_alignment()
        self.rsi_zone = self._rsi_zone()
        self.wt_zone = self._wt_zone()

    def _ema_alignment(self) -> str:
        c = self.close
        emas = [self.ema20, self.ema50, self.ema100, self.ema200]
        bull = c > emas[0] > emas[1] > emas[2] > emas[3]
        bear = c < emas[0] < emas[1] < emas[2] < emas[3]
        above = sum(c > e for e in emas)
        if bull:
            return "BULL_FULL"
        if bear:
            return "BEAR_FULL"
        if above >= 3:
            return "BULL_PARTIAL"
        if above <= 1:
            return "BEAR_PARTIAL"
        return "MIXED"

    def _rsi_zone(self) -> str:
        if self.rsi <= 30:
            return "OVERSOLD"
        if self.rsi >= 70:
            return "OVERBOUGHT"
        return "NEUTRAL"

    def _wt_zone(self) -> str:
        if self.wt1 <= -53:
            return "OVERSOLD"
        if self.wt1 >= 53:
            return "OVERBOUGHT"
        return "NEUTRAL"


@dataclass
class TradeRecommendation:
    signal: Signal
    strength: Strength
    patterns: list[PatternMatch] = field(default_factory=list)
    indicators: Optional[IndicatorSnapshot] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    pattern_score: int = 0
    indicator_score: int = 0

    def __str__(self) -> str:
        ind = self.indicators
        lines = [
            "=" * 62,
            f"  Signal      : {self.signal.value} ({self.strength.value})",
            f"  Confidence  : {self.confidence:.0%}  "
            f"(patterns {self.pattern_score:+d} / indicators {self.indicator_score:+d})",
            f"  Entry       : {self.entry_price:.4f}" if self.entry_price else "  Entry       : —",
            f"  Stop Loss   : {self.stop_loss:.4f}" if self.stop_loss else "  Stop Loss   : —",
            f"  Take Profit : {self.take_profit:.4f}" if self.take_profit else "  Take Profit : —",
            "=" * 62,
        ]

        if ind:
            lines += [
                "  INDICATORS",
                f"  EMA20={ind.ema20:.2f}  EMA50={ind.ema50:.2f}  "
                f"EMA100={ind.ema100:.2f}  EMA200={ind.ema200:.2f}",
                f"  EMA alignment : {ind.ema_alignment}",
                f"  RSI(14)       : {ind.rsi:.1f}  [{ind.rsi_zone}]",
                f"  WaveTrend     : WT1={ind.wt1:.2f}  WT2={ind.wt2:.2f}  "
                f"[{ind.wt_zone}]  cross={ind.wt_cross}",
                "-" * 62,
            ]

        if self.patterns:
            lines.append("  CANDLESTICK PATTERNS")
            for p in self.patterns:
                bar_label = "son mum" if p.bars_ago == 0 else f"{p.bars_ago} mum önce"
                lines.append(f"  [{p.signal.value:4s}] {p.name:<26s} ({p.strength.value})  [{bar_label}]")
                lines.append(f"         {p.description}")
        else:
            lines.append("  No candlestick patterns detected in lookback window.")

        lines.append("=" * 62)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _wavetrend(high: pd.Series, low: pd.Series, close: pd.Series,
               n1: int = 10, n2: int = 21, avg_len: int = 4) -> tuple[pd.Series, pd.Series]:
    ap = (high + low + close) / 3
    esa = _ema(ap, n1)
    d = _ema((ap - esa).abs(), n1)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = _ema(ci, n2)
    wt2 = wt1.rolling(avg_len).mean()
    return wt1, wt2


def _compute_indicators(df: pd.DataFrame) -> dict:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema100 = _ema(close, 100)
    ema200 = _ema(close, 200)
    rsi = _rsi(close)
    wt1, wt2 = _wavetrend(high, low, close)

    return {
        "ema20": ema20, "ema50": ema50,
        "ema100": ema100, "ema200": ema200,
        "rsi": rsi, "wt1": wt1, "wt2": wt2,
    }


def _indicator_score(inds: dict, df: pd.DataFrame) -> tuple[IndicatorSnapshot, int]:
    """Returns (IndicatorSnapshot at last bar, net score where + = bullish)."""
    n = len(df)
    last_close = df["close"].iloc[-1]

    def last(s: pd.Series) -> float:
        v = s.iloc[-1]
        return float(v) if not (isinstance(v, float) and math.isnan(v)) else float(s.dropna().iloc[-1])

    snap = IndicatorSnapshot(
        ema20=last(inds["ema20"]),
        ema50=last(inds["ema50"]),
        ema100=last(inds["ema100"]),
        ema200=last(inds["ema200"]),
        rsi=last(inds["rsi"]),
        wt1=last(inds["wt1"]),
        wt2=last(inds["wt2"]),
        close=last_close,
    )

    score = 0

    # --- EMA alignment ---
    if snap.ema_alignment == "BULL_FULL":
        score += 3
    elif snap.ema_alignment == "BULL_PARTIAL":
        score += 1
    elif snap.ema_alignment == "BEAR_FULL":
        score -= 3
    elif snap.ema_alignment == "BEAR_PARTIAL":
        score -= 1

    # --- RSI ---
    rsi = snap.rsi
    if rsi <= 20:
        score += 3
    elif rsi <= 30:
        score += 2
    elif rsi <= 40:
        score += 1
    elif rsi >= 80:
        score -= 3
    elif rsi >= 70:
        score -= 2
    elif rsi >= 60:
        score -= 1

    # --- WaveTrend zone ---
    wt1 = snap.wt1
    if wt1 <= -60:
        score += 3
    elif wt1 <= -53:
        score += 2
    elif wt1 >= 60:
        score -= 3
    elif wt1 >= 53:
        score -= 2

    # --- WaveTrend crossover (last 3 bars) ---
    wt1_s = inds["wt1"]
    wt2_s = inds["wt2"]
    cross = "NONE"
    if n >= 2:
        prev1 = wt1_s.iloc[-2]
        prev2 = wt2_s.iloc[-2]
        cur1 = wt1_s.iloc[-1]
        cur2 = wt2_s.iloc[-1]
        if prev1 <= prev2 and cur1 > cur2:
            cross = "BULL_CROSS"
            score += 3 if wt1 < -40 else 1
        elif prev1 >= prev2 and cur1 < cur2:
            cross = "BEAR_CROSS"
            score -= 3 if wt1 > 40 else 1

    snap.wt_cross = cross
    return snap, score


# ---------------------------------------------------------------------------
# Candlestick pattern helpers
# ---------------------------------------------------------------------------

def _body(o: float, c: float) -> float:
    return abs(c - o)


def _upper_wick(o: float, h: float, c: float) -> float:
    return h - max(o, c)


def _lower_wick(o: float, l: float, c: float) -> float:
    return min(o, c) - l


def _candle_range(h: float, l: float) -> float:
    return h - l


def _is_bullish(o: float, c: float) -> bool:
    return c > o


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------

class _Patterns:

    @staticmethod
    def doji(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        o, h, l, c = df.loc[i, ["open", "high", "low", "close"]]
        rng = _candle_range(h, l)
        if rng == 0:
            return None
        if _body(o, c) / rng < 0.05:
            return PatternMatch("Doji", Signal.NEUTRAL, Strength.WEAK, i,
                                "Body <5% of range — indecision, possible reversal.")
        return None

    @staticmethod
    def hammer(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        o, h, l, c = df.loc[i, ["open", "high", "low", "close"]]
        body = _body(o, c)
        lower = _lower_wick(o, l, c)
        upper = _upper_wick(o, h, c)
        rng = _candle_range(h, l)
        if rng == 0 or body == 0:
            return None
        if lower >= 2 * body and upper <= body * 0.5:
            return PatternMatch("Hammer", Signal.BUY, Strength.MODERATE, i,
                                "Long lower wick — buyers rejected lower prices, bullish reversal.")
        return None

    @staticmethod
    def inverted_hammer(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        o, h, l, c = df.loc[i, ["open", "high", "low", "close"]]
        body = _body(o, c)
        lower = _lower_wick(o, l, c)
        upper = _upper_wick(o, h, c)
        if body == 0:
            return None
        if upper >= 2 * body and lower <= body * 0.5:
            return PatternMatch("Inverted Hammer", Signal.BUY, Strength.WEAK, i,
                                "Long upper wick after downtrend — potential bullish reversal.")
        return None

    @staticmethod
    def shooting_star(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        o, h, l, c = df.loc[i, ["open", "high", "low", "close"]]
        body = _body(o, c)
        lower = _lower_wick(o, l, c)
        upper = _upper_wick(o, h, c)
        if body == 0:
            return None
        if upper >= 2 * body and lower <= body * 0.5 and not _is_bullish(o, c):
            return PatternMatch("Shooting Star", Signal.SELL, Strength.MODERATE, i,
                                "Long upper wick after uptrend — sellers pushed price down.")
        return None

    @staticmethod
    def marubozu_bull(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        o, h, l, c = df.loc[i, ["open", "high", "low", "close"]]
        rng = _candle_range(h, l)
        if rng == 0:
            return None
        if _is_bullish(o, c) and _body(o, c) / rng > 0.95:
            return PatternMatch("Bullish Marubozu", Signal.BUY, Strength.STRONG, i,
                                "Full-body candle with no wicks — strong buying pressure.")
        return None

    @staticmethod
    def marubozu_bear(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        o, h, l, c = df.loc[i, ["open", "high", "low", "close"]]
        rng = _candle_range(h, l)
        if rng == 0:
            return None
        if not _is_bullish(o, c) and _body(o, c) / rng > 0.95:
            return PatternMatch("Bearish Marubozu", Signal.SELL, Strength.STRONG, i,
                                "Full-body bearish candle — strong selling pressure.")
        return None

    @staticmethod
    def spinning_top(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        o, h, l, c = df.loc[i, ["open", "high", "low", "close"]]
        rng = _candle_range(h, l)
        if rng == 0:
            return None
        body = _body(o, c)
        upper = _upper_wick(o, h, c)
        lower = _lower_wick(o, l, c)
        if body / rng < 0.3 and upper > body and lower > body:
            return PatternMatch("Spinning Top", Signal.NEUTRAL, Strength.WEAK, i,
                                "Small body with both wicks — market indecision.")
        return None

    @staticmethod
    def bullish_engulfing(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        if i < 1:
            return None
        o1, c1 = df.loc[i - 1, ["open", "close"]]
        o2, c2 = df.loc[i, ["open", "close"]]
        if not _is_bullish(o2, c2) or _is_bullish(o1, c1):
            return None
        if o2 <= c1 and c2 >= o1:
            return PatternMatch("Bullish Engulfing", Signal.BUY, Strength.STRONG, i,
                                "Bullish candle fully engulfs prior bearish candle — strong reversal.")
        return None

    @staticmethod
    def bearish_engulfing(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        if i < 1:
            return None
        o1, c1 = df.loc[i - 1, ["open", "close"]]
        o2, c2 = df.loc[i, ["open", "close"]]
        if _is_bullish(o2, c2) or not _is_bullish(o1, c1):
            return None
        if o2 >= c1 and c2 <= o1:
            return PatternMatch("Bearish Engulfing", Signal.SELL, Strength.STRONG, i,
                                "Bearish candle fully engulfs prior bullish candle — strong reversal.")
        return None

    @staticmethod
    def tweezer_bottom(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        if i < 1:
            return None
        l1 = df.loc[i - 1, "low"]
        l2 = df.loc[i, "low"]
        o2, c2 = df.loc[i, ["open", "close"]]
        tol = (df.loc[i, "high"] - l2) * 0.01
        if abs(l1 - l2) <= tol and _is_bullish(o2, c2):
            return PatternMatch("Tweezer Bottom", Signal.BUY, Strength.MODERATE, i,
                                "Two candles share same low — strong support, bullish.")
        return None

    @staticmethod
    def tweezer_top(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        if i < 1:
            return None
        h1 = df.loc[i - 1, "high"]
        h2 = df.loc[i, "high"]
        o2, c2 = df.loc[i, ["open", "close"]]
        tol = (h2 - df.loc[i, "low"]) * 0.01
        if abs(h1 - h2) <= tol and not _is_bullish(o2, c2):
            return PatternMatch("Tweezer Top", Signal.SELL, Strength.MODERATE, i,
                                "Two candles share same high — strong resistance, bearish.")
        return None

    @staticmethod
    def piercing_line(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        if i < 1:
            return None
        o1, c1 = df.loc[i - 1, ["open", "close"]]
        o2, c2 = df.loc[i, ["open", "close"]]
        if _is_bullish(o1, c1) or not _is_bullish(o2, c2):
            return None
        midpoint = (o1 + c1) / 2
        if o2 < c1 and c2 > midpoint and c2 < o1:
            return PatternMatch("Piercing Line", Signal.BUY, Strength.MODERATE, i,
                                "Bullish candle closes above midpoint of prior bearish candle.")
        return None

    @staticmethod
    def dark_cloud_cover(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        if i < 1:
            return None
        o1, c1 = df.loc[i - 1, ["open", "close"]]
        o2, c2 = df.loc[i, ["open", "close"]]
        if not _is_bullish(o1, c1) or _is_bullish(o2, c2):
            return None
        midpoint = (o1 + c1) / 2
        if o2 > c1 and c2 < midpoint and c2 > o1:
            return PatternMatch("Dark Cloud Cover", Signal.SELL, Strength.MODERATE, i,
                                "Bearish candle closes below midpoint of prior bullish candle.")
        return None

    @staticmethod
    def morning_star(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        if i < 2:
            return None
        o1, c1 = df.loc[i - 2, ["open", "close"]]
        o2, c2 = df.loc[i - 1, ["open", "close"]]
        o3, c3 = df.loc[i, ["open", "close"]]
        avg = df["body"].iloc[max(0, i - 14):i].mean()
        if (not _is_bullish(o1, c1) and _body(o1, c1) > avg * 0.8
                and _body(o2, c2) < avg * 0.3
                and _is_bullish(o3, c3) and c3 > (o1 + c1) / 2):
            return PatternMatch("Morning Star", Signal.BUY, Strength.STRONG, i,
                                "3-candle reversal: large bearish → small body → large bullish.")
        return None

    @staticmethod
    def evening_star(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        if i < 2:
            return None
        o1, c1 = df.loc[i - 2, ["open", "close"]]
        o2, c2 = df.loc[i - 1, ["open", "close"]]
        o3, c3 = df.loc[i, ["open", "close"]]
        avg = df["body"].iloc[max(0, i - 14):i].mean()
        if (_is_bullish(o1, c1) and _body(o1, c1) > avg * 0.8
                and _body(o2, c2) < avg * 0.3
                and not _is_bullish(o3, c3) and c3 < (o1 + c1) / 2):
            return PatternMatch("Evening Star", Signal.SELL, Strength.STRONG, i,
                                "3-candle reversal: large bullish → small body → large bearish.")
        return None

    @staticmethod
    def three_white_soldiers(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        if i < 2:
            return None
        rows = [df.loc[i - 2], df.loc[i - 1], df.loc[i]]
        if all(_is_bullish(r["open"], r["close"]) for r in rows):
            closes = [r["close"] for r in rows]
            opens = [r["open"] for r in rows]
            if closes[0] < closes[1] < closes[2] and opens[1] > opens[0] and opens[2] > opens[1]:
                return PatternMatch("Three White Soldiers", Signal.BUY, Strength.STRONG, i,
                                    "3 consecutive bullish candles — sustained buying pressure.")
        return None

    @staticmethod
    def three_black_crows(df: pd.DataFrame, i: int) -> Optional[PatternMatch]:
        if i < 2:
            return None
        rows = [df.loc[i - 2], df.loc[i - 1], df.loc[i]]
        if all(not _is_bullish(r["open"], r["close"]) for r in rows):
            closes = [r["close"] for r in rows]
            opens = [r["open"] for r in rows]
            if closes[0] > closes[1] > closes[2] and opens[1] < opens[0] and opens[2] < opens[1]:
                return PatternMatch("Three Black Crows", Signal.SELL, Strength.STRONG, i,
                                    "3 consecutive bearish candles — sustained selling pressure.")
        return None


_ALL_DETECTORS = [
    _Patterns.doji, _Patterns.hammer, _Patterns.inverted_hammer,
    _Patterns.shooting_star, _Patterns.marubozu_bull, _Patterns.marubozu_bear,
    _Patterns.spinning_top, _Patterns.bullish_engulfing, _Patterns.bearish_engulfing,
    _Patterns.tweezer_bottom, _Patterns.tweezer_top, _Patterns.piercing_line,
    _Patterns.dark_cloud_cover, _Patterns.morning_star, _Patterns.evening_star,
    _Patterns.three_white_soldiers, _Patterns.three_black_crows,
]


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

def analyze(
    df: pd.DataFrame,
    lookback: int = 5,
    risk_reward: float = 2.0,
    atr_multiplier: float = 1.5,
) -> TradeRecommendation:
    """
    Analyze OHLC DataFrame using candlestick patterns + EMA/RSI/WaveTrend.

    Parameters
    ----------
    df            : DataFrame with columns open, high, low, close.
    lookback      : recent candles to scan for patterns.
    risk_reward   : take-profit distance / stop-loss distance.
    atr_multiplier: stop = entry ± ATR(14) * atr_multiplier.
    """
    required = {"open", "high", "low", "close"}
    df = df.copy().reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    df["body"] = (df["close"] - df["open"]).abs()

    # ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    n = len(df)
    start = max(2, n - lookback)

    # --- Candlestick patterns ---
    matches: list[PatternMatch] = []
    for i in range(start, n):
        for detector in _ALL_DETECTORS:
            result = detector(df, i)
            if result is not None:
                result.bars_ago = n - 1 - i
                matches.append(result)

    pattern_buy = sum(
        3 if m.strength == Strength.STRONG else 2 if m.strength == Strength.MODERATE else 1
        for m in matches if m.signal == Signal.BUY
    )
    pattern_sell = sum(
        3 if m.strength == Strength.STRONG else 2 if m.strength == Strength.MODERATE else 1
        for m in matches if m.signal == Signal.SELL
    )
    pat_net = pattern_buy - pattern_sell

    # --- Indicators ---
    inds = _compute_indicators(df)
    snap, ind_net = _indicator_score(inds, df)

    # --- Combine ---
    total_net = pat_net + ind_net

    last = df.iloc[-1]
    last_atr = atr.iloc[-1]
    if math.isnan(last_atr):
        last_atr = last["high"] - last["low"]

    if total_net > 0:
        signal = Signal.BUY
        net = total_net
        entry = last["close"]
        stop = entry - last_atr * atr_multiplier
        tp = entry + (entry - stop) * risk_reward
    elif total_net < 0:
        signal = Signal.SELL
        net = abs(total_net)
        entry = last["close"]
        stop = entry + last_atr * atr_multiplier
        tp = entry - (stop - entry) * risk_reward
    else:
        signal = Signal.NEUTRAL
        net = 0
        entry = stop = tp = None

    if net >= 8:
        strength = Strength.STRONG
    elif net >= 4:
        strength = Strength.MODERATE
    else:
        strength = Strength.WEAK

    max_possible = 12 + 9  # pattern ceiling ~12, indicator ~9
    confidence = min(1.0, net / max_possible)

    return TradeRecommendation(
        signal=signal,
        strength=strength,
        patterns=matches,
        indicators=snap,
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,
        confidence=confidence,
        pattern_score=pat_net,
        indicator_score=ind_net,
    )


def from_csv(path: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    rename = {col: col.lower() for col in df.columns
              if col.lower() in {"open", "high", "low", "close", "volume"}}
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

def _demo() -> None:
    np.random.seed(42)
    n = 250
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    opens = np.roll(closes, 1)
    opens[0] = closes[0] - 0.2
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 0.3)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 0.3)
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})

    print("  CANDLESTICK + INDICATOR ANALYSIS — DEMO")
    rec = analyze(df, lookback=10)
    print(rec)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        data = from_csv(sys.argv[1])
        rec = analyze(data)
        print(rec)
    else:
        _demo()
