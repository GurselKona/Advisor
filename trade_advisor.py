#!/usr/bin/env python3
"""Trade Advisor — Streamlit browser UI"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_searchbox import st_searchbox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candlestick_analyzer
import ml_analyzer
from candlestick_analyzer import Signal, Strength
from data_providers import REGISTRY, DataProvider
from hmm_regime import detect_regime, apply_regime_filter, REGIME_ICON

ENGINES = {
    "Kural Tabanlı": candlestick_analyzer,
    "Makine Öğrenmesi": ml_analyzer,
}

# ── Sabitler ──────────────────────────────────────────────────────────────────

INTERVALS = [
    ("1m",  "1 Dakika"), ("2m",  "2 Dakika"), ("5m",  "5 Dakika"),
    ("15m", "15 Dakika"), ("30m", "30 Dakika"), ("60m", "1 Saat"),
    ("2h",  "2 Saat"),   ("4h",  "4 Saat"),
    ("1d",  "1 Gün"),    ("1wk", "1 Hafta"),   ("1mo", "1 Ay"),
]
# Tüm aday presetler — her interval kendi limitine göre filtrelenir
_ALL_PRESETS = [
    (1,    "Son 1 gün"),
    (3,    "Son 3 gün"),
    (7,    "Son 7 gün"),
    (30,   "Son 1 ay"),
    (90,   "Son 3 ay"),
    (180,  "Son 6 ay"),
    (365,  "Son 1 yıl"),
    (730,  "Son 2 yıl"),
    (1095, "Son 3 yıl"),
    (1460, "Son 4 yıl"),
    (1825, "Son 5 yıl"),
]

def build_presets(max_days: int | None) -> list[tuple[str, int]]:
    """max_days limitine sığan presetleri döndürür. None → limitsiz (günlük+)."""
    limit = max_days if max_days is not None else 9999
    return [(label, days) for days, label in _ALL_PRESETS if days <= limit]

SIG_ICON  = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}
SIG_COLOR = {"BUY": "#1a7f4b", "SELL": "#b91c1c", "NEUTRAL": "#92400e"}
SIG_BG    = {"BUY": "#d1fae5", "SELL": "#fee2e2", "NEUTRAL": "#fef3c7"}
STR_STARS = {"STRONG": "⭐⭐⭐", "MODERATE": "⭐⭐", "WEAK": "⭐"}

# ── Candlestick grafik ────────────────────────────────────────────────────────

def make_candle_chart(df: pd.DataFrame, ticker: str, interval: str, rec) -> go.Figure:
    close = df["Close"]
    ema20  = close.ewm(span=20,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    has_volume = "Volume" in df.columns and df["Volume"].sum() > 0
    row_heights = [0.72, 0.28] if has_volume else [1.0]
    rows = 2 if has_volume else 1

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
    )

    # Candlestick
    is_dt = isinstance(df.index, pd.DatetimeIndex)
    has_time = is_dt and ((df.index.hour != 0).any() or (df.index.minute != 0).any())
    has_weekend = is_dt and df.index[:min(60, len(df))].dayofweek.isin([5, 6]).any()
    hover_fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"

    # x değerlerini Python datetime listesine çevir — Plotly 6.x için en güvenli format
    if is_dt:
        x_vals = df.index.to_pydatetime().tolist()
    else:
        x_vals = df.index.tolist()

    fig.add_trace(
        go.Candlestick(
            x=x_vals,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name=ticker,
            increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
            decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
            hovertext=[
                f"<b>{ts.strftime(hover_fmt)}</b><br>"
                f"A: {o:.4f}  Y: {h:.4f}  D: {l:.4f}  K: {c:.4f}"
                for ts, o, h, l, c in zip(
                    df.index, df["Open"], df["High"], df["Low"], df["Close"]
                )
            ],
            hoverinfo="text",
        ),
        row=1, col=1,
    )

    # EMA çizgileri
    for period, series, color in [
        (20,  ema20,  "#f6c90e"),
        (50,  ema50,  "#3fc1c9"),
        (100, ema100, "#fc5185"),
        (200, ema200, "#a3de83"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=series.tolist(),
                name=f"EMA{period}",
                line=dict(width=1.2, color=color),
                hovertemplate=f"EMA{period}: %{{y:.4f}}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Giriş / Stop / TP yatay çizgileri
    if rec.entry_price:
        for price, label, color, dash in [
            (rec.entry_price, "Giriş",       "#60a5fa", "dash"),
            (rec.stop_loss,   "Stop Loss",   "#f87171", "dot"),
            (rec.take_profit, "Take Profit", "#34d399", "dot"),
        ]:
            fig.add_hline(
                y=price, row=1, col=1,
                line=dict(color=color, width=1.2, dash=dash),
                annotation_text=f" {label}: {price:.4f}",
                annotation_position="right",
                annotation_font=dict(color=color, size=11),
            )

    # Hacim
    if has_volume:
        bar_colors = [
            "#26a69a" if c >= o else "#ef5350"
            for o, c in zip(df["Open"], df["Close"])
        ]
        fig.add_trace(
            go.Bar(
                x=x_vals, y=df["Volume"].tolist(),
                name="Hacim",
                marker_color=bar_colors,
                opacity=0.55,
                hovertemplate="Hacim: %{y:,.0f}<extra></extra>",
            ),
            row=2, col=1,
        )

    # Crosshair + layout
    spike_style = dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#6b7280",
        spikethickness=1,
        spikedash="dot",
    )
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b>  [{interval}]", font=dict(size=15, color="#e5e7eb")),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#d1d5db", size=11),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1f2937", font_color="#f9fafb", font_size=12),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=120, t=55, b=10),
    )

    # Hafta sonu boşluklarını gizle — sadece Mon-Fri piyasalar için
    rangebreaks = []
    if not has_weekend:
        rangebreaks.append(dict(bounds=["sat", "mon"]))

    fig.update_xaxes(**spike_style, gridcolor="#1f2937", zeroline=False, showline=False,
                     rangebreaks=rangebreaks)
    fig.update_yaxes(
        **spike_style, gridcolor="#1f2937", zeroline=False,
        showline=False, side="right",
    )

    return fig

# ── Rapor metni ───────────────────────────────────────────────────────────────

def format_report(ticker: str, interval: str, start: str, end: str, rec) -> str:
    ind = rec.indicators
    lines = [
        f"\n{'='*65}",
        f"  TICKER        : {ticker}",
        f"  Granülerite   : {interval}",
        f"  Tarih Aralığı : {start} → {end}",
        f"  Analiz Zamanı : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{'='*65}",
        f"  Sinyal        : {rec.signal.value} ({rec.strength.value})",
        f"  Güven         : {rec.confidence:.0%}  "
        f"(pattern {rec.pattern_score:+d} / indikatör {rec.indicator_score:+d})",
    ]
    if rec.entry_price:
        lines += [
            f"  Giriş Fiyatı  : {rec.entry_price:.4f}",
            f"  Stop Loss     : {rec.stop_loss:.4f}",
            f"  Take Profit   : {rec.take_profit:.4f}",
        ]
    else:
        lines.append("  Fiyat Seviyeleri : — (Nötr sinyal)")

    if ind:
        lines += [
            f"{'-'*65}",
            "  İNDİKATÖRLER",
            f"  EMA20={ind.ema20:.2f}  EMA50={ind.ema50:.2f}  "
            f"EMA100={ind.ema100:.2f}  EMA200={ind.ema200:.2f}",
            f"  EMA Hizalama  : {ind.ema_alignment}",
            f"  RSI(14)       : {ind.rsi:.1f}  [{ind.rsi_zone}]",
            f"  WaveTrend     : WT1={ind.wt1:.2f}  WT2={ind.wt2:.2f}  "
            f"[{ind.wt_zone}]  cross={ind.wt_cross}",
        ]

    if rec.patterns:
        lines += [f"{'-'*65}", "  MUM FORMASYONLARI"]
        for p in rec.patterns:
            lines.append(
                f"  [{p.signal.value:4s}] {p.name:<28s} ({p.strength.value})  bar={p.index}"
            )
            lines.append(f"           {p.description}")
    else:
        lines += [f"{'-'*65}", "  Mum formasyonu tespit edilmedi."]

    lines.append(f"{'='*65}")
    return "\n".join(lines)

# ── Ana uygulama ──────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Trade Advisor", page_icon="📈", layout="wide")

    st.markdown("""
    <style>
    .banner {
        padding: 14px 20px; border-radius: 10px; font-size: 1.25em;
        font-weight: bold; text-align: center; margin-bottom: 18px;
        border-width: 2px; border-style: solid;
    }
    div[data-testid="stMetricValue"] { font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📈 Trade Advisor")
    st.caption("Otomatik candlestick + teknik indikatör analizi · En fazla 5 ticker")

    # ── Parametreler ──────────────────────────────────────────────────────────
    st.subheader("⚙️ Parametreler")

    # Veri kaynağı ve motor seçimi
    src_col, eng_col, _ = st.columns([1.5, 1.5, 1])
    with src_col:
        provider_name = st.selectbox(
            "Veri Kaynağı",
            list(REGISTRY.keys()),
            help="\n".join(f"**{n}**: {p.description}" for n, p in REGISTRY.items()),
        )
    with eng_col:
        engine_name = st.selectbox(
            "Analiz Motoru",
            list(ENGINES.keys()),
            help="**Kural Tabanlı**: EMA/RSI/WaveTrend + mum formasyonları\n"
                 "**Makine Öğrenmesi**: RandomForest — en az 120 bar gerekli",
        )
    provider: DataProvider = REGISTRY[provider_name]
    engine = ENGINES[engine_name]

    # Ticker / dosya girişleri
    ticker_cols = st.columns(5)
    tickers = []
    for i, col in enumerate(ticker_cols):
        with col:
            st.caption(f"{provider.ticker_label} {i + 1}")
            val = st_searchbox(
                provider.search,
                key=f"sb_{provider_name}_{i}",
                placeholder=provider.ticker_placeholder or f"{provider.ticker_label} {i+1}",
                default_use_searchterm=True,
                clear_on_submit=False,
                debounce=300,
            )
            if val:
                tickers.append(str(val).strip().upper())

    row2 = st.columns([1.5, 1.5, 1, 1, 0.8])

    with row2[0]:
        iv_labels = [f"{lbl}  [{iv}]" for iv, lbl in INTERVALS]
        sel_iv_lbl = st.selectbox("Granülerite", iv_labels, index=6)
        interval = next(iv for iv, lbl in INTERVALS if f"{lbl}  [{iv}]" == sel_iv_lbl)

    max_days = provider.max_days_for_interval(interval)
    is_intraday = max_days is not None
    presets = build_presets(max_days)

    with row2[1]:
        preset_names = [p[0] for p in presets] + ["Özel tarih"]
        preset_sel = st.selectbox("Dönem", preset_names, index=len(presets) - 1)

    if preset_sel == "Özel tarih":
        with row2[2]:
            start_date = st.date_input("Başlangıç", value=datetime.today() - timedelta(days=365))
        with row2[3]:
            end_date = st.date_input("Bitiş", value=datetime.today())
    else:
        days = dict(presets)[preset_sel]
        if is_intraday and max_days:
            days = min(days, max_days)
        end_date   = datetime.today().date()
        start_date = (datetime.today() - timedelta(days=days)).date()
        with row2[2]:
            st.metric("Başlangıç", start_date.strftime("%Y-%m-%d"))
        with row2[3]:
            st.metric("Bitiş", end_date.strftime("%Y-%m-%d"))

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    with row2[4]:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_clicked = st.button(
            "🔍 Analiz Et", type="primary",
            disabled=(len(tickers) == 0),
            width="stretch",
        )

    st.divider()

    # ── Analiz ────────────────────────────────────────────────────────────────
    if analyze_clicked and tickers:
        results, errors = {}, {}
        prog   = st.progress(0)
        status = st.empty()

        for i, ticker in enumerate(tickers):
            status.info(f"⏳ {ticker} verisi çekiliyor ve analiz ediliyor…")
            prog.progress(i / len(tickers))
            try:
                df = provider.fetch(ticker, interval, start_str, end_str)
                if df.empty:
                    errors[ticker] = "Veri bulunamadı — ticker doğru mu, dönem yeterince kısa mu?"
                    continue
                rec = engine.analyze(df)
                regime = detect_regime(df)
                if regime is not None:
                    apply_regime_filter(rec, regime)
                effective_iv = provider.detect_interval(df) or interval
                results[ticker] = {"df": df, "rec": rec, "interval": effective_iv, "regime": regime}
            except Exception as exc:
                errors[ticker] = str(exc)

        prog.progress(1.0)
        status.empty()

        # Raporu oluştur ve kaydet
        if results:
            header = (
                f"TRADE ADVISOR RAPORU\n"
                f"Oluşturma  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Granülerite: {interval}  |  {start_str} → {end_str}\n"
            )
            body = "".join(
                format_report(t, d["interval"], start_str, end_str, d["rec"])
                for t, d in results.items()
            )
            all_text = header + body

            fname = f"trade_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(all_text)

            st.session_state.update(
                results=results, all_text=all_text,
                fname=fname, fpath=fpath,
                interval=interval, start_str=start_str, end_str=end_str,
                engine_name=engine_name,
            )

    # ── Sonuçları göster ──────────────────────────────────────────────────────
    if "results" not in st.session_state or not st.session_state.results:
        if not analyze_clicked:
            st.info("Ticker girin ve **Analiz Et** butonuna basın.")
        return

    results  = st.session_state.results
    all_text = st.session_state.all_text
    fname    = st.session_state.fname
    fpath    = st.session_state.fpath

    # Hata bildirimleri
    if analyze_clicked:
        errors = {t: e for t, e in (errors if "errors" in dir() else {}).items()}
        for t, e in errors.items():
            st.error(f"❌ {t}: {e}")
        if results:
            st.success(f"✅ {len(results)} ticker analiz edildi — rapor kaydedildi: `{fpath}`")

    # Özet tablo
    st.subheader("Özet")
    summary_rows = []
    for ticker, data in results.items():
        rec    = data["rec"]
        ind    = rec.indicators
        regime = data.get("regime")
        regime_str = (
            f"{REGIME_ICON[regime.label]} {regime.label}" if regime else "—"
        )
        summary_rows.append({
            "Ticker":  ticker,
            "Sinyal":  f"{SIG_ICON[rec.signal.value]} {rec.signal.value}",
            "Güç":     f"{STR_STARS[rec.strength.value]} {rec.strength.value}",
            "Güven":   f"{rec.confidence:.0%}",
            "Rejim":   regime_str,
            "Giriş":   f"{rec.entry_price:.4f}" if rec.entry_price else "—",
        })
    st.dataframe(pd.DataFrame(summary_rows).set_index("Ticker"), width="stretch")

    # Ticker tabları
    st.subheader("Detaylı Analiz")
    tab_labels = [
        f"{SIG_ICON[data['rec'].signal.value]} {ticker}"
        for ticker, data in results.items()
    ]
    tabs = st.tabs(tab_labels)

    for tab, (ticker, data) in zip(tabs, results.items()):
        df  = data["df"]
        rec = data["rec"]
        ind = rec.indicators
        effective_iv = data["interval"]

        with tab:
            regime = data.get("regime")
            # Sinyal banner
            bg = SIG_BG[rec.signal.value]
            fg = SIG_COLOR[rec.signal.value]
            regime_part = (
                f"&nbsp;&nbsp;|&nbsp;&nbsp;Rejim: {REGIME_ICON[regime.label]} {regime.label}"
                if regime else ""
            )
            filter_note = (
                "&nbsp;&nbsp;|&nbsp;&nbsp;⚠️ Rejim filtresi uygulandı"
                if regime and regime.filtered else ""
            )
            st.markdown(
                f'<div class="banner" style="background:{bg};color:{fg};border-color:{fg};">'
                f'{SIG_ICON[rec.signal.value]}&nbsp; {ticker} — {rec.signal.value}'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;{rec.strength.value}'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;Güven: {rec.confidence:.0%}'
                f'{regime_part}{filter_note}'
                f'</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("##### Fiyat Seviyeleri")
                if rec.entry_price:
                    st.metric("Giriş Fiyatı", f"{rec.entry_price:.4f}")
                    st.metric("Stop Loss",   f"{rec.stop_loss:.4f}",
                              delta=f"{rec.stop_loss - rec.entry_price:+.4f}")
                    st.metric("Take Profit", f"{rec.take_profit:.4f}",
                              delta=f"{rec.take_profit - rec.entry_price:+.4f}")
                else:
                    st.info("Nötr sinyal — fiyat seviyesi hesaplanmadı.")

            with c2:
                st.markdown("##### RSI & WaveTrend")
                if ind:
                    st.metric("RSI (14)", f"{ind.rsi:.1f}",
                              help="≤30 aşırı satım · ≥70 aşırı alım")
                    st.metric("WaveTrend WT1", f"{ind.wt1:.2f}")
                    st.metric("WaveTrend WT2", f"{ind.wt2:.2f}")
                    st.caption(f"RSI Bölgesi : **{ind.rsi_zone}**")
                    st.caption(f"WT Bölgesi  : **{ind.wt_zone}**")
                    st.caption(f"WT Kesişim  : **{ind.wt_cross}**")

            with c3:
                st.markdown("##### EMA Seviyeleri")
                if ind:
                    for name, val in [
                        ("EMA 20", ind.ema20), ("EMA 50", ind.ema50),
                        ("EMA 100", ind.ema100), ("EMA 200", ind.ema200),
                    ]:
                        diff_pct = (ind.close - val) / val * 100 if val else 0
                        st.metric(name, f"{val:.4f}", delta=f"{diff_pct:+.2f}%")
                    st.caption(f"Hizalama: **{ind.ema_alignment}**")

            # Mum formasyonları (kural tabanlı) / ML bilgileri
            active_engine = st.session_state.get("engine_name", "Kural Tabanlı")
            if active_engine == "Makine Öğrenmesi":
                st.markdown("##### Model Bilgisi")
                fi  = getattr(rec, "_feature_importances", [])
                tb  = getattr(rec, "_training_bars", "?")
                ld  = getattr(rec, "_label_dist", {})
                src = getattr(rec, "_model_source", "onthefly")

                src_badge = (
                    "🟢 **Pre-trained model**" if src == "pretrained"
                    else "🟡 **Anlık eğitim** (pre-trained model bulunamadı)"
                )
                st.markdown(src_badge)

                tb_str = f"{tb:,}" if isinstance(tb, int) else str(tb)
                st.caption(
                    f"Eğitim: **{tb_str} bar**  |  "
                    f"BUY: {ld.get('BUY',0):,}  "
                    f"SELL: {ld.get('SELL',0):,}  "
                    f"NEUTRAL: {ld.get('NEUTRAL',0):,}"
                )
                if fi:
                    st.markdown("**En etkili özellikler:**")
                    for feat, imp in fi:
                        st.markdown(f"&nbsp;&nbsp;`{feat}` — {imp:.1%}")
            else:
                st.markdown("##### Mum Formasyonları")
                if rec.patterns:
                    for sig_val, label in [("BUY", "Al"), ("SELL", "Sat"), ("NEUTRAL", "Nötr")]:
                        grp = [p for p in rec.patterns if p.signal.value == sig_val]
                        if grp:
                            st.markdown(f"**{SIG_ICON[sig_val]} {label} sinyalli formasyonlar:**")
                            for p in grp:
                                bar_label = "son mum" if p.bars_ago == 0 else f"{p.bars_ago} mum önce"
                                st.markdown(
                                    f"&nbsp;&nbsp;**{p.name}** ({p.strength.value})"
                                    f" · `{bar_label}` — {p.description}"
                                )
                else:
                    st.info("Analiz penceresinde mum formasyonu tespit edilmedi.")

            # Grafik
            st.markdown("##### Grafik")
            st.plotly_chart(
                make_candle_chart(df, ticker, effective_iv, rec),
                width="stretch",
                config={"scrollZoom": True, "displaylogo": False,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
            )

    # İndirme & ham metin
    st.divider()
    dl_col, raw_col = st.columns([1, 3])
    with dl_col:
        st.download_button(
            "⬇️ Raporu İndir (.txt)",
            data=all_text.encode("utf-8"),
            file_name=fname,
            mime="text/plain",
            width="stretch",
        )
    with raw_col:
        with st.expander("📋 Ham Rapor Metni"):
            st.code(all_text, language=None)


if __name__ == "__main__":
    main()
