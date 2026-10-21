import datetime as dt

import ccxt
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Coin Scanner", layout="wide")

# Coins that are never scanned (stablecoins and other non-trading assets)
STABLES = {"USDC", "USDT", "DAI", "TUSD", "FDUSD", "USDD", "PYUSD", "USDP", "EUR",
           "USDG", "USD1", "USDE", "USDS", "RLUSD", "XSOXL"}


def candles(ex, sym, tf, limit):
    raw = ex.fetch_ohlcv(sym, tf, limit=limit)
    return pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vol"])


def find_zones(d):
    """Unmitigated FVG and OB zones as (kind, side, low, high)."""
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    n = len(d)
    avg_range = pd.Series(h - l).rolling(14, min_periods=5).mean().values
    found = []
    for i in range(2, n):
        if l[i] > h[i - 2]:
            found.append(("FVG", "bull", h[i - 2], l[i], i))
        if h[i] < l[i - 2]:
            found.append(("FVG", "bear", h[i], l[i - 2], i))
        p = i - 1
        # displacement candle = body at least 1.5x the average range; OB = candle before it
        if pd.notna(avg_range[p]) and abs(c[i] - o[i]) >= 1.5 * avg_range[p]:
            if c[i] > o[i] and c[p] < o[p]:
                found.append(("OB", "bull", l[p], h[p], i))
            elif c[i] < o[i] and c[p] > o[p]:
                found.append(("OB", "bear", l[p], h[p], i))
    live = []
    for kind, side, lo, hi, i in found:
        later = c[i + 1:]
        if side == "bull" and (later < lo).any():
            continue
        if side == "bear" and (later > hi).any():
            continue
        live.append((kind, side, float(lo), float(hi)))
    return live


def daily_bias(d):
    """d = daily candles including the current (unfinished) one."""
    closed = d.iloc[:-1].reset_index(drop=True)
    # Rule 1: last closed day vs the day before it
    bias = "bull" if closed.close.iloc[-1] > closed.close.iloc[-2] else "bear"
    # Rule 2: rejection wick at a key level (FVG / OB) flips the bias
    zones = find_zones(closed.iloc[:-1])
    k = closed.iloc[-1]
    o, h, l, c = k.open, k.high, k.low, k.close
    body, rg = abs(c - o), h - l
    low_wick, up_wick = min(o, c) - l, h - max(o, c)
    if rg > 0:
        if low_wick >= 2 * body and low_wick >= 0.5 * rg:
            for kind, side, lo, hi in zones:
                if side == "bull" and l <= hi and min(o, c) >= lo:
                    return "bull", f"Rule 2: rejection at bull {kind}"
        if up_wick >= 2 * body and up_wick >= 0.5 * rg:
            for kind, side, lo, hi in zones:
                if side == "bear" and h >= lo and max(o, c) <= hi:
                    return "bear", f"Rule 2: rejection at bear {kind}"
    return bias, "Rule 1"


def turtle_soup(d, bias, look=20, recent=3):
    h, l, c = d.high.values, d.low.values, d.close.values
    n = len(d)
    for k in range(n - recent, n):
        if k - look < 0:
            continue
        if bias == "bull":
            ref = l[k - look:k].min()
            if l[k] < ref and c[k] > ref:
                return True
        else:
            ref = h[k - look:k].max()
            if h[k] > ref and c[k] < ref:
                return True
    return False


def cisd(d, bias, recent=3, scan=12):
    o, c = d.open.values, d.close.values
    n = len(d)
    opposing = (c < o) if bias == "bull" else (c > o)
    for e in range(n - 2, max(n - scan, 1) - 1, -1):
        if opposing[e] and not opposing[e + 1]:
            s = e
            while s > 0 and opposing[s - 1]:
                s -= 1
            run_open = o[s]
            for j in range(e + 1, n):
                broke = c[j] > run_open if bias == "bull" else c[j] < run_open
                if broke:
                    return j >= n - recent
            return False
    return False


def dxy_bias():
    try:
        d = yf.download("DX-Y.NYB", period="3mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.dropna(subset=["Close"])
        if d.index[-1].date() >= dt.datetime.now(dt.timezone.utc).date():
            d = d.iloc[:-1]
        return "bull" if d["Close"].iloc[-1] > d["Close"].iloc[-2] else "bear"
    except Exception:
        return None


def label(b):
    return {"bull": "Bullish", "bear": "Bearish"}.get(b, "unavailable")


st.title("Coin Scanner")

c1, c2, c3 = st.columns(3)
EXCHANGES = {
    "Binance Futures (USDT-M)": "binanceusdm",
    "OKX Perpetual (USDT)": "okx",
    "Bybit Perpetual (USDT)": "bybit",
}
exch_name = c1.selectbox("Exchange (data source)", list(EXCHANGES))
exch = EXCHANGES[exch_name]
mode = c2.radio("Higher timeframe", ["4H (entry 15m)", "1H (entry 5m)"], horizontal=True)
top_n = c3.slider("Coins to scan (top by volume)", 20, 150, 50, 10)

c4, c5, c6 = st.columns(3)
prox = c4.slider("Max distance to OB zone (%)", 0.5, 10.0, 3.0, 0.5)
min_score = c5.slider("Minimum score", 1, 8, 2)
use_dxy = c6.checkbox("Use DXY filter", True)
news = c6.checkbox("High-impact news today (stricter)", False)

if st.button("Scan", type="primary"):
    htf = "4h" if mode.startswith("4H") else "1h"
    entry = "15m" if htf == "4h" else "5m"
    try:
        cfg = {"enableRateLimit": True}
        if exch in ("okx", "bybit"):
            cfg["options"] = {"defaultType": "swap", "defaultSubType": "linear"}
        ex = getattr(ccxt, exch)(cfg)
        ex.load_markets()
        tick = ex.fetch_tickers()
    except Exception as e:
        st.error(f"Could not get data from {exch_name}: {e}. Try another exchange.")
        st.stop()

    pairs = []
    for s, m in ex.markets.items():
        if (m.get("swap") and m.get("linear") and m.get("quote") == "USDT"
                and m.get("settle") == "USDT" and m.get("active") is not False
                and m.get("base") not in STABLES and s in tick):
            pairs.append((float(tick[s].get("quoteVolume") or 0), s))
    syms = [s for _, s in sorted(pairs, reverse=True)[:top_n]]
    if not syms:
        st.error("No USDT pairs found on this exchange. Try another one.")
        st.stop()

    dxy = dxy_bias()
    rows, fails = [], 0
    bar = st.progress(0.0)
    for i, s in enumerate(syms):
        bar.progress((i + 1) / len(syms), text=s)
        try:
            dd = candles(ex, s, "1d", 60)
            if len(dd) < 25:
                continue
            # skip pegged / stable-like assets (price barely moves over 60 days)
            if (dd.high.max() - dd.low.min()) / dd.close.iloc[-1] < 0.05:
                continue
            bias, why = daily_bias(dd)
            hd = candles(ex, s, htf, 120)
            if len(hd) < 40:
                continue
            closed = hd.iloc[:-1].reset_index(drop=True)
            price = float(hd.close.iloc[-1])
            sigs, score = [], 0
            if turtle_soup(closed, bias):
                sigs.append("Turtle soup")
                score += 2
            if cisd(closed, bias):
                sigs.append("CISD")
                score += 2
            best = None
            for kind, side, lo, hi in find_zones(closed.tail(100)):
                if kind != "OB" or side != bias:
                    continue
                if lo <= price <= hi:
                    dist = 0.0
                elif price > hi:
                    dist = (price - hi) / price * 100
                else:
                    dist = (lo - price) / price * 100
                if dist <= prox and (best is None or dist < best[0]):
                    best = (dist, lo, hi)
            if best:
                sigs.append("OB")
                score += 3 if best[0] == 0 else 2
            if sigs:
                adj = 0 if dxy is None else (1 if dxy != bias else -1)
                rows.append({
                    "Coin": ex.markets[s]["base"],
                    "Bias": "🟢 Bullish" if bias == "bull" else "🔴 Bearish",
                    "Bias basis": why,
                    "Signals": ", ".join(sigs),
                    "OB zone": f"{best[1]:.6g} – {best[2]:.6g}" if best else "-",
                    "Dist %": round(best[0], 2) if best else None,
                    "Price": price,
                    "Entry TF": entry,
                    "base": score,
                    "dxy_adj": adj,
                })
        except Exception:
            fails += 1
    bar.empty()
    st.session_state["res"] = {
        "rows": rows, "fails": fails, "n": len(syms), "exch": exch_name, "htf": htf.upper(),
        "dxy": dxy, "time": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
    }

res = st.session_state.get("res")
if res:
    st.caption(
        f"Scanned {res['n']} coins on {res['exch']} at {res['time']} UTC | HTF {res['htf']} | "
        f"DXY daily bias: {label(res['dxy'])} | failed: {res['fails']}"
    )
    if news:
        st.warning("News risk is on: the minimum score is raised by 1. Check the calendar before entering.")
    need = min_score + (1 if news else 0)
    df = pd.DataFrame(res["rows"])
    if df.empty:
        st.info("No coins showed a signal in this scan.")
    else:
        df["Score"] = df["base"] + (df["dxy_adj"] if use_dxy else 0)
        df = df[df["Score"] >= need].sort_values(["Score", "Dist %"], ascending=[False, True])
        cols = ["Coin", "Score", "Bias", "Bias basis", "Signals", "OB zone", "Dist %", "Price", "Entry TF"]
        if df.empty:
            st.info("No coins reached the minimum score. Lower it or rescan later.")
        else:
            st.dataframe(df[cols], hide_index=True)

st.caption(
    "Candidates only, not trade advice. Confirm on the chart, find your entry on the entry timeframe, "
    "and backtest the rules before trading real money."
)
