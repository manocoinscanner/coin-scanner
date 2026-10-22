import datetime as dt
import re

import ccxt
import requests
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


@st.cache_data(ttl=600, show_spinner=False)
def dxy_info():
    """Dollar index daily bias (Rule 1) and last closed-day % change."""
    try:
        d = yf.download("DX-Y.NYB", period="3mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.dropna(subset=["Close"])
        if d.index[-1].date() >= dt.datetime.now(dt.timezone.utc).date():
            d = d.iloc[:-1]
        last, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
        return {"bias": "bull" if last > prev else "bear", "pct": (last / prev - 1) * 100}
    except Exception:
        return None


def dxy_bias():
    info = dxy_info()
    return info["bias"] if info else None


@st.cache_data(ttl=300, show_spinner=False)
def btc_overview(exch, htf):
    """BTC daily bias plus higher-timeframe signals in the bias direction."""
    cfg = {"enableRateLimit": True}
    if exch in ("okx", "bybit"):
        cfg["options"] = {"defaultType": "swap", "defaultSubType": "linear"}
    ex = getattr(ccxt, exch)(cfg)
    sym = "BTC/USDT:USDT"
    dd = candles(ex, sym, "1d", 60)
    hd = candles(ex, sym, htf, 120)
    bias, why = daily_bias(dd)
    done = dd.iloc[:-1]
    day_pct = (done.close.iloc[-1] / done.close.iloc[-2] - 1) * 100
    price = float(hd.close.iloc[-1])
    today_pct = (price / float(dd.open.iloc[-1]) - 1) * 100
    closed = hd.iloc[:-1].reset_index(drop=True)
    sigs = []
    if turtle_soup(closed, bias):
        sigs.append("Turtle soup")
    if cisd(closed, bias):
        sigs.append("CISD")
    for kind, side, lo, hi in find_zones(closed.tail(100)):
        if kind == "OB" and side == bias and lo * 0.97 <= price <= hi * 1.03:
            sigs.append(f"OB {lo:.0f}-{hi:.0f}")
            break
    return {"bias": bias, "why": why, "day_pct": day_pct, "today_pct": today_pct,
            "price": price, "sigs": sigs}


def norm(base):
    """1000PEPE -> PEPE, 1MBABYDOGE -> BABYDOGE, so names match across exchanges."""
    return re.sub(r"^(1000000|1000|1M)", "", base.upper())


@st.cache_data(ttl=21600, show_spinner=False)
def binance_futures_bases():
    """Coins with a Binance USDT perpetual. Binance's own API is blocked from this server,
    so the list comes from CoinGecko (cached for 6 hours)."""
    r = requests.get(
        "https://api.coingecko.com/api/v3/derivatives/exchanges/binance_futures",
        params={"include_tickers": "unexpired"},
        headers={"accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    out = set()
    for t in r.json().get("tickers", []):
        if t.get("target") == "USDT" and t.get("contract_type", "perpetual") == "perpetual":
            out.add(norm(t["base"]))
    return sorted(out)


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
min_vol = c4.slider("Min 24h volume (million USD)", 1, 500, 20)
min_score = c5.slider("Minimum score", 1, 8, 3)
use_dxy = c6.checkbox("Use DXY filter", True)
only_bn = c6.checkbox("Only coins also on Binance Futures", True)
only_pd = c6.checkbox("Only OBs on the correct side (bull OB in Discount, bear OB in Premium)", False)
news = c6.checkbox("High-impact news today (stricter)", False)

htf = "4h" if mode.startswith("4H") else "1h"

st.subheader("Market overview: BTC + Dollar index")
o1, o2, o3 = st.columns(3)
btc = None
dxy_i = dxy_info()
try:
    btc = btc_overview(exch, htf)
except Exception as e:
    o1.warning(f"BTC overview not available from {exch_name}: {e}")
if btc:
    o1.metric("BTC daily bias", "🟢 Bullish" if btc["bias"] == "bull" else "🔴 Bearish",
              f"{btc['day_pct']:+.2f}% last closed day")
    o1.caption(
        f"{btc['why']} | Price {btc['price']:,.1f} | today so far {btc['today_pct']:+.2f}%. "
        f"{htf.upper()} signals in the bias direction: {', '.join(btc['sigs']) or 'none right now'}."
    )
if dxy_i:
    o2.metric("Dollar index (DXY) daily bias",
              "🟢 Bullish ↑" if dxy_i["bias"] == "bull" else "🔴 Bearish ↓",
              f"{dxy_i['pct']:+.2f}% last closed day", delta_color="inverse")
    o2.caption("DXY ඉහළ ගියාම crypto වලට පීඩනයක්, පහළ ගියාම සහායක් වෙන්න සාමාන්‍යයෙන් ඉඩ තියෙනවා (හැමවිටම නෙවෙයි).")
else:
    o2.warning("DXY data is not available right now.")
if btc and dxy_i:
    if btc["bias"] == "bull" and dxy_i["bias"] == "bear":
        o3.success("BTC ↑ සහ DXY ↓: දෙකම crypto වලට හිතකරයි. Long setups වලට වාසිදායක තත්ත්වයක්.")
    elif btc["bias"] == "bear" and dxy_i["bias"] == "bull":
        o3.warning("BTC ↓ සහ DXY ↑: දෙකම crypto වලට අහිතකරයි. Short setups වලට වාසිදායක තත්ත්වයක්, long වලදී ප්‍රවේශම් වන්න.")
    else:
        o3.info("BTC සහ DXY සංඥා මිශ්‍රයි (එකිනෙකට ගැළපෙන්නේ නැහැ). ප්‍රවේශම් වන්න, ලකුණු අඩු coins මගහරින්න.")

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

    bn = None
    if only_bn and exch != "binanceusdm":
        try:
            bn = set(binance_futures_bases())
        except Exception as e:
            st.error(
                f"Could not get the Binance Futures coin list: {e}. "
                "Wait a minute and scan again, or untick the Binance filter."
            )
            st.stop()

    pairs = []
    for s, m in ex.markets.items():
        if (m.get("swap") and m.get("linear") and m.get("quote") == "USDT"
                and m.get("settle") == "USDT" and m.get("active") is not False
                and m.get("base") not in STABLES and s in tick):
            if bn is not None and norm(m["base"]) not in bn:
                continue
            t = tick[s]
            vol = t.get("quoteVolume")
            if not vol and t.get("baseVolume") and t.get("last"):
                vol = t["baseVolume"] * t["last"]
            pairs.append((float(vol or 0), s))
    if any(v > 0 for v, _ in pairs):  # volume data available: keep liquid coins only
        pairs = [p for p in pairs if p[0] >= min_vol * 1_000_000]
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
            # nearest unmitigated OB (bullish OR bearish) around the current price
            zones = find_zones(closed.tail(100))
            best = None
            for kind, side, lo, hi in zones:
                if kind != "OB":
                    continue
                if (side == "bull" and price < lo) or (side == "bear" and price > hi):
                    continue  # price is already beyond the zone
                if lo <= price <= hi:
                    dist = 0.0
                elif price > hi:
                    dist = (price - hi) / price * 100
                else:
                    dist = (lo - price) / price * 100
                key = (dist, 0 if side == bias else 1)
                if dist <= prox and (best is None or key < best[0]):
                    best = (key, side, lo, hi)
            if best:  # OB is mandatory; the rest are extra confirmations that add score
                (dist, _), ob_side, lo, hi = best
                score += 3 if dist == 0 else 2
                if turtle_soup(closed, ob_side):
                    sigs.append("Turtle soup")
                    score += 2
                if cisd(closed, ob_side):
                    sigs.append("CISD")
                    score += 2
                for kind, side, flo, fhi in zones:  # same-side FVG overlapping/near the OB
                    if kind != "FVG" or side != ob_side:
                        continue
                    gap = 0.0 if (fhi >= lo and flo <= hi) else (flo - hi if flo > hi else lo - fhi)
                    if gap / price * 100 <= 1.0:
                        sigs.append("FVG near OB")
                        score += 1
                        break
                # premium / discount: OB midpoint vs the high-low range of the last 100 closed HTF candles
                rng_hi = float(closed.high.tail(100).max())
                rng_lo = float(closed.low.tail(100).min())
                pos = (((lo + hi) / 2 - rng_lo) / (rng_hi - rng_lo) * 100) if rng_hi > rng_lo else 50.0
                pd_zone = "Premium" if pos > 50 else "Discount"
                pd_ok = (ob_side == "bull" and pos <= 50) or (ob_side == "bear" and pos > 50)
                adj = 0 if dxy is None else (1 if dxy != ob_side else -1)
                rows.append({
                    "Coin": ex.markets[s]["base"],
                    "Daily bias": "🟢 Bullish" if bias == "bull" else "🔴 Bearish",
                    "Bias basis": why,
                    "OB type": "🟢 Bullish OB" if ob_side == "bull" else "🔴 Bearish OB",
                    "OB TF": htf.upper(),
                    "Premium/Discount": f"{pd_zone} ({pos:.0f}%)",
                    "PD fit": "✅ Correct side" if pd_ok else "⚠️ Wrong side",
                    "Setup": "With bias" if ob_side == bias else "Pullback (against bias)",
                    "Confirmations": ", ".join(sigs) or "-",
                    "OB zone": f"{lo:.6g} – {hi:.6g}",
                    "Dist %": round(dist, 2),
                    "Price": price,
                    "Entry TF": entry,
                    "base": score,
                    "dxy_adj": adj,
                    "pd_ok": pd_ok,
                })
        except Exception:
            fails += 1
    bar.empty()
    st.session_state["res"] = {
        "rows": rows, "fails": fails, "n": len(syms), "exch": exch_name, "htf": htf.upper(),
        "bn": bn is not None,
        "dxy": dxy, "time": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
    }

res = st.session_state.get("res")
if res:
    st.caption(
        f"Scanned {res['n']} coins on {res['exch']} at {res['time']} UTC | HTF {res['htf']} | "
        f"DXY daily bias: {label(res['dxy'])} | Binance Futures filter: {'on' if res['bn'] else 'off'} | "
        f"failed: {res['fails']}"
    )
    if news:
        st.warning("News risk is on: the minimum score is raised by 1. Check the calendar before entering.")
    need = min_score + (1 if news else 0)
    df = pd.DataFrame(res["rows"])
    if df.empty:
        st.info("No coins had an OB near the price in this scan. Try a larger OB distance or a lower minimum volume.")
    else:
        df["Score"] = df["base"] + (df["dxy_adj"] if use_dxy else 0)
        if only_pd:
            df = df[df["pd_ok"]]
        df = df[df["Score"] >= need].sort_values(["Score", "Dist %"], ascending=[False, True])
        cols = ["Coin", "Score", "Daily bias", "Bias basis", "OB type", "OB TF", "Premium/Discount",
                "PD fit", "Setup", "Confirmations", "OB zone", "Dist %", "Price", "Entry TF"]
        if df.empty:
            st.info("No coins reached the minimum score. Lower it or rescan later.")
        else:
            st.dataframe(df[cols], hide_index=True)

with st.expander("📖 Terms explained (අර්ථ විස්තර)"):
    st.markdown(
        """
### 1. Daily Bias (දවසේ දිශාව)
- **Bias**: Daily chart එකෙන් තීරණය කරන දිශාව. **Bullish 🟢** = ඉහළ යන්න වැඩි ඉඩක් (long setups හොයන්න), **Bearish 🔴** = පහළ යන්න වැඩි ඉඩක් (short setups හොයන්න). මේක අනාවැකියක් නෙවෙයි, ඔයාගේ rules වලට අනුව හදන දිශා නිගමනයක් විතරයි.
- **Rule 1 (close එකට close)**: අන්තිමට වැසුණු daily candle එක, ඊට කලින් දවසේ close එකට වඩා ඉහළින් close වුණා නම් Bullish, පහළින් close වුණා නම් Bearish.
- **Rule 2 (key level rejection)**: අන්තිම daily candle එකේ දිග wick එකක් (wick එක body එකට වඩා දෙගුණයක් හෝ වැඩි, candle එකෙන් අඩකට වඩා) key level එකක (FVG හෝ OB) ළඟ හැදුණා නම්, එතැනින් market එක හැරුණා කියලා සලකලා Bias එක ඒ දිශාවට වෙනස් කරනවා (Rule 1 අභිබවා). පහළ wick + bullish zone = Bullish, ඉහළ wick + bearish zone = Bearish.
- **Bias basis** (table එකේ column එක): Bias එක ආවේ Rule 1 එකෙන්ද Rule 2 එකෙන්ද කියලා පෙන්නනවා.
- **Key level**: මිල කලින් ප්‍රතිචාර දක්වපු වැදගත් මිල කලාපයක්. මෙහිදී FVG සහ OB.
- **FVG (Fair Value Gap)**: candle 3ක් අතරේ, මැද candle එක වේගයෙන් ගිය නිසා පළමු candle එකේ high එකයි තුන්වැනි candle එකේ low එකයි අතර හිස් ඉඩක් ඉතිරි වීම. මිල ආපහු ඒක පුරවන්න එන්න ඉඩ තියෙනවා කියලා සලකනවා. Candle එකක් ඒ කලාපය හරහා close වුණාම ඒක අවලංගුයි.

### 2. Higher timeframe (HTF) සංඥා
- **HTF**: සංඥා හොයන chart එක, 4H හෝ 1H. **Entry TF**: ඔයා chart එකේ බලලා entry ගන්න timeframe එක (4H නම් 15m, 1H නම් 5m). Scanner එක Entry TF එක විශ්ලේෂණය කරන්නේ නැහැ.
- **OB (Order Block)**: ශක්තිමත් වේගවත් ගමනකට (සාමාන්‍ය candle ප්‍රමාණයේ 1.5 ගුණයකට වඩා විශාල candle එකකට) කලින් තිබුණු අන්තිම ප්‍රතිවිරුද්ධ candle එක. Bullish OB = ඉහළ යන වේගවත් ගමනකට කලින් තිබුණු අන්තිම bearish candle එක. Candle එකක් ඒ කලාපය හරහා close වුණාම අවලංගුයි. Bearish OB = පහළ යන වේගවත් ගමනකට කලින් තිබුණු අන්තිම bullish (ඉහළ) candle එක. Scanner එක බලන්නේ Bullish OB හෝ Bearish OB එකක් (Daily bias එකට ගැළපුණත් නැතත්) මිලට ළඟද කියලායි. දෙකම වෙන වෙනම පෙන්නනවා: **Daily bias** column එකේ දවසේ දිශාව, **OB type** column එකේ OB එකේ වර්ගය. Daily bias එක Bearish වෙලා Bullish OB එකක් ළඟ ඉන්නවා නම් ඒක **Pullback (against bias)** setup එකක්. OB එක bias එකටම ගැළපෙනවා නම් **With bias**.
- **OB zone**: OB candle එකේ low සිට high දක්වා මිල කලාපය. **Dist %**: දැන් මිල ඒ කලාපයට ඇති දුර (%). `0` = මිල කලාපය ඇතුළේ. `None` හෝ `-` = ළඟ OB එකක් නැහැ.
- **Premium / Discount**: Dealing range එකක් (මෙහිදී HTF එකේ අන්තිම candles 100ක highest high සිට lowest low දක්වා) මැද 50% ලකුණෙන් (equilibrium) දෙකට බෙදනවා. 50%ට **පහළ** කොටස **Discount** (මිල අඩු කලාපය), 50%ට **ඉහළ** කොටස **Premium** (මිල වැඩි කලාපය). **Premium/Discount** column එකේ පෙන්නන්නේ OB zone එකේ මැද ඒ range එකේ කොයි තැනද කියන එකයි (0% = range එකේ පහළම, 100% = ඉහළම).
- **PD fit**: Bullish OB එකක් **Discount** එකේ, Bearish OB එකක් **Premium** එකේ තියෙනවා නම් ✅ Correct side. Bullish OB එකක් Premium එකේ හෝ Bearish OB එකක් Discount එකේ තියෙනවා නම් ⚠️ Wrong side. ICT සංකල්පයට අනුව ඒ වගේ OB එකක් අවලංගු වෙන්න (කැඩෙන්න) වැඩි ඉඩක් තියෙනවා. "Only OBs on the correct side" ටික් කළොත් ⚠️ ඒවා ලැයිස්තුවෙන් අයින් වෙනවා.
- **OB TF**: ඒ OB එක හමුවුණු timeframe එක. දැනට OB හොයන්නේ ඔයා තෝරපු Higher timeframe (4H හෝ 1H) එකේ විතරයි.
- **Turtle soup**: Bullish නම්, මෑත candles 20ක අඩුම low එක මිල කඩලා (liquidity sweep), ඒත් candle එක ආපහු ඒ මට්ටමට උඩින් close වුණා නම්. Bearish නම් ඒකේ ප්‍රතිවිරුද්ධය (high එක). අන්තිම candles 3 ඇතුළත විතරයි බලන්නේ.
- **CISD (Change in State of Delivery)**: මෙහි සරල අර්ථ දැක්වීම: Bullish සඳහා, අඛණ්ඩ bearish candles පෙළක් ඇවිත් ඊට පස්සේ candle එකක් ඒ පෙළේ පළමු candle එකේ open එකට උඩින් close වුණා නම්. (Bearish සඳහා ප්‍රතිවිරුද්ධය.) අන්තිම candles 3 ඇතුළත විතරයි බලන්නේ.
- **Confirmations**: OB එක අනිවාර්යයි (මිලට ළඟ, Max distance ඇතුළත OB එකක් නැති coins ලැයිස්තුවට එන්නේ නැහැ). ඒක ඇත්නම් අමතර තහවුරු කිරීම් විදිහට පෙන්නන්නේ: Turtle soup, CISD, සහ **FVG near OB** (OB එකේ දිශාවටම ගැළපෙන FVG එකක් OB කලාපය ඇතුළේ හෝ ඊට මිලෙන් 1% ඇතුළත තියෙනවා නම්). Turtle soup සහ CISD බලන්නේ **OB එකේ දිශාවටයි** (Bullish OB එකකට bullish ඒවා, Bearish OB එකකට bearish ඒවා).

### 3. Score (ලකුණු)
- **Score**: OB (zone එකට ළඟ) +2, මිල zone එක ඇතුළේ නම් +3. Turtle soup +2, CISD +2, FVG near OB +1. DXY filter එක on නම්: OB එකේ දිශාවට (Bullish OB = long, Bearish OB = short) DXY ගැළපෙනවා නම් (උදා: Bullish OB, DXY Bearish) +1, ගැළපෙන්නේ නැත්නම් -1. ලකුණු වැඩි = ඔයාගේ rules වලට වැඩියෙන් ගැළපෙනවා. ඒත් ජයග්‍රහණයක් සහතික කරන්නේ නැහැ. With bias සහ Pullback setups වලට වෙනම ලකුණු නැහැ, ඒවා Setup column එකෙන් වෙන්කරලා බලන්න.
- **Minimum score**: මේ ලකුණට වඩා අඩු coins ලැයිස්තුවෙන් අයින් කරනවා. "High-impact news today" ටික් කළොත් අවශ්‍ය අවම ලකුණ 1කින් වැඩි වෙනවා.

### 4. Settings (පාලන)
- **Exchange (data source)**: දත්ත ගන්න තැන. Binance Futures server එක block වෙන්න පුළුවන්. OKX Perpetual වැඩ කරනවා.
- **Higher timeframe**: HTF එක 4H ද 1H ද කියලා තෝරනවා.
- **Coins to scan (top by volume)**: volume වැඩිම coins කීයක් scan කරනවද කියන ගණන.
- **Min 24h volume (million USD)**: පසුගිය පැය 24 තුළ ඒ coin එක ඩොලර් කීයක් වටිනාකමට trade වුණාද (මිලියන වලින්). වැඩි = trade කරන අය ගොඩක්, ලේසියෙන් මිලදී ගන්න/විකුණන්න පුළුවන් (හොඳ liquidity). අඩු = trade කරන අය ටික දෙනෙක්, මිල ඉක්මනින් අස්ථාවර වෙනවා, fake moves වැඩියි. මේ අගය OKX දත්ත වලින් එන නිසා Binance volume එකට වඩා වෙනස් වෙන්න පුළුවන්.
- **Max distance to OB zone (%)**: මිල OB කලාපයෙන් මීට වඩා දුරින් තියෙනවා නම් ඒ OB එක සංඥාවක් විදිහට ගණන් ගන්නේ නැහැ.
- **Use DXY filter**: Dollar index (DXY) එකේ bias එක coins වල ලකුණු වලට එකතු කරනවා/අඩු කරනවා.
- **Only coins also on Binance Futures**: OKX එකේත් Binance Futures එකේත් තියෙන coins විතරක් scan කරනවා.
- **High-impact news today**: අද CPI, Fed වගේ ලොකු news තියෙනවා නම් ටික් කරන්න. අවශ්‍ය අවම ලකුණ 1කින් වැඩි වෙනවා.
- **Price**: coin එකේ දැන් මිල (තෝරපු exchange එකේ).

### 5. Market overview (පිටුවේ උඩ කොටස)
- **BTC daily bias**: BTC එකටත් එකම Rule 1 සහ Rule 2 යොදලා දවසේ දිශාව පෙන්නනවා. යටින් පෙන්නන්නේ අන්තිම වැසුණු දවසේ % වෙනස, දැන් මිල, අද දවසේ වෙනස, HTF එකේ Bias දිශාවට හමුවෙන සංඥා.
- **Dollar index (DXY)**: US ඩොලරයේ ශක්තිය මනින දර්ශකය. සාමාන්‍යයෙන් DXY ඉහළ යනකොට crypto පහළ යන්නත්, DXY පහළ යනකොට crypto ඉහළ යන්නත් ඉඩ වැඩියි. ඒත් ඒක හැමවිටම සත්‍ය නෙවෙයි.
- **පණිවිඩය**: BTC සහ DXY එකිනෙකට ගැළපෙනවද (BTC ↑ + DXY ↓, හෝ BTC ↓ + DXY ↑) කියලා කෙටියෙන් කියනවා. ගැළපෙන්නේ නැත්නම් සංඥා මිශ්‍රයි කියලා ප්‍රවේශම් කරනවා.

### අවවාදය
මේ rules ඔයාගේ strategy එකේ සරල කරපු අර්ථ දැක්වීම්. ඔයාගේ chart එකේ දකින දේට වෙනස් වෙන්න පුළුවන්. Scanner එක candidates ලැයිස්තුවක් විතරයි, trade advice නෙවෙයි. Chart එකේ තහවුරු කරලා, backtest කරලා පාවිච්චි කරන්න.
"""
    )

st.caption(
    "Candidates only, not trade advice. Confirm on the chart, find your entry on the entry timeframe, "
    "and backtest the rules before trading real money."
)
