import datetime as dt
import re
import time

import ccxt
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Coin Scanner", layout="wide")

# Coins that are never scanned (stablecoins and other non-trading assets)
STABLES = {"USDC", "USDT", "DAI", "TUSD", "FDUSD", "USDD", "PYUSD", "USDP", "EUR",
           "USDG", "USD1", "USDE", "USDS", "RLUSD", "XSOXL"}
TF_MAP = {"4H": "4h", "1H": "1h", "15m": "15m", "5m": "5m"}
MAX_AGE = 150  # an OB counts only if its structure break happened within the last 150 closed candles


def candles(ex, sym, tf, limit):
    raw = ex.fetch_ohlcv(sym, tf, limit=limit)
    return pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vol"])


class BinanceSpot:
    """Minimal Binance Spot market-data client (public data-api.binance.vision, no key needed).
    It offers the few calls this app uses, so it can be used in place of a ccxt exchange."""
    BASE = "https://data-api.binance.vision/api/v3"

    def __init__(self):
        self.markets = {}
        self._tick = {}

    def _get(self, path, params=None):
        r = requests.get(self.BASE + path, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def fetch_tickers(self):
        if self._tick:
            return self._tick
        for t in self._get("/ticker/24hr"):
            sym = t["symbol"]
            if not sym.endswith("USDT"):
                continue
            base = sym[:-4]
            # same keys the OKX filter in the scanner looks for
            self.markets[sym] = {"swap": True, "linear": True, "quote": "USDT", "settle": "USDT",
                                 "active": True, "base": base}
            self._tick[sym] = {"quoteVolume": float(t["quoteVolume"]), "baseVolume": float(t["volume"]),
                               "last": float(t["lastPrice"])}
        return self._tick

    def load_markets(self):
        if not self.markets:
            self.fetch_tickers()
        return self.markets

    def fetch_ohlcv(self, sym, tf, limit=300):
        time.sleep(0.05)  # stay well under Binance's request limits
        rows = self._get("/klines", {"symbol": sym, "interval": tf, "limit": limit})
        return [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows]


def make_ex(exch):
    """OKX perpetuals through ccxt, or Binance Spot through the small client above."""
    if exch == "binancespot":
        return BinanceSpot()
    return getattr(ccxt, exch)({"enableRateLimit": True,
                                "options": {"defaultType": "swap", "defaultSubType": "linear"}})


def find_coin(exch, coin, fallback):
    """Find a coin on the chosen exchange; if it is missing on OKX and fallback is on,
    look for it on Binance Spot. Returns (exchange object, symbol, source key) or (None, None, None)."""
    key = norm(coin)
    try:
        ex = make_ex(exch)
        ex.load_markets()
        for s, m in ex.markets.items():
            if (m.get("swap") and m.get("linear") and m.get("quote") == "USDT"
                    and m.get("settle") == "USDT" and m.get("active") is not False
                    and norm(m["base"]) == key):
                return ex, s, exch
    except Exception:
        pass
    if exch == "okx" and fallback:
        try:
            sp = BinanceSpot()
            sp.load_markets()
            for s, m in sp.markets.items():
                if norm(m["base"]) == key:
                    return sp, s, "binancespot"
        except Exception:
            pass
    return None, None, None


def swings(d, k=2):
    """Swing highs/lows as (index, 'H' or 'L', price), strictly alternating."""
    h, l = d.high.values, d.low.values
    n = len(d)
    raw = []
    for i in range(k, n - k):
        is_h = h[i] >= h[i - k:i].max() and h[i] > h[i + 1:i + k + 1].max()
        is_l = l[i] <= l[i - k:i].min() and l[i] < l[i + 1:i + k + 1].min()
        if is_h == is_l:
            continue  # neither, or an outside candle that is both
        raw.append((i, "H" if is_h else "L", float(h[i] if is_h else l[i])))
    out = []
    for i, kind, price in raw:
        if out and out[-1][1] == kind:
            if (kind == "H" and price > out[-1][2]) or (kind == "L" and price < out[-1][2]):
                out[-1] = (i, kind, price)
            continue
        out.append((i, kind, price))
    return out


def find_structure_obs(d, k=2):
    """OBs from market structure, as (side, low, high, break_index).
    Bullish: down-trend (lower high + lower low). The last lower low must be a liquidity sweep
             ("torch"): its candle wicks below the previous swing low but CLOSES back above it.
             Then a candle closes above the last lower high (structure break).
             Zone = high of the last lower high down to the low of the last lower low.
    Bearish: the mirror image. The last higher high must sweep the previous swing high
             (wick above it, close back below it), then a candle closes below the last higher low.
    A zone dies when a later candle closes beyond it."""
    h, l, c = d.high.values, d.low.values, d.close.values
    n = len(d)
    sw = swings(d, k)
    found = []
    for t in range(3, len(sw)):
        s0, s1, s2, s3 = sw[t - 3], sw[t - 2], sw[t - 1], sw[t]
        kinds = (s0[1], s1[1], s2[1], s3[1])
        if kinds == ("H", "L", "H", "L") and s2[2] < s0[2] and s3[2] < s1[2]:
            # liquidity sweep: the last lower low took out the previous low and closed back above it
            if not (l[s3[0]] < s1[2] and c[s3[0]] > s1[2]):
                continue
            top, bot = s2[2], s3[2]  # last lower high, last lower low
            for j in range(s3[0] + 1, n):
                if l[j] < bot:
                    break  # a deeper low came first, so this is not the last lower low
                if c[j] > top:
                    found.append(("bull", bot, top, j))
                    break
        elif kinds == ("L", "H", "L", "H") and s2[2] > s0[2] and s3[2] > s1[2]:
            # liquidity sweep: the last higher high took out the previous high and closed back below it
            if not (h[s3[0]] > s1[2] and c[s3[0]] < s1[2]):
                continue
            bot, top = s2[2], s3[2]  # last higher low, last higher high
            for j in range(s3[0] + 1, n):
                if h[j] > top:
                    break
                if c[j] < bot:
                    found.append(("bear", bot, top, j))
                    break
    live = []
    for side, lo, hi, j in found:
        later = c[j + 1:]
        if side == "bull" and (later < lo).any():
            continue
        if side == "bear" and (later > hi).any():
            continue
        live.append((side, float(lo), float(hi), j))
    return live


def find_fvgs(d):
    """Unfilled fair value gaps as (side, low, high)."""
    h, l, c = d.high.values, d.low.values, d.close.values
    n = len(d)
    found = []
    for i in range(2, n):
        if l[i] > h[i - 2]:
            found.append(("bull", h[i - 2], l[i], i))
        if h[i] < l[i - 2]:
            found.append(("bear", h[i], l[i - 2], i))
    live = []
    for side, lo, hi, i in found:
        later = c[i + 1:]
        if side == "bull" and (later < lo).any():
            continue
        if side == "bear" and (later > hi).any():
            continue
        live.append((side, float(lo), float(hi)))
    return live


def key_zones(d, k=2):
    """Daily key levels for Rule 2: FVGs and structure OBs as (kind, side, low, high)."""
    z = [("FVG", s, lo, hi) for s, lo, hi in find_fvgs(d)]
    z += [("OB", s, lo, hi) for s, lo, hi, _ in find_structure_obs(d, k)]
    return z


def daily_bias(d):
    """d = daily candles including the current (unfinished) one."""
    closed = d.iloc[:-1].reset_index(drop=True)
    # Rule 1: last closed day vs the day before it
    bias = "bull" if closed.close.iloc[-1] > closed.close.iloc[-2] else "bear"
    # Rule 2: rejection wick at a key level (FVG / OB) flips the bias
    zones = key_zones(closed.iloc[:-1].reset_index(drop=True))
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


def turtle_soup(d, side, look=20, recent=3):
    h, l, c = d.high.values, d.low.values, d.close.values
    n = len(d)
    for k in range(n - recent, n):
        if k - look < 0:
            continue
        if side == "bull":
            ref = l[k - look:k].min()
            if l[k] < ref and c[k] > ref:
                return True
        else:
            ref = h[k - look:k].max()
            if h[k] > ref and c[k] < ref:
                return True
    return False


def cisd(d, side, recent=3, scan=12):
    o, c = d.open.values, d.close.values
    n = len(d)
    opposing = (c < o) if side == "bull" else (c > o)
    for e in range(n - 2, max(n - scan, 1) - 1, -1):
        if opposing[e] and not opposing[e + 1]:
            s = e
            while s > 0 and opposing[s - 1]:
                s -= 1
            run_open = o[s]
            for j in range(e + 1, n):
                broke = c[j] > run_open if side == "bull" else c[j] < run_open
                if broke:
                    return j >= n - recent
            return False
    return False


def volatility_stats(d, n=14, er_n=40):
    """Returns (ATR in price, ATR as % of price, efficiency ratio) from closed candles.
    Efficiency ratio (ER) = net move / total path over the last er_n candles.
    Near 1 = clean trend, near 0 = choppy sideways noise."""
    h, l, c = d.high, d.low, d.close
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(n).mean())
    price = float(c.iloc[-1])
    atr_pct = atr / price * 100 if price > 0 else 0.0
    er = 0.0
    if len(c) > er_n:
        net = abs(float(c.iloc[-1]) - float(c.iloc[-1 - er_n]))
        path = float(c.diff().abs().tail(er_n).sum())
        er = net / path if path > 0 else 0.0
    return atr, atr_pct, er


def analyze_tf(hd, bias, prox, k):
    """Best OB (bullish or bearish) near the current price on one timeframe, or None.
    hd = candles including the current unfinished one."""
    closed = hd.iloc[:-1].reset_index(drop=True)
    price = float(hd.close.iloc[-1])
    n = len(closed)
    best = None
    for side, lo, hi, j in find_structure_obs(closed, k):
        if j < n - MAX_AGE:
            continue
        if (side == "bull" and price < lo) or (side == "bear" and price > hi):
            continue  # price is already beyond the zone
        if lo <= price <= hi:
            dist = 0.0
        elif price > hi:
            dist = (price - hi) / price * 100
        else:
            dist = (lo - price) / price * 100
        key = (dist, 0 if side == bias else 1, -j)
        if dist <= prox and (best is None or key < best[0]):
            best = (key, side, lo, hi, j)
    if best is None:
        return None
    (dist, _, _), side, lo, hi, j = best
    score = 3 if dist == 0 else 2
    sigs = []
    if turtle_soup(closed, side):
        sigs.append("Turtle soup")
        score += 2
    if cisd(closed, side):
        sigs.append("CISD")
        score += 2
    for fside, flo, fhi in find_fvgs(closed.tail(100)):  # same-side FVG overlapping/near the OB
        if fside != side:
            continue
        gap = 0.0 if (fhi >= lo and flo <= hi) else (flo - hi if flo > hi else lo - fhi)
        if gap / price * 100 <= 1.0:
            sigs.append("FVG near OB")
            score += 1
            break
    # premium / discount: OB midpoint vs the high-low range of the last 100 closed candles
    rng_hi = float(closed.high.tail(100).max())
    rng_lo = float(closed.low.tail(100).min())
    pos = (((lo + hi) / 2 - rng_lo) / (rng_hi - rng_lo) * 100) if rng_hi > rng_lo else 50.0
    pd_ok = (side == "bull" and pos <= 50) or (side == "bear" and pos > 50)
    atr, atr_pct, er = volatility_stats(closed)
    zone_atr = (hi - lo) / atr if atr > 0 else 0.0
    return {"side": side, "lo": lo, "hi": hi, "dist": dist, "brk_ago": n - 1 - j, "sigs": sigs,
            "score": score, "pd_zone": "Premium" if pos > 50 else "Discount", "pos": pos,
            "pd_ok": pd_ok, "price": price, "er": er, "atr_pct": atr_pct, "zone_atr": zone_atr}


def norm(base):
    """1000PEPE -> PEPE, 1MBABYDOGE -> BABYDOGE, so names match across exchanges."""
    return re.sub(r"^(1000000|1000|1M)", "", base.upper())


@st.cache_data(ttl=21600, show_spinner=False)
def binance_futures_info():
    """Coins with a Binance USDT perpetual as {coin: 24h volume in USD}.
    If Binance's own API is blocked from this server, the list and volumes come from
    CoinGecko instead (cached for 6 hours)."""
    try:  # first choice: straight from Binance
        bx = ccxt.binanceusdm({"enableRateLimit": True})
        bx.load_markets()
        tk = bx.fetch_tickers()
        got = {}
        for sym, m in bx.markets.items():
            if (m.get("swap") and m.get("linear") and m.get("quote") == "USDT"
                    and m.get("settle") == "USDT" and m.get("active") is not False):
                v = (tk.get(sym) or {}).get("quoteVolume")
                key = norm(m["base"])
                got[key] = max(got.get(key, 0.0), float(v or 0))
        if len(got) > 100:
            return got
    except Exception:
        pass
    r = requests.get(
        "https://api.coingecko.com/api/v3/derivatives/exchanges/binance_futures",
        params={"include_tickers": "unexpired"},
        headers={"accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    out = {}
    for t in r.json().get("tickers", []):
        if t.get("target") == "USDT" and t.get("contract_type", "perpetual") == "perpetual":
            usd = (t.get("converted_volume") or {}).get("usd")
            key = norm(t["base"])
            out[key] = max(out.get(key, 0.0), float(usd or 0))
    return out


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
def btc_overview(exch, tfs, k, prox):
    """BTC daily bias plus the nearest OB on each selected timeframe."""
    ex = make_ex(exch)
    sym = "BTCUSDT" if exch == "binancespot" else "BTC/USDT:USDT"
    dd = candles(ex, sym, "1d", 60)
    bias, why = daily_bias(dd)
    done = dd.iloc[:-1]
    day_pct = (done.close.iloc[-1] / done.close.iloc[-2] - 1) * 100
    price = float(dd.close.iloc[-1])
    today_pct = (price / float(dd.open.iloc[-1]) - 1) * 100
    per_tf = {}
    for name in tfs:
        per_tf[name] = analyze_tf(candles(ex, sym, TF_MAP[name], 300), bias, prox, k)
    return {"bias": bias, "why": why, "day_pct": day_pct, "today_pct": today_pct,
            "price": price, "tfs": per_tf}


def label(b):
    return {"bull": "Bullish", "bear": "Bearish"}.get(b, "unavailable")


st.title("Coin Scanner")

EXCHANGES = {
    "OKX Perpetual (USDT)": "okx",
    "Binance Spot (USDT)": "binancespot",
}

c1, c2, c3 = st.columns(3)
exch_name = c1.selectbox("Exchange (data source)", list(EXCHANGES))
exch = EXCHANGES[exch_name]
tf_choice = c2.multiselect("OB timeframe(s)", list(TF_MAP), default=["4H"])
scan_all = c3.checkbox("Scan ALL matching coins (no limit)", True)
top_n = c3.slider("Coins to scan (top by volume)", 20, 600, 50, 10, disabled=scan_all)

c4, c5, c6 = st.columns(3)
prox = c4.slider("Max distance to OB zone (%)", 0.5, 10.0, 3.0, 0.5)
min_vol = c4.slider("Min 24h volume (million USD, 0 = no limit)", 0, 500, 100)
min_score = c5.slider("Minimum score", 1, 9, 3)
swing_k = c5.slider("Swing sensitivity (candles on each side)", 1, 5, 2)
use_dxy = c6.checkbox("Use DXY filter", True)
only_bn = c6.checkbox("Only coins also on Binance Futures", True)
fallback = c6.checkbox("If a coin is not on OKX, check it on Binance Spot (auto)", True)
only_pd = c6.checkbox("Only OBs on the correct side (bull OB in Discount, bear OB in Premium)", False)
news = c6.checkbox("High-impact news today (stricter)", False)

c7, c8, c9 = st.columns(3)
min_er = c7.slider("Min trend efficiency (0 = off, higher = cleaner trend)", 0.0, 0.6, 0.2, 0.05)
min_zone_atr = c8.slider("Min OB zone size (in ATR, 0 = off)", 0.0, 3.0, 1.0, 0.25)
min_atr_pct = c9.slider("Min ATR % per candle (0 = off)", 0.0, 3.0, 0.0, 0.1)

c10, c11, c12 = st.columns(3)
min_chg = c10.slider("Min move from today's open (± %, 0 = off)", 0.0, 20.0, 5.0, 0.5)

st.subheader("Market overview: BTC + Dollar index")
o1, o2, o3 = st.columns(3)
btc = None
dxy_i = dxy_info()
try:
    btc = btc_overview(exch, tuple(tf_choice), swing_k, prox)
except Exception as e:
    o1.warning(f"BTC overview not available from {exch_name}: {e}")
if btc:
    o1.metric("BTC daily bias", "🟢 Bullish" if btc["bias"] == "bull" else "🔴 Bearish",
              f"{btc['day_pct']:+.2f}% last closed day")
    lines = []
    for name, r in btc["tfs"].items():
        if r:
            txt = f"{name}: {'Bullish' if r['side'] == 'bull' else 'Bearish'} OB {r['lo']:.0f}-{r['hi']:.0f} ({r['dist']:.2f}% away)"
            if r["sigs"]:
                txt += " + " + ", ".join(r["sigs"])
        else:
            txt = f"{name}: no OB near price"
        lines.append(txt)
    o1.caption(
        f"{btc['why']} | Price {btc['price']:,.1f} | today so far {btc['today_pct']:+.2f}%"
        + (" | " + " | ".join(lines) if lines else "")
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

tab_scan, tab_search = st.tabs(["📊 Coin Scan", "🔎 Coin Search"])

with tab_search:
    st.caption("Type a coin name to see its details. It uses the same settings as the scanner above.")
    st.subheader("Check a single coin")
    sc1, sc2 = st.columns([3, 1])
    one = sc1.text_input("Coin name (e.g. BTC, SOL, PEPE)", "").strip().upper()
    sc2.write("")
    check_one = sc2.button("Check coin")
    if check_one and one:
        try:
            ex1, sym, src1 = find_coin(exch, one, fallback)
            if sym is None:
                st.warning(f"{one} was not found on {exch_name}"
                           + (" or on Binance Spot." if fallback and exch == "okx" else "."))
            else:
                if src1 != exch:
                    st.info(f"{one} was not found on {exch_name}, so the data below comes from Binance Spot.")
                dd1 = candles(ex1, sym, "1d", 60)
                bias1, why1 = daily_bias(dd1)
                price1 = float(dd1.close.iloc[-1])
                chg1 = (price1 / float(dd1.open.iloc[-1]) - 1) * 100
                dxy1 = dxy_bias()
                m_a, m_b, m_c = st.columns(3)
                m_a.metric(f"{one} daily bias", "🟢 Bullish" if bias1 == "bull" else "🔴 Bearish", why1)
                m_b.metric("Price", f"{price1:.6g}", f"{chg1:+.2f}% from today's open")
                m_c.metric("DXY bias", label(dxy1))
                need1 = min_score + (1 if news else 0)
                out = []
                for tfn in TF_MAP:
                    hd1 = candles(ex1, sym, TF_MAP[tfn], 300)
                    r1 = analyze_tf(hd1, bias1, 50.0, swing_k)  # wide distance: show the nearest OB anywhere
                    if not r1:
                        out.append({"TF": tfn, "OB": "no live OB", "Scanner": "❌ no OB"})
                        continue
                    adj1 = 0 if dxy1 is None else (1 if dxy1 != r1["side"] else -1)
                    score1 = r1["score"] + (adj1 if use_dxy else 0)
                    bad = []
                    if r1["dist"] > prox:
                        bad.append(f"far from OB ({r1['dist']:.1f}% > {prox}%)")
                    if r1["er"] < min_er:
                        bad.append("choppy (low ER)")
                    if r1["zone_atr"] < min_zone_atr:
                        bad.append("OB zone too small")
                    if r1["atr_pct"] < min_atr_pct:
                        bad.append("ATR % too low")
                    if abs(chg1) < min_chg:
                        bad.append("small move from open")
                    if score1 < need1:
                        bad.append(f"score {score1} < {need1}")
                    if only_pd and not r1["pd_ok"]:
                        bad.append("wrong Premium/Discount side")
                    out.append({
                        "TF": tfn,
                        "OB": "🟢 Bullish OB" if r1["side"] == "bull" else "🔴 Bearish OB",
                        "OB zone": f"{r1['lo']:.6g} – {r1['hi']:.6g}",
                        "Dist %": round(r1["dist"], 2),
                        "Setup": "With bias" if r1["side"] == bias1 else "Pullback (against bias)",
                        "Premium/Discount": f"{r1['pd_zone']} ({r1['pos']:.0f}%)",
                        "PD fit": "✅" if r1["pd_ok"] else "⚠️",
                        "Confirmations": ", ".join(r1["sigs"]) or "-",
                        "Break (candles ago)": r1["brk_ago"],
                        "Trend (ER)": round(r1["er"], 2),
                        "ATR %": round(r1["atr_pct"], 2),
                        "Zone/ATR": round(r1["zone_atr"], 1),
                        "Score": score1,
                        "Scanner": "✅ passes" if not bad else "❌ " + ", ".join(bad),
                    })
                st.dataframe(pd.DataFrame(out), hide_index=True)
                st.caption("Shows the nearest live OB on every timeframe, even if it is far from price. "
                           "The 'Scanner' column tells whether this coin would pass your current filter settings, and why not.")
        except Exception as e:
            st.error(f"Could not check {one}: {e}")

with tab_scan:
    if st.button("Scan", type="primary"):
        if not tf_choice:
            st.warning("Select at least one OB timeframe.")
            st.stop()
        try:
            ex = make_ex(exch)
            ex.load_markets()
            tick = ex.fetch_tickers()
        except Exception as e:
            st.error(f"Could not get data from {exch_name}: {e}. Try another exchange.")
            st.stop()

        bn = None
        if only_bn or (fallback and exch == "okx"):
            try:
                bn = binance_futures_info()  # {coin: Binance 24h USD volume}
            except Exception as e:
                st.error(
                    f"Could not get the Binance Futures coin list: {e}. "
                    "Wait a minute and scan again, or untick the Binance filter."
                )
                st.stop()

        pairs = []
        n_ex = 0  # USDT perpetuals on the chosen exchange
        n_bn_vol = 0  # coins whose volume came from Binance
        for s, m in ex.markets.items():
            if (m.get("swap") and m.get("linear") and m.get("quote") == "USDT"
                    and m.get("settle") == "USDT" and m.get("active") is not False
                    and m.get("base") not in STABLES and s in tick):
                n_ex += 1
                if only_bn and bn is not None and norm(m["base"]) not in bn:
                    continue
                t = tick[s]
                vol = None
                if bn is not None:
                    vol = bn.get(norm(m["base"]))  # Binance 24h volume in USD
                    if vol:
                        n_bn_vol += 1
                if not vol:  # no Binance volume for this coin: use the exchange's own number
                    vol = t.get("quoteVolume")
                    if not vol and t.get("baseVolume") and t.get("last"):
                        vol = t["baseVolume"] * t["last"]
                pairs.append((float(vol or 0), s))
        n_fb = 0  # coins taken from Binance Spot because OKX does not list them
        if fallback and exch == "okx" and bn is not None:
            okx_bases = {norm(m["base"]) for m in ex.markets.values()
                         if m.get("swap") and m.get("linear") and m.get("quote") == "USDT"}
            try:
                spot_tick = BinanceSpot().fetch_tickers()
            except Exception:
                spot_tick = {}
            for key, bvol in bn.items():
                ssym = key + "USDT"
                if key in okx_bases or key in STABLES or ssym not in spot_tick:
                    continue
                pairs.append((float(bvol or spot_tick[ssym]["quoteVolume"] or 0), "SPOT:" + ssym))
                n_fb += 1
        n_match = len(pairs)  # coins left after the Binance filter
        if any(v > 0 for v, _ in pairs):  # volume data available: keep liquid coins only
            pairs = [p for p in pairs if p[0] >= min_vol * 1_000_000]
        n_vol = len(pairs)  # coins left after the volume filter
        ranked = [s for _, s in sorted(pairs, reverse=True)]
        syms = ranked if scan_all else ranked[:top_n]
        if not syms:
            st.error("No coins matched the filters. Lower the minimum volume or untick the Binance filter.")
            st.stop()

        dxy = dxy_bias()
        spot_ex = BinanceSpot()
        rows, fails, skipped_chg = [], 0, 0
        bar = st.progress(0.0)
        for i, s in enumerate(syms):
            bar.progress((i + 1) / len(syms), text=s)
            is_spot = s.startswith("SPOT:")
            exs = spot_ex if is_spot else ex
            real = s[5:] if is_spot else s
            try:
                dd = candles(exs, real, "1d", 60)
                if len(dd) < 25:
                    continue
                # skip pegged / stable-like assets (price barely moves over 60 days)
                if (dd.high.max() - dd.low.min()) / dd.close.iloc[-1] < 0.05:
                    continue
                # change from today's daily open (like TradingView "Chg from open")
                chg_open = (float(dd.close.iloc[-1]) / float(dd.open.iloc[-1]) - 1) * 100
                if abs(chg_open) < min_chg:
                    skipped_chg += 1
                    continue
                bias, why = daily_bias(dd)
                for name in tf_choice:
                    hd = candles(exs, real, TF_MAP[name], 300)
                    if len(hd) < 60:
                        continue
                    r = analyze_tf(hd, bias, prox, swing_k)
                    if not r:
                        continue  # no OB near price on this timeframe
                    # skip choppy / sleepy coins
                    if r["er"] < min_er or r["zone_atr"] < min_zone_atr or r["atr_pct"] < min_atr_pct:
                        continue
                    adj = 0 if dxy is None else (1 if dxy != r["side"] else -1)
                    rows.append({
                        "Coin": real[:-4] if is_spot else ex.markets[s]["base"],
                        "Data": "Binance Spot" if (is_spot or exch == "binancespot") else "OKX",
                        "Daily bias": "🟢 Bullish" if bias == "bull" else "🔴 Bearish",
                        "Bias basis": why,
                        "OB type": "🟢 Bullish OB" if r["side"] == "bull" else "🔴 Bearish OB",
                        "OB TF": name,
                        "Premium/Discount": f"{r['pd_zone']} ({r['pos']:.0f}%)",
                        "PD fit": "✅ Correct side" if r["pd_ok"] else "⚠️ Wrong side",
                        "Setup": "With bias" if r["side"] == bias else "Pullback (against bias)",
                        "Confirmations": ", ".join(r["sigs"]) or "-",
                        "OB zone": f"{r['lo']:.6g} – {r['hi']:.6g}",
                        "Dist %": round(r["dist"], 2),
                        "Break (candles ago)": r["brk_ago"],
                        "Price": r["price"],
                        "Chg from open %": round(chg_open, 2),
                        "Trend (ER)": round(r["er"], 2),
                        "ATR %": round(r["atr_pct"], 2),
                        "Zone/ATR": round(r["zone_atr"], 1),
                        "base": r["score"],
                        "dxy_adj": adj,
                        "pd_ok": r["pd_ok"],
                    })
            except Exception:
                fails += 1
        bar.empty()
        st.session_state["res"] = {
            "rows": rows, "fails": fails, "n": len(syms), "exch": exch_name, "tfs": list(tf_choice),
            "bn": only_bn, "dxy": dxy,
            "n_ex": n_ex, "n_bn": len(bn) if bn is not None else None, "n_match": n_match, "n_vol": n_vol,
            "min_vol": min_vol, "scan_all": scan_all, "top_n": top_n,
            "min_chg": min_chg, "skipped_chg": skipped_chg, "n_bn_vol": n_bn_vol, "n_fb": n_fb,
            "time": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        }

    res = st.session_state.get("res")
    if res:
        st.caption(
            f"Scanned {res['n']} coins on {res['exch']} at {res['time']} UTC | OB timeframes: {', '.join(res['tfs'])} | "
            f"DXY daily bias: {label(res['dxy'])} | Binance Futures filter: {'on' if res['bn'] else 'off'} | "
            f"failed: {res['fails']}"
        )
        st.caption(
            f"Coin counts: {res['exch']} USDT perpetuals = {res['n_ex']}"
            + (f" | Binance Futures list = {res['n_bn']}" if res["n_bn"] is not None else "")
            + f" | common / after Binance filter = {res['n_match']}"
            + (f" (incl. {res['n_fb']} from the Binance Spot fallback)" if res["n_fb"] else "")
            + f" | after volume filter (min {res['min_vol']}M) = {res['n_vol']} | actually scanned = {res['n']}"
        )
        if res["n_bn"] is not None:
            st.caption(
                f"Volume source: Binance 24h volume used for {res['n_bn_vol']} of {res['n_match']} common coins "
                "(the rest use the data-source exchange's own volume). Candles come from the data-source exchange."
            )
        if res["min_chg"] > 0:
            st.caption(
                f"Move-from-open filter (± {res['min_chg']}%): removed {res['skipped_chg']} of {res['n']} coins "
                "before the OB check."
            )
        if res["n_vol"] < res["n_match"]:
            st.warning(
                f"Min 24h volume ({res['min_vol']}M USD) removed {res['n_match'] - res['n_vol']} coins. "
                "Set 'Min 24h volume' to 0 to include them."
            )
        if not res["scan_all"] and res["n"] < res["n_vol"]:
            st.warning(
                f"'Scan ALL' is off, so only the top {res['top_n']} coins by volume were scanned. "
                "Tick 'Scan ALL matching coins' to scan every coin."
            )
        if news:
            st.warning("News risk is on: the minimum score is raised by 1. Check the calendar before entering.")
        need = min_score + (1 if news else 0)
        df = pd.DataFrame(res["rows"])
        if df.empty:
            st.info("No coins had an OB near the price in this scan. Try a larger OB distance, more timeframes, or a lower minimum volume.")
        else:
            df["Score"] = df["base"] + (df["dxy_adj"] if use_dxy else 0)
            if only_pd:
                df = df[df["pd_ok"]]
            df = df[df["Score"] >= need].sort_values(["Score", "Dist %"], ascending=[False, True])
            cols = ["Coin", "Data", "Score", "Daily bias", "Bias basis", "OB type", "OB TF", "Premium/Discount",
                    "PD fit", "Setup", "Confirmations", "OB zone", "Dist %", "Break (candles ago)", "Price", "Chg from open %", "Trend (ER)", "ATR %", "Zone/ATR"]
            if df.empty:
                st.info("No coins reached the minimum score. Lower it or rescan later.")
            else:
                st.dataframe(df[cols], hide_index=True)

with st.expander("📖 Terms explained (අර්ථ විස්තර)"):
    st.markdown(
        """
### 1. Daily Bias (දවසේ දිශාව)
- **Bias**: Daily chart එකෙන් තීරණය කරන දිශාව. **Bullish 🟢** = ඉහළ යන්න වැඩි ඉඩක්, **Bearish 🔴** = පහළ යන්න වැඩි ඉඩක්. මේක අනාවැකියක් නෙවෙයි, ඔයාගේ rules වලට අනුව හදන දිශා නිගමනයක් විතරයි.
- **Rule 1 (close එකට close)**: අන්තිමට වැසුණු daily candle එක, ඊට කලින් දවසේ close එකට වඩා ඉහළින් close වුණා නම් Bullish, පහළින් close වුණා නම් Bearish.
- **Rule 2 (key level rejection)**: අන්තිම daily candle එකේ දිග wick එකක් (wick එක body එකට වඩා දෙගුණයක් හෝ වැඩි, candle එකෙන් අඩකට වඩා) key level එකක (Daily FVG හෝ Daily OB) ළඟ හැදුණා නම්, එතැනින් market එක හැරුණා කියලා සලකලා Bias එක ඒ දිශාවට වෙනස් කරනවා (Rule 1 අභිබවා). පහළ wick + bullish zone = Bullish, ඉහළ wick + bearish zone = Bearish.
- **Bias basis** (table එකේ column එක): Bias එක ආවේ Rule 1 එකෙන්ද Rule 2 එකෙන්ද කියලා පෙන්නනවා.
- **FVG (Fair Value Gap)**: candle 3ක් අතරේ, මැද candle එක වේගයෙන් ගිය නිසා පළමු candle එකේ high එකයි තුන්වැනි candle එකේ low එකයි අතර හිස් ඉඩක් ඉතිරි වීම (bearish නම් අනිත් පැත්ත). Candle එකක් ඒ කලාපය හරහා close වුණාම ඒක අවලංගුයි.

### 2. OB (ඔයාගේ නිර්වචනය)
- **Swing high / Swing low**: candle එකක high එක, ඊට කලින් සහ පස්සේ candles (Swing sensitivity ගණන) වල high වලට වඩා උසස් නම් ඒක swing high. low එක ඒවායේ low වලට වඩා පහළ නම් swing low. Sensitivity එක අඩු කළොත් කුඩා swings ගොඩක් හමුවෙනවා, වැඩි කළොත් විශාල swings විතරයි.
- **Liquidity sweep ("torch") — අනිවාර්යයි**: OB එකක් හැදෙන්න නම් අන්තිම Lower Low / Higher High එක හැදෙන candle එක කලින් swing low / high එක (liquidity) wick එකෙන් කඩලා, ඒත් ආපහු ඒ මට්ටමට ඇතුළතින් **close** වෙන්න ඕන. torch එකක් වගේ spike එකක් ගහලා හැරෙනවා. මේක නැති OB ලැයිස්තුවට එන්නේ නැහැ.
- **Bullish OB**: මුලින් down trend එකක් (Lower High සහ Lower Low) තියෙන්න ඕන. අන්තිම **Lower Low (LL)** එක කලින් swing low එක sweep කරලා ආපහු ඒකට උඩින් close වෙන්න ඕන. ඊට පස්සේ candle එකක් අන්තිම **Lower High (LH)** එකේ high එකට උඩින් **close** වුණාම (structure break එක, "shot" එක), **LH එකේ high එකේ සිට LL එකේ low එක දක්වා මුළු පරාසයම** OB zone එක.
- **Bearish OB**: ඒකේ ප්‍රතිවිරුද්ධය. Up trend එකක් (Higher High සහ Higher Low) තියෙනවා. අන්තිම Higher High එක කලින් swing high එක sweep කරලා ආපහු ඒකට යටින් close වෙන්න ඕන. ඊට පස්සේ candle එකක් අන්තිම **Higher Low** එකේ low එකට යටින් close වුණාම, ඒ Higher Low එකේ low එකේ සිට Higher High එකේ high එක දක්වා පරාසය OB zone එක.
- **අවලංගු වීම**: Bullish OB එකට යටින් candle එකක් close වුණාම, Bearish OB එකට උඩින් close වුණාම ඒ OB එක අවලංගුයි. ඒ වගේම structure break එක අන්තිම candles 150 ඇතුළත විතරයි ගන්නේ.
- **OB timeframe(s)**: 4H, 1H, 15m, 5m අතරින් එකක් හෝ කිහිපයක් තෝරන්න පුළුවන්. කිහිපයක් තෝරුවොත් ඒ ඒ timeframe එකට වෙනම row එකක් එනවා. Daily එකේ OB scan කරන්නේ නැහැ (Daily එක Bias සඳහා විතරයි, Rule 2 වලට Daily FVG/OB key level විදිහට පාවිච්චි වෙනවා).
- **OB zone**: OB එකේ low සිට high දක්වා මිල කලාපය. **Dist %**: දැන් මිල ඒ කලාපයට ඇති දුර (%). `0` = මිල කලාපය ඇතුළේ. **Break (candles ago)**: structure break එක හැදිලා candles කීයක්ද (අඩු = අලුත්).
- **OB type**: Bullish OB ද Bearish OB ද කියන එක. Daily bias එක Bearish වෙලා Bullish OB එකක් ළඟ ඉන්නවා නම් ඒක **Pullback (against bias)** setup එකක්. OB එක bias එකටම ගැළපෙනවා නම් **With bias** (Setup column එක).

### 3. Premium / Discount
- **Premium / Discount**: ඒ OB එකේම timeframe එකේ අන්තිම candles 100ක highest high සිට lowest low දක්වා range එක 50% ලකුණෙන් (equilibrium) දෙකට බෙදනවා. 50%ට **පහළ** කොටස **Discount**, 50%ට **ඉහළ** කොටස **Premium**. **Premium/Discount** column එකේ පෙන්නන්නේ OB zone එකේ මැද ඒ range එකේ කොයි තැනද කියන එකයි (0% = පහළම, 100% = ඉහළම).
- **PD fit**: Bullish OB එකක් **Discount** එකේ, Bearish OB එකක් **Premium** එකේ තියෙනවා නම් ✅ Correct side. Bullish OB එකක් Premium එකේ හෝ Bearish OB එකක් Discount එකේ තියෙනවා නම් ⚠️ Wrong side. ICT සංකල්පයට අනුව ඒ වගේ OB එකක් අවලංගු වෙන්න වැඩි ඉඩක් තියෙනවා. "Only OBs on the correct side" ටික් කළොත් ⚠️ ඒවා ලැයිස්තුවෙන් අයින් වෙනවා.

### 4. Confirmations සහ Score
- **Confirmations**: OB එක අනිවාර්යයි (මිලට ළඟ, Max distance ඇතුළත OB එකක් නැති coins ලැයිස්තුවට එන්නේ නැහැ). ඒක ඇත්නම් අමතර තහවුරු කිරීම් විදිහට: **Turtle soup**, **CISD**, සහ **FVG near OB** පෙන්නනවා. Turtle soup සහ CISD බලන්නේ OB එකේ දිශාවටයි.
- **Turtle soup**: Bullish නම්, මෑත candles 20ක අඩුම low එක මිල කඩලා (liquidity sweep), ඒත් candle එක ආපහු ඒ මට්ටමට උඩින් close වුණා නම්. Bearish නම් ඒකේ ප්‍රතිවිරුද්ධය. අන්තිම candles 3 ඇතුළත විතරයි බලන්නේ. (මේක OB එක හැදුණු sweep එකට වෙනස්: මේක දැන් මිල ළඟ වෙච්ච අලුත් sweep එකක් සඳහා අමතර ලකුණු දෙනවා.)
- **CISD (Change in State of Delivery)**: සරල අර්ථ දැක්වීම: Bullish සඳහා, අඛණ්ඩ bearish candles පෙළක් ඇවිත් ඊට පස්සේ candle එකක් ඒ පෙළේ පළමු candle එකේ open එකට උඩින් close වුණා නම් (Bearish සඳහා ප්‍රතිවිරුද්ධය). අන්තිම candles 3 ඇතුළත විතරයි.
- **FVG near OB**: OB එකේ දිශාවටම ගැළපෙන FVG එකක් OB කලාපය ඇතුළේ හෝ ඊට මිලෙන් 1% ඇතුළත තියෙනවා නම්.
- **Score**: OB (zone එකට ළඟ) +2, මිල zone එක ඇතුළේ නම් +3. Turtle soup +2, CISD +2, FVG near OB +1. DXY filter එක on නම්: OB එකේ දිශාවට (Bullish OB = long, Bearish OB = short) DXY ගැළපෙනවා නම් (උදා: Bullish OB, DXY Bearish) +1, ගැළපෙන්නේ නැත්නම් -1. ලකුණු වැඩි = ඔයාගේ rules වලට වැඩියෙන් ගැළපෙනවා, ඒත් ජයග්‍රහණයක් සහතික කරන්නේ නැහැ.
- **Minimum score**: මේ ලකුණට වඩා අඩු coins ලැයිස්තුවෙන් අයින් කරනවා. "High-impact news today" ටික් කළොත් අවශ්‍ය අවම ලකුණ 1කින් වැඩි වෙනවා.

### 4b. Choppy / sideways coins අයින් කරන filters
- **ප්‍රශ්නය**: සමහර coins එක තැනක ගොඩ වෙලා sideways යනවා, පොඩි පොඩි movement විතරයි දෙන්නේ. එහෙම coin එකක OB එකක් ගත්තාම stop loss එක hit වෙනවා, ආපහු profit පැත්තට යනවා, ආපහු stop loss hit වෙනවා. මේ filters තුනෙන් ඒ වගේ coins ලැයිස්තුවෙන් අයින් කරනවා.
- **Trend (ER) = Efficiency Ratio**: අන්තිම candles 40 තුළ මිල ඇත්තටම ගියපු දුර (මුල සිට අගට) බෙදන්නේ මිල ඉහළ පහළ ගිහින් ගියපු මුළු දුරින්. **1ට ළඟ** = පිරිසිදු trend එකක්, **0ට ළඟ** = අතන මෙතන හැමතැනම ගිහින් ආපහු ආවා (choppy). ඔයාගේ "Min trend efficiency" slider එකේ default 0.2. ඊට අඩු coins අයින් වෙනවා. තවත් strict වෙන්න ඕන නම් 0.3 දක්වා වැඩි කරන්න.
- **Zone/ATR**: OB zone එකේ උස, සාමාන්‍ය candle එකක සරල ප්‍රමාණයෙන් (ATR) කීයෙන් එකක්ද කියන එක. 1ට අඩු නම් zone එක candle එකකටත් වඩා පොඩියි, stop loss එක noise එකටම වැදෙනවා. "Min OB zone size" default 1.0.
- **ATR %**: candle එකක සාමාන්‍ය ප්‍රමාණය මිලෙන් %ක් විදිහට. අඩු = coin එක නිදිමතයි. "Min ATR % per candle" default 0 (off). Timeframe එක අනුව ප්‍රමාණය වෙනස් වෙනවා (4H එකේ සාමාන්‍යයෙන් 1-3%, 5m එකේ 0.2-0.4%), ඒ නිසා ඒක පාවිච්චි කරනවා නම් timeframe එකට ගැළපෙන අගයක් දාන්න.
- වැදගත්: මේ filters වලින් trend එකක් **සහතික** වෙන්නේ නැහැ, choppy වෙන්න වැඩි ඉඩ තියෙන coins අඩු කරනවා විතරයි. ඒ filters 0 කළොත් කලින් වගේම හැම coin එකක්ම එනවා.

### 4c. Min move from today's open (TradingView "Chg from open")
- අද දවසේ daily candle එකේ **open** මිලට වඩා දැන් මිල % කීයකින් වෙනස් වෙලාද කියන එක. TradingView screener එකේ "Chg from open outside -5% to 5%" කියන්නේ මේකමයි.
- **Min move from today's open** slider එක 5 කළොත්, අද open එකෙන් 5%ට වඩා ඉහළට හෝ පහළට ගිය coins විතරක් OB check එකට යනවා. 0 නම් මේ filter එක off.
- Results table එකේ **Chg from open %** column එකෙන් ඒ අගය පෙනෙනවා (+ = ඉහළට, − = පහළට).
- මේ filter එක OB timeframes scan කරන්න **කලින්** වැඩ කරන නිසා scan එක ඉක්මන් වෙනවා.

### 4d. Check a single coin (එක coin එකක් බලන්න)
- coin නම (උදා: BTC, SOL, PEPE) type කරලා **Check coin** එබුවාම, ඒ coin එකේ Daily bias, මිල, අද open එකෙන් වෙනස, සහ timeframe හතරේම (4H, 1H, 15m, 5m) ළඟම තියෙන OB එක පෙන්නනවා.
- OB එක මිලෙන් දුරින් තිබුණත් පෙන්නනවා, ඒ නිසා ඒ coin එකේ OB එකක් තියෙනවද කියලා බලන්න පුළුවන්.
- **Scanner** column එකෙන් කියන්නේ ඒ coin එක දැන් තියෙන filter settings වලට අනුව scan එකේ ලැයිස්තුවට එනවද කියලයි. ✅ නම් එනවා. ❌ නම් ඒකට හේතුව (OB එකෙන් දුරයි, choppy, score අඩුයි, ආදිය) පෙන්නනවා. Scan එකේ coin එකක් නොපෙනුණොත් ඒකට හේතුව දැනගන්න මේක පාවිච්චි කරන්න.

### 5. Settings (පාලන)
- **Exchange (data source)**: candles (OB, bias සඳහා) ගන්න තැන. **OKX Perpetual** (පළවෙනි, default) සහ **Binance Spot** තෝරන්න පුළුවන්. Binance Futures සහ Bybit අයින් කළා.
- **If a coin is not on OKX, check it on Binance Spot (auto)**: ටික් කරලා තියෙනවා නම්, Binance Futures එකේ තියෙන ඒත් OKX එකේ නැති coins ස්වයංක්‍රීයව Binance Spot candles වලින් scan වෙනවා (results වල **Data** column එකේ Binance Spot කියලා පෙනෙනවා). Coin Search එකේත් OKX එකේ coin එක නැත්නම් Binance Spot එකෙන් බලනවා. Spot pair එකක් නැති coins මේකෙන් scan වෙන්නේ නැහැ. Binance Spot එක විතරක් වෙනම බලන්න ඕන නම් Exchange එක ඒක තෝරන්න.
- **Coins to scan (top by volume)**: volume වැඩිම coins කීයක් scan කරනවද කියන ගණන. Timeframes වැඩිපුර තෝරන තරමට scan එක වැඩි වෙලාවක් ගත වෙනවා.
- **Min 24h volume (million USD)**: පසුගිය පැය 24 තුළ ඒ coin එක ඩොලර් කීයක් වටිනාකමට trade වුණාද (මිලියන වලින්). වැඩි = trade කරන අය ගොඩක්, හොඳ liquidity. අඩු = මිල ඉක්මනින් අස්ථාවර වෙනවා, fake moves වැඩියි. මේ අගය තෝරපු exchange එකේ දත්ත වලින් එන නිසා Binance volume එකට වඩා වෙනස් වෙන්න පුළුවන්.
- **Max distance to OB zone (%)**: මිල OB කලාපයෙන් මීට වඩා දුරින් තියෙනවා නම් ඒ OB එක ගණන් ගන්නේ නැහැ.
- **Use DXY filter**: Dollar index (DXY) එකේ bias එක coins වල ලකුණු වලට එකතු කරනවා/අඩු කරනවා.
- **Only coins also on Binance Futures**: තෝරපු exchange එකේත් Binance Futures එකේත් තියෙන coins විතරක් scan කරනවා.
- **High-impact news today**: අද CPI, Fed වගේ ලොකු news තියෙනවා නම් ටික් කරන්න. අවශ්‍ය අවම ලකුණ 1කින් වැඩි වෙනවා.
- **Price**: coin එකේ දැන් මිල (තෝරපු exchange එකේ).

### 6. Market overview (පිටුවේ උඩ කොටස)
- **BTC daily bias**: BTC එකටත් එකම Rule 1 සහ Rule 2 යොදලා දවසේ දිශාව පෙන්නනවා. යටින් පෙන්නන්නේ අන්තිම වැසුණු දවසේ % වෙනස, දැන් මිල, අද දවසේ වෙනස, සහ ඔයා තෝරපු timeframe වල BTC එකට ළඟම තියෙන OB (තියෙනවා නම්).
- **Dollar index (DXY)**: US ඩොලරයේ ශක්තිය මනින දර්ශකය. සාමාන්‍යයෙන් DXY ඉහළ යනකොට crypto පහළ යන්නත්, DXY පහළ යනකොට crypto ඉහළ යන්නත් ඉඩ වැඩියි. ඒත් ඒක හැමවිටම සත්‍ය නෙවෙයි.
- **පණිවිඩය**: BTC සහ DXY එකිනෙකට ගැළපෙනවද කියලා කෙටියෙන් කියනවා.

### අවවාදය
මේ rules ඔයාගේ strategy එකේ සරල කරපු අර්ථ දැක්වීම්. ඔයාගේ chart එකේ දකින දේට වෙනස් වෙන්න පුළුවන්. Scanner එක candidates ලැයිස්තුවක් විතරයි, trade advice නෙවෙයි. Chart එකේ තහවුරු කරලා, backtest කරලා පාවිච්චි කරන්න.
"""
    )

st.caption(
    "Candidates only, not trade advice. Confirm on the chart and backtest the rules "
    "before trading real money."
)
