import datetime as dt
import re
import time

import ccxt
import numpy as np
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

    def fetch_ohlcv(self, sym, tf, since=None, limit=300):
        time.sleep(0.05)  # stay well under Binance's request limits
        params = {"symbol": sym, "interval": tf, "limit": limit}
        if since:
            params["startTime"] = int(since)
        rows = self._get("/klines", params)
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


def tf_ms(tf):
    """'15m' -> milliseconds per candle."""
    return int(tf[:-1]) * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[tf[-1]]


def fetch_history(ex, sym, tf, days):
    """Download about `days` days of completed candles, page by page. Returns a DataFrame."""
    step = tf_ms(tf)
    now = int(time.time() * 1000)
    since = now - days * 86_400_000
    page = 1000 if isinstance(ex, BinanceSpot) else 100
    out = []
    for _ in range(600):  # safety cap on the number of pages
        rows = ex.fetch_ohlcv(sym, tf, since=since, limit=page)
        if not rows:
            break
        out.extend(rows)
        nxt = rows[-1][0] + step
        if nxt >= now or nxt <= since:
            break
        since = nxt
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "vol"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df[df["ts"] + step <= now].reset_index(drop=True)  # drop the unfinished candle


def volume_profile(h, l, v, bins=100, va=0.70):
    """Volume profile of one range of candles. Each candle's volume is spread evenly over the
    price bins its high-low range covers. Returns POC, VAH, VAL (or None if there is no data)."""
    lo, hi = float(np.min(l)), float(np.max(h))
    if hi <= lo or float(np.sum(v)) <= 0:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    for hh, ll, vv in zip(h, l, v):
        if hh <= ll:
            vol[min(int((hh - lo) / (hi - lo) * bins), bins - 1)] += vv
            continue
        inside = np.clip(edges[1:], ll, hh) - np.clip(edges[:-1], ll, hh)
        vol += vv * inside / (hh - ll)
    poc_i = int(vol.argmax())
    total, got = vol.sum(), vol[poc_i]
    lo_i = hi_i = poc_i
    while got < va * total and (lo_i > 0 or hi_i < bins - 1):  # grow the value area around the POC
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1.0
        dn = vol[lo_i - 1] if lo_i > 0 else -1.0
        if up >= dn:
            hi_i += 1
            got += vol[hi_i]
        else:
            lo_i -= 1
            got += vol[lo_i]
    return {"POC": float((edges[poc_i] + edges[poc_i + 1]) / 2),
            "VAH": float(edges[hi_i + 1]), "VAL": float(edges[lo_i])}


def backtest_vp(df, tf_min, levels, stop_atr=1.0, rr=1.5, va=0.70, fee_pct=0.1, atr_n=14):
    """Previous-day volume profile reaction backtest on completed candles.
    Each UTC day the POC/VAH/VAL of the PREVIOUS day are used (no look-ahead). The first touch of a
    level is traded with a limit order at the level: long if price comes down onto it, short if it
    comes up to it. Stop = stop_atr x ATR beyond the level, target = rr x risk, time exit at day end.
    If stop and target are inside the same candle the stop is assumed first. One trade at a time."""
    n = len(df)
    if n < atr_n + 3 or not levels:
        return []
    ts = df["ts"].values.astype("int64")
    o, h, l, c, v = (df[k].values.astype(float) for k in ("open", "high", "low", "close", "vol"))
    day = ts // 86_400_000
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.full(n, np.nan)
    for i in range(atr_n, n):
        atr[i] = tr[i - atr_n:i].mean()  # only candles before i
    per_day = 1440 / tf_min
    days = np.unique(day)
    trades = []

    def rec(d, name, side, entry, stop, target, exit_px, outcome, risk):
        gross = (exit_px - entry) / risk if side == "long" else (entry - exit_px) / risk
        net = gross - (fee_pct / 100.0) * entry / risk
        trades.append({"day": pd.Timestamp(int(d) * 86_400_000, unit="ms").strftime("%Y-%m-%d"),
                       "level": name, "side": side, "entry": entry, "stop": stop, "target": target,
                       "exit": exit_px, "outcome": outcome, "R": float(net),
                       "risk_pct": float(risk / entry * 100)})

    for k in range(1, len(days)):
        prev = np.where(day == days[k - 1])[0]
        cur = np.where(day == days[k])[0]
        if len(cur) == 0 or len(prev) < 0.8 * per_day:
            continue
        prof = volume_profile(h[prev], l[prev], v[prev], va=va)
        if prof is None:
            continue
        lv = {name: prof[name] for name in levels}
        used, pos = set(), None
        for i in cur:
            if pos is not None:
                if pos["side"] == "long":
                    if l[i] <= pos["stop"]:
                        rec(days[k], pos["name"], "long", pos["entry"], pos["stop"], pos["target"], pos["stop"], "stop", pos["risk"]); pos = None; continue
                    if h[i] >= pos["target"]:
                        rec(days[k], pos["name"], "long", pos["entry"], pos["stop"], pos["target"], pos["target"], "target", pos["risk"]); pos = None; continue
                else:
                    if h[i] >= pos["stop"]:
                        rec(days[k], pos["name"], "short", pos["entry"], pos["stop"], pos["target"], pos["stop"], "stop", pos["risk"]); pos = None; continue
                    if l[i] <= pos["target"]:
                        rec(days[k], pos["name"], "short", pos["entry"], pos["stop"], pos["target"], pos["target"], "target", pos["risk"]); pos = None; continue
            touched = []
            for name, L in lv.items():
                if name in used:
                    continue
                if o[i] > L and l[i] <= L:
                    touched.append((abs(o[i] - L), name, L, "long"))
                elif o[i] < L and h[i] >= L:
                    touched.append((abs(o[i] - L), name, L, "short"))
            if not touched:
                continue
            touched.sort()
            for t in touched:
                used.add(t[1])  # only the first touch of a level each day counts
            if pos is not None:
                continue
            _, name, L, side = touched[0]
            a = atr[i]
            if not np.isfinite(a) or a <= 0:
                continue
            risk = stop_atr * a
            stop = L - risk if side == "long" else L + risk
            target = L + rr * risk if side == "long" else L - rr * risk
            if (side == "long" and l[i] <= stop) or (side == "short" and h[i] >= stop):
                rec(days[k], name, side, L, stop, target, stop, "stop", risk)  # stopped in the entry candle
                continue
            pos = {"name": name, "side": side, "entry": L, "stop": stop, "target": target, "risk": risk}
        if pos is not None:  # time exit at the end of the day
            rec(days[k], pos["name"], pos["side"], pos["entry"], pos["stop"], pos["target"],
                float(c[cur[-1]]), "time", pos["risk"])
    return trades


def summarize(trades):
    """Headline numbers of a list of trades."""
    if not trades:
        return {"trades": 0, "win": 0.0, "avg_r": 0.0, "total_r": 0.0, "pf": 0.0, "max_dd": 0.0,
                "h1": float("nan"), "h2": float("nan"), "risk_pct": 0.0}
    R = np.array([t["R"] for t in trades])
    wins, losses = R[R > 0].sum(), -R[R < 0].sum()
    pf = wins / losses if losses > 0 else (99.0 if wins > 0 else 0.0)
    cum = np.cumsum(R)
    half = len(R) // 2
    return {"trades": int(len(R)), "win": float((R > 0).mean() * 100), "avg_r": float(R.mean()),
            "total_r": float(R.sum()), "pf": float(min(pf, 99.0)),
            "max_dd": float((np.maximum.accumulate(cum) - cum).max()),
            "h1": float(R[:half].mean()) if half > 0 else float("nan"),
            "h2": float(R[half:].mean()),
            "risk_pct": float(np.mean([t["risk_pct"] for t in trades]))}


@st.cache_resource(ttl=3600, show_spinner=False)
def get_ex(exch):
    """One shared exchange object with its markets loaded (used for the long history downloads)."""
    ex = make_ex(exch)
    ex.load_markets()
    return ex


@st.cache_data(ttl=3600, show_spinner=False)
def history_df(exch, sym, tf, days):
    """Completed candles for roughly `days` days (cached for an hour, so changing stop / R settings is instant)."""
    return fetch_history(get_ex(exch), sym, tf, days)


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

def run_scan():
    """Runs the full coin scan with the current settings and stores it in session_state["res"]."""
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
                    "ob_lo": r["lo"],
                    "ob_hi": r["hi"],
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



prog_box = st.container()  # scan progress and messages show here, above the tabs
tab_scan, tab_entry, tab_search, tab_vp = st.tabs(["📊 Coin Scan", "🎯 Entry Ready", "🔎 Coin Search", "🧪 VP Backtest"])

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
    scan_clicked = st.button("Scan", type="primary")

with tab_entry:
    st.subheader("Entry-ready coins")
    st.caption("Coins that match ALL your settings AND the entry rules below right now. "
               "Untick a rule to make the list looser.")
    e1, e2 = st.columns(2)
    e_inside = e1.checkbox("Price is inside the OB zone now", True)
    e_bias = e1.checkbox("OB direction matches the daily bias", True)
    e_pd = e1.checkbox("OB is on the correct Premium/Discount side", True)
    e_conf = e2.checkbox("At least one confirmation (Turtle soup / CISD / FVG)", True)
    e_dxy = e2.checkbox("DXY supports the direction", True)
    entry_clicked = st.button("Find entries now", type="primary")

if scan_clicked or entry_clicked:
    with prog_box:
        run_scan()

with tab_scan:
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

with tab_entry:
    eres = st.session_state.get("res")
    if not eres:
        st.info("Press 'Find entries now' (or run a Coin Scan) first.")
    else:
        st.caption(f"Based on the scan of {eres['n']} coins on {eres['exch']} at {eres['time']} UTC | "
                   f"OB timeframes: {', '.join(eres['tfs'])}")
        edf = pd.DataFrame(eres["rows"])
        if edf.empty:
            st.info("The scan found no coin with an OB near the price.")
        else:
            edf["Score"] = edf["base"] + (edf["dxy_adj"] if use_dxy else 0)
            edf = edf[edf["Score"] >= min_score + (1 if news else 0)]
            if e_inside:
                edf = edf[edf["Dist %"] == 0]
            if e_bias:
                edf = edf[edf["Setup"] == "With bias"]
            if e_pd:
                edf = edf[edf["pd_ok"]]
            if e_conf:
                edf = edf[edf["Confirmations"] != "-"]
            if e_dxy and eres["dxy"] is not None:
                edf = edf[edf["dxy_adj"] == 1]
            if edf.empty:
                st.info("No coin matches all the entry rules right now. Try again later, "
                        "or untick some of the rules above.")
            else:
                edf = edf.sort_values(["Score", "Break (candles ago)"], ascending=[False, True]).copy()
                edf["Direction"] = edf["OB type"].map(lambda t: "🟢 LONG" if "Bullish" in t else "🔴 SHORT")
                edf["Invalidation"] = [f"{lo:.6g}" if "Bullish" in t else f"{hi:.6g}"
                                       for lo, hi, t in zip(edf["ob_lo"], edf["ob_hi"], edf["OB type"])]
                ecols = ["Coin", "Data", "Direction", "Score", "OB TF", "OB zone", "Price", "Invalidation",
                         "Daily bias", "Premium/Discount", "Confirmations", "Chg from open %"]
                st.dataframe(edf[ecols], hide_index=True)
                st.caption("Invalidation = the far edge of the OB zone. A candle closing beyond it cancels that OB. "
                           "Candidates only, not trade advice: confirm on the chart first.")

with tab_vp:
    st.subheader("Volume profile reaction backtest")
    st.caption("Separate from the other pages. Every day the app builds a volume profile of the PREVIOUS UTC day and "
               "tests what happens when price first touches that profile's POC, VAH or VAL: a limit order at the level, "
               "stop beyond it, fixed R target, exit at the end of the day. Uses the data source chosen at the top.")
    p1, p2, p3, p4 = st.columns(4)
    vp_months = p1.slider("Backtest length (months)", 1, 12, 3)
    vp_tf = p2.selectbox("Candle size", ["15m", "30m", "1h", "5m"], index=0)
    vp_levels = p3.multiselect("Levels to trade", ["POC", "VAH", "VAL"], default=["POC", "VAH", "VAL"])
    vp_va = p4.slider("Value area (%)", 60, 80, 70)
    p5, p6, p7 = st.columns(3)
    vp_stop = p5.slider("Stop distance (x ATR beyond the level)", 0.25, 3.0, 1.0, 0.25)
    vp_rr = p6.slider("Target (R multiple)", 0.5, 4.0, 1.5, 0.25)
    vp_fee = p7.slider("Round-trip fees + slippage (%)", 0.0, 0.3, 0.1, 0.01)


def vp_params():
    return (f"{vp_months} months of {vp_tf} | levels {', '.join(vp_levels)} | value area {vp_va}% | "
            f"stop {vp_stop}x ATR | target {vp_rr}R | fees {vp_fee}%")


def vp_eval(df, name, src):
    """Backtest one coin's candles with the current settings."""
    trades = backtest_vp(df, tf_ms(vp_tf) // 60_000, vp_levels, vp_stop, vp_rr, vp_va / 100.0, vp_fee)
    days = int(len(np.unique(df["ts"].values // 86_400_000))) if len(df) else 0
    return {"coin": name, "src": src, "days": days, "trades": trades, "sum": summarize(trades),
            "params": vp_params()}


def vp_run_coin(coin):
    """Download history for one coin (OKX first, Binance Spot as a fallback) and backtest it."""
    ex1, sym1, src1 = find_coin(exch, coin, fallback)
    if sym1 is None:
        return None
    df = history_df(src1, sym1, vp_tf, vp_months * 30 + 1)
    return vp_eval(df, coin, src1)


def vp_row(r):
    z = r["sum"]
    return {"Coin": r["coin"], "Data": "Binance Spot" if r["src"] == "binancespot" else "OKX",
            "Days": r["days"], "Trades": z["trades"], "Win %": round(z["win"], 1),
            "Avg R": round(z["avg_r"], 2), "Profit factor": round(z["pf"], 2),
            "Total R": round(z["total_r"], 1), "Max drawdown (R)": round(z["max_dd"], 1),
            "1st half avg R": round(z["h1"], 2) if z["h1"] == z["h1"] else None,
            "2nd half avg R": round(z["h2"], 2), "Avg stop %": round(z["risk_pct"], 2)}


with tab_vp:
    st.markdown("#### 1. Backtest one coin")
    b1, b2 = st.columns([3, 1])
    vp_coin = b1.text_input("Coin name (e.g. PUMP, SOL)", "PUMP", key="vp_coin").strip().upper()
    b2.write("")
    if b2.button("Run backtest", key="vp_run_one") and vp_coin:
        if not vp_levels:
            st.warning("Pick at least one level to trade.")
        else:
            with st.spinner(f"Downloading {vp_months} months of {vp_tf} candles for {vp_coin} ..."):
                try:
                    out1 = vp_run_coin(vp_coin)
                    if out1 is None:
                        st.warning(f"{vp_coin} was not found on {exch_name}"
                                   + (" or on Binance Spot." if fallback and exch == "okx" else "."))
                    else:
                        st.session_state["vp_one"] = out1
                except Exception as e:
                    st.error(f"Could not backtest {vp_coin}: {e}")
    r1 = st.session_state.get("vp_one")
    if r1:
        z = r1["sum"]
        st.caption(f"{r1['coin']} | data: {'Binance Spot' if r1['src'] == 'binancespot' else 'OKX'} | "
                   f"{r1['days']} days of candles | {r1['params']}")
        if r1["days"] < vp_months * 30 * 0.8:
            st.warning(f"Only {r1['days']} days of candles were found (new coin or limited history), "
                       "so the period is shorter than you asked for.")
        if z["trades"] < 20:
            st.warning(f"Only {z['trades']} trades: too few to trust. Use a longer period or more levels.")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Trades", z["trades"])
        m2.metric("Win rate", f"{z['win']:.1f}%")
        m3.metric("Avg R / trade", f"{z['avg_r']:+.2f}")
        m4.metric("Profit factor", f"{z['pf']:.2f}")
        m5.metric("Total R", f"{z['total_r']:+.1f}")
        h1txt = "n/a" if z["h1"] != z["h1"] else f"{z['h1']:+.2f}"
        st.caption(f"Max drawdown {z['max_dd']:.1f} R | average R: first half {h1txt}, second half {z['h2']:+.2f} | "
                   f"average stop distance {z['risk_pct']:.2f}% of price. (1R = the distance from entry to stop.)")
        if r1["trades"]:
            tdf = pd.DataFrame(r1["trades"])
            lv = tdf.groupby("level").agg(Trades=("R", "size"), AvgR=("R", "mean"), TotalR=("R", "sum"),
                                          Win=("R", lambda x: (x > 0).mean() * 100)).reset_index()
            lv = lv.rename(columns={"level": "Level", "AvgR": "Avg R", "TotalR": "Total R", "Win": "Win %"}).round(2)
            st.dataframe(lv, hide_index=True)
            st.line_chart(pd.DataFrame({"Cumulative R": tdf["R"].cumsum().values}))
            with st.expander("All trades"):
                st.dataframe(tdf[["day", "level", "side", "entry", "stop", "target", "exit", "outcome", "R"]],
                             hide_index=True)

    st.markdown("#### 2. Find coins like this")
    st.caption("Backtests many coins with the settings above and keeps the ones that pass your quality limits. "
               "Each coin downloads thousands of candles, so a long period with many coins takes minutes.")
    q1, q2, q3 = st.columns(3)
    sc_n = q1.slider("Coins to test (top by 24h volume)", 10, 150, 30, 10)
    sc_vol = q1.slider("Min 24h volume (million USD)", 0, 500, 0, key="vp_sc_vol")
    sc_trades = q2.slider("Min trades", 5, 100, 20)
    sc_avg = q2.slider("Min average R per trade", -0.5, 1.0, 0.1, 0.05)
    sc_pf = q3.slider("Min profit factor", 0.0, 3.0, 1.2, 0.1)
    sc_both = q3.checkbox("Both halves of the period must be profitable", True)
    sc_bench = st.text_input("Benchmark coin (shown on top for comparison, empty = none)", "PUMP",
                             key="vp_bench").strip().upper()
    if st.button("Scan coins", key="vp_scan_btn"):
        if not vp_levels:
            st.warning("Pick at least one level to trade.")
        else:
            try:
                exu = get_ex(exch)
                tick = exu.fetch_tickers()
            except Exception as e:
                tick = None
                st.error(f"Could not get the coin list from {exch_name}: {e}")
            if tick is not None:
                pairs = []
                for s_, m_ in exu.markets.items():
                    if (m_.get("swap") and m_.get("linear") and m_.get("quote") == "USDT"
                            and m_.get("settle") == "USDT" and m_.get("active") is not False
                            and m_.get("base") not in STABLES and s_ in tick):
                        t_ = tick[s_]
                        vol_ = t_.get("quoteVolume")
                        if not vol_ and t_.get("baseVolume") and t_.get("last"):
                            vol_ = t_["baseVolume"] * t_["last"]
                        pairs.append((float(vol_ or 0), s_))
                if any(v_ > 0 for v_, _ in pairs):
                    pairs = [p_ for p_ in pairs if p_[0] >= sc_vol * 1_000_000]
                syms = [s_ for _, s_ in sorted(pairs, reverse=True)[:sc_n]]
                if not syms:
                    st.error("No coins matched. Lower the minimum volume.")
                else:
                    bar = st.progress(0.0)
                    res_rows, fails = [], 0
                    for i, s_ in enumerate(syms):
                        bar.progress((i + 1) / len(syms), text=s_)
                        try:
                            dfh = history_df(exch, s_, vp_tf, vp_months * 30 + 1)
                            res_rows.append(vp_eval(dfh, exu.markets[s_]["base"], exch))
                        except Exception:
                            fails += 1
                    bar.empty()
                    bench = None
                    if sc_bench:
                        try:
                            bench = vp_run_coin(sc_bench)
                        except Exception:
                            bench = None
                    st.session_state["vp_scan"] = {
                        "rows": res_rows, "bench": bench, "fails": fails, "n": len(syms),
                        "params": vp_params(), "exch": exch_name,
                        "time": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")}

    rs = st.session_state.get("vp_scan")
    if rs:
        def vp_pass(z):
            ok = z["trades"] >= sc_trades and z["avg_r"] >= sc_avg and z["pf"] >= sc_pf
            if sc_both:
                ok = ok and z["h1"] == z["h1"] and z["h1"] > 0 and z["h2"] > 0
            return ok

        st.caption(f"Tested {rs['n']} coins on {rs['exch']} at {rs['time']} UTC | {rs['params']} | failed: {rs['fails']}")
        if rs["bench"]:
            st.markdown("**Benchmark**")
            st.dataframe(pd.DataFrame([vp_row(rs["bench"])]), hide_index=True)
        good = [vp_row(r) for r in rs["rows"] if vp_pass(r["sum"])]
        if not good:
            st.info("No coin passed your quality limits. Lower the limits above, or test more coins.")
        else:
            gdf = pd.DataFrame(good).sort_values(["Avg R", "Trades"], ascending=[False, False])
            st.markdown(f"**{len(gdf)} coins passed**")
            st.dataframe(gdf, hide_index=True)
        st.caption("Picking the best coins out of many from the same period flatters the numbers: some of them are just "
                   "lucky. Treat this list as candidates, then check each one with Backtest one coin on a longer "
                   "period or a different candle size. Backtests are not a promise of future results.")

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

### 4e. Entry Ready page (දැන්ම entry එකක් ගන්න පුළුවන් coins)
- **Find entries now** එබුවාම (හෝ Coin Scan එකක් කළාට පස්සේ) ඔයාගේ settings ඔක්කොමත්, පහළ entry rules වලටත් **එකවර ගැළපෙන** coins විතරක් පෙන්නනවා.
- Entry rules (ටික් එක අයින් කළොත් ඒ rule එක නැතිව ලැයිස්තුව පුළුල් වෙනවා): **මිල OB zone එක ඇතුළේ**, **OB එකේ දිශාව Daily bias එකට ගැළපෙනවා**, **Premium/Discount හරි පැත්තේ**, **අඩුම තරමේ confirmation එකක්** (Turtle soup / CISD / FVG), **DXY දිශාවට හිතකරයි**.
- **Direction**: Bullish OB = 🟢 LONG, Bearish OB = 🔴 SHORT. **Invalidation**: OB zone එකේ අනිත් කෙළවර (LONG නම් zone එකේ පහළ, SHORT නම් ඉහළ). candle එකක් ඒක ඉක්මවා close වුණොත් ඒ OB එක අවලංගුයි.
- ලැයිස්තුව හිස් නම්, දැන් එහෙම coin එකක් නැහැ කියන එකයි. නැවත පස්සේ scan කරන්න, නැත්නම් rules ටික් අයින් කරන්න.
- මේ candidates ලැයිස්තුවක් විතරයි, trade advice නෙවෙයි. chart එකේ තහවුරු කරගන්න.

### 4f. VP Backtest page (Volume profile backtest)
- **අදහස**: කලින් දවසේ (ඊයේ, UTC) fixed-range volume profile එකේ **POC** (වැඩිම volume ගිය මිල), **VAH** / **VAL** (volume වලින් 70%ක් ගිය පරාසයේ ඉහළ / පහළ කෙළවර) වලට අද දවසේ මිල react කරනවද කියලා, දවස් ගාණක් පුරා backtest කරනවා. Coin එකේ මිල ඒ මට්ටම් වලට ගොඩක් වෙලාවට respond කරනවා නම්, ඒ වගේ coin එකක trade කිරීම ලේසියි කියන ඔයාගේ අදහසට ගැළපෙනවා.
- **Trade නීතිය**: හැම දවසකම ඊයේ දවසේ profile එකෙන් POC/VAH/VAL ගන්නවා (අදට කලින් දත්ත විතරයි, අනාගතය බලන්නේ නැහැ). මිල ඉහළින් ඇවිත් මට්ටමට පහළට ගැටුණොත් **LONG**, පහළින් ඇවිත් ඉහළට ගැටුණොත් **SHORT**, මට්ටමේම limit order එකක්. **Stop** = මට්ටමෙන් ඔබ්බට ATR ගුණයක් (default 1×), **Target** = risk එකේ R ගුණයක් (default 1.5R), තවම open නම් දවස අවසානයේ close එකෙන් ඉවත් වෙනවා. මට්ටමක් දවසකට පළවෙනි වතාවට ගැටෙන වෙලාවේ විතරයි trade කරන්නේ, එක වරකට එක trade.
- **R**: 1R = entry එකේ සිට stop එක දක්වා දුර. ජයග්‍රහණයක් +1.5R, පරාජයක් −1R. **Avg R** = trade එකකින් සාමාන්‍යයෙන් R කීයක්ද (fees අඩුකරලා). **Profit factor** = මුළු ලාභය ÷ මුළු පාඩුව (1ට වැඩි = ලාභයි). **Max drawdown** = එකතුවේ ඉහළම ලක්ෂ්‍යයෙන් වැටුණු උපරිම R ප්‍රමාණය.
- **1st half / 2nd half avg R**: trades අඩකට බෙදලා දෙපැත්තේම ලාභයද බලනවා. දෙකම ධනාත්මක නම් එක වාසනාවන්ත කාලයකින් එන ප්‍රතිඵලයක් නෙවෙයි වෙන්න වැඩි ඉඩක්.
- **1. Backtest one coin**: coin නමක් (උදා: PUMP) type කරලා මාස ගාණක් (1-12) තෝරලා run කරන්න. OKX එකේ නැත්නම් Binance Spot එකෙන් ගන්නවා. Trades 20ට අඩු නම් ප්‍රතිඵලය විශ්වාස කරන්න බැහැ කියලා අවවාද කරනවා.
- **2. Find coins like this**: volume වැඩිම coins N ගාණක් එකම settings වලින් backtest කරලා, ඔයා දාපු ගුණාත්මක සීමා (min trades, min avg R, min profit factor, දෙපැත්තම ලාභයි) පසුකරන coins ලැයිස්තුවක් දෙනවා. **Benchmark** (default PUMP) උඩින් පෙන්නන නිසා ඒ coin එකත් එක්ක සංසන්දනය කරන්න පුළුවන්. Scan කරලා ඉවරවුණාම සීමා slider වෙනස් කළොත් ආයෙත් scan කරන්න ඕන නැහැ.
- **Download**: coin එකකට candles දහස් ගණනක් බාගන්නවා, ඒ නිසා මාස 3ක් coins ගොඩක් scan කරද්දී විනාඩි ගණනක් යනවා. බාගත් දත්ත පැයක් cache වෙනවා, ඒ නිසා stop / R / fees වෙනස් කරලා ආයෙත් run කරන එක ඉක්මනයි.
- **අවවාදය**: coins ගොඩකින් හොඳම ඒවා තෝරද්දී සමහර ඒවා හුදෙක් වාසනාවට හොඳ පෙනෙන්න පුළුවන්. මේ ලැයිස්තුව candidates විතරයි, ඒවා වෙනම backtest කරලා (වැඩි කාලයක් / වෙනස් candle size එකකින්) තහවුරු කරන්න. අතීත ප්‍රතිඵල අනාගතය සහතික කරන්නේ නැහැ.

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
