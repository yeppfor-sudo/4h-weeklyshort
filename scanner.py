import os
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import matplotlib

matplotlib.use("Agg")
import mplfinance as mpf  # noqa: E402

BASE = "https://contract.mexc.com"
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")
STATE_FILE = "state.json"

# ---------------- SETTINGS ----------------
# Weekly "already pumped" filter
W_LOOKBACK = 26        # weeks to look back
W_PUMP_MIN = 80.0      # min % pump (low -> high) inside lookback
W_KEEP_MIN = 0.50      # price must still hold >= 50% of that weekly move

# 4h pattern
H4_CANDLES = 300
IMPULSE_LOOKBACK = 120   # candles to search for the pump top (20 days)
BASE_WINDOW = 60         # candles before the top used to find base low
IMPULSE_MIN = 40.0       # min % pump base low -> top
MIN_BARS_AFTER = 6       # min candles since the top
MAX_BARS_AFTER = 70      # max candles since the top
MAX_RETRACE = 0.50       # max pullback as fraction of the impulse
RANGE_BARS = 12          # tight range window
RANGE_MAX_PCT = 25.0     # max % width of that range
EMA_SPREAD_MAX = 6.0     # max % spread between EMA 6/12/20
NEAR_EMA20_MAX = 8.0     # price must be within this % above EMA20
HOLD_MIN_CLOSES = 8      # of last 12 closes, at least this many above EMA20

COOLDOWN_H = 48          # do not repeat the same pair within this many hours
WORKERS = 6
MAX_ALERTS = 15
# ------------------------------------------

S = requests.Session()


def get(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.5 * (i + 1))
                continue
            r.raise_for_status()
            time.sleep(0.05)
            return r.json()
        except Exception:
            time.sleep(0.8 * (i + 1))
    return None


def get_symbols():
    j = get(BASE + "/api/v1/contract/detail")
    out = []
    for c in (j or {}).get("data", []) or []:
        if c.get("quoteCoin") == "USDT" and c.get("state") == 0 and not c.get("isHidden"):
            out.append(c["symbol"])
    return sorted(out)


def klines(sym, interval, n, secs):
    end = int(time.time())
    start = end - n * secs
    j = get(
        f"{BASE}/api/v1/contract/kline/{sym}",
        {"interval": interval, "start": start, "end": end},
    )
    d = (j or {}).get("data")
    if not d or not d.get("time"):
        return None
    try:
        df = pd.DataFrame(
            {
                "time": d["time"],
                "open": d["open"],
                "high": d["high"],
                "low": d["low"],
                "close": d["close"],
                "vol": d["vol"],
            }
        ).astype(float)
    except Exception:
        return None
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("dt").sort_index()
    return df


# ---------------- WEEKLY FILTER ----------------
def weekly_pumped(df):
    if df is None or len(df) < 8:
        return None
    df = df.tail(W_LOOKBACK)
    lows = df["low"].values
    highs = df["high"].values
    min_low = np.inf
    best = 0.0
    best_low = best_high = None
    for i in range(len(df)):
        if lows[i] > 0:
            min_low = min(min_low, lows[i])
        if np.isfinite(min_low) and min_low > 0:
            g = (highs[i] / min_low - 1) * 100
            if g > best:
                best, best_low, best_high = g, min_low, highs[i]
    if best < W_PUMP_MIN or best_low is None:
        return None
    close = df["close"].values[-1]
    rng = best_high - best_low
    if rng <= 0:
        return None
    keep = (close - best_low) / rng
    if keep < W_KEEP_MIN:
        return None
    return {"w_pump": best, "w_low": best_low, "w_high": best_high, "w_keep": keep}


def weekly_job(sym):
    df = klines(sym, "Week1", W_LOOKBACK + 4, 7 * 86400)
    r = weekly_pumped(df)
    return (sym, r) if r else None


# ---------------- 4H PATTERN (CONSOLIDATION ONLY) ----------------
def detect(df):
    c = df.copy()
    now = pd.Timestamp.now(tz="UTC")
    if c.index[-1] + pd.Timedelta(hours=4) > now:
        c = c.iloc[:-1]  # drop the forming candle
    if len(c) < 130:
        return None
    for n in (6, 12, 20):
        c[f"e{n}"] = c["close"].ewm(span=n, adjust=False).mean()

    w = c.tail(IMPULSE_LOOKBACK)
    H = w["high"].values
    L = w["low"].values
    C = w["close"].values
    E6 = w["e6"].values
    E12 = w["e12"].values
    E20 = w["e20"].values

    # --- impulse top (search all candles before the last one)
    pre_H = H[:-1]
    hi_i = int(pre_H.argmax())
    after = len(pre_H) - 1 - hi_i
    if after < MIN_BARS_AFTER or after > MAX_BARS_AFTER or hi_i < 20:
        return None

    lo_start = max(0, hi_i - BASE_WINDOW)
    base_low = float(L[lo_start : hi_i + 1].min())
    swing_high = float(pre_H[hi_i])
    if base_low <= 0 or swing_high <= base_low:
        return None
    impulse = (swing_high / base_low - 1) * 100
    if impulse < IMPULSE_MIN:
        return None

    # --- shallow pullback after the top
    seg_low = float(L[hi_i + 1 : -1].min())
    retrace = (swing_high - seg_low) / (swing_high - base_low)
    if retrace > MAX_RETRACE:
        return None

    # --- tight range (levels from candles BEFORE the last one)
    rh_prev = float(H[-RANGE_BARS - 1 : -1].max())
    rl_prev = float(L[-RANGE_BARS - 1 : -1].min())
    range_pct = (rh_prev / rl_prev - 1) * 100
    if range_pct > RANGE_MAX_PCT:
        return None

    e6, e12, e20 = E6[-1], E12[-1], E20[-1]
    last_c = C[-1]

    # --- EMAs compressed, bullish, rising
    spread = (max(e6, e12, e20) - min(e6, e12, e20)) / last_c * 100
    if spread > EMA_SPREAD_MAX:
        return None
    if e12 < e20 * 0.995:
        return None
    if E20[-1] <= E20[-20]:
        return None

    # --- NO breakout: close must not be above the range high
    if last_c > rh_prev:
        return None
    # --- NO breakdown: close must hold EMA20 and the range low
    if last_c < e20 or last_c < rl_prev:
        return None

    # --- price hugging the EMAs
    near = (last_c / e20 - 1) * 100
    if near > NEAR_EMA20_MAX:
        return None
    held = int((C[-RANGE_BARS:] >= E20[-RANGE_BARS:] * 0.995).sum())
    if held < HOLD_MIN_CLOSES:
        return None

    return {
        "sig": "CONSOLIDATION",
        "candle_time": c.index[-1],
        "price": float(last_c),
        "base_low": base_low,
        "swing_high": swing_high,
        "range_high": rh_prev,
        "range_low": rl_prev,
        "impulse": impulse,
        "retrace": retrace * 100,
        "range_pct": range_pct,
        "spread": spread,
        "after": after,
        "near": near,
    }


# ---------------- CHART ----------------
def make_chart(sym, df, info):
    d = df.tail(130).copy()
    full = df["close"]
    e6 = full.ewm(span=6, adjust=False).mean().loc[d.index]
    e12 = full.ewm(span=12, adjust=False).mean().loc[d.index]
    e20 = full.ewm(span=20, adjust=False).mean().loc[d.index]
    d = d.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "vol": "Volume"}
    )
    d.index = d.index.tz_convert(None)
    e6.index = e12.index = e20.index = d.index

    mc = mpf.make_marketcolors(
        up="#1fa971", down="#ef4e5c", edge="inherit", wick="inherit", volume="inherit"
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        facecolor="white",
        figcolor="white",
        gridcolor="#eeeeee",
        gridstyle="-",
    )
    ap = [
        mpf.make_addplot(e6, color="#e056c8", width=1.1),
        mpf.make_addplot(e12, color="#ff9f1c", width=1.1),
        mpf.make_addplot(e20, color="#6c5ce7", width=1.3),
    ]
    levels = [info["range_high"], info["range_low"], info["swing_high"], info["base_low"]]
    colors = ["#2d98da", "#2d98da", "#8e44ad", "#8e44ad"]
    fig, axes = mpf.plot(
        d,
        type="candle",
        style=style,
        addplot=ap,
        hlines=dict(hlines=levels, colors=colors, linestyle="--", linewidths=0.8),
        figsize=(9, 6.2),
        title=f"{sym}  4H  CONSOLIDATION",
        returnfig=True,
        volume=False,
        warn_too_much_data=1000,
    )
    fig.text(
        0.5,
        0.5,
        "RANJHA",
        fontsize=70,
        color="gray",
        alpha=0.10,
        ha="center",
        va="center",
        rotation=25,
        weight="bold",
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ---------------- TELEGRAM ----------------
def tg_photo(img, caption):
    if not TG_TOKEN or not TG_CHAT:
        print("Telegram secrets missing")
        return False
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
        data={"chat_id": TG_CHAT, "caption": caption, "parse_mode": "HTML"},
        files={"photo": ("chart.png", img, "image/png")},
        timeout=60,
    )
    if r.status_code != 200:
        print("Telegram error:", r.text[:200])
    return r.status_code == 200


def tg_text(text):
    if not TG_TOKEN or not TG_CHAT:
        return
    requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        data={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"},
        timeout=30,
    )


def fmt(x):
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


# ---------------- STATE ----------------
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    cutoff = time.time() - 7 * 86400
    state = {k: v for k, v in state.items() if v > cutoff}
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def cooldown_ok(state, key, hours):
    return (time.time() - state.get(key, 0)) > hours * 3600


# ---------------- MAIN ----------------
def h4_job(item):
    sym, wk = item
    df = klines(sym, "Hour4", H4_CANDLES, 4 * 3600)
    if df is None:
        return None
    info = detect(df)
    if not info:
        return None
    info.update(wk)
    return sym, df, info


def main():
    syms = get_symbols()
    print(f"Pairs: {len(syms)}")
    if not syms:
        return
    state = load_state()

    pumped = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(weekly_job, s) for s in syms]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                r = None
            if r:
                pumped.append(r)
    pumped.sort(key=lambda x: -x[1]["w_pump"])
    print(f"Weekly pumped: {len(pumped)}")

    hits = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(h4_job, it) for it in pumped]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                r = None
            if r:
                hits.append(r)
    print(f"4H consolidation hits: {len(hits)}")

    hits.sort(key=lambda x: -x[2]["impulse"])

    sent = 0
    for sym, df, info in hits:
        if sent >= MAX_ALERTS:
            break
        key = f"{sym}|CONSOLIDATION"
        if not cooldown_ok(state, key, COOLDOWN_H):
            continue
        name = sym.replace("_", "")
        cap = (
            f"⏳ <b>{name}</b> | 4H CONSOLIDATION\n"
            f"Price: {fmt(info['price'])}\n"
            f"Weekly pump: +{info['w_pump']:.0f}% (holding {info['w_keep']*100:.0f}%)\n"
            f"4H impulse: +{info['impulse']:.0f}% | pullback {info['retrace']:.0f}%\n"
            f"Range: {fmt(info['range_low'])} - {fmt(info['range_high'])} ({info['range_pct']:.1f}%)\n"
            f"Candles since top: {info['after']} | EMA spread {info['spread']:.1f}%\n"
            f"https://www.mexc.com/futures/{sym}"
        )
        try:
            img = make_chart(sym, df, info)
            ok = tg_photo(img, cap)
        except Exception as e:
            print("chart error", sym, e)
            tg_text(cap)
            ok = True
        if ok:
            state[key] = time.time()
            sent += 1
            time.sleep(1)

    save_state(state)
    print(f"Alerts sent: {sent}")


if __name__ == "__main__":
    main()
