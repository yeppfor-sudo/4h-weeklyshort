# ============================================================
# MEXC PERPETUAL 4H SCANNER -> TELEGRAM (GitHub Actions)
# Pattern (CFG-style):
#   1) Downtrend  -> 2) Base at low -> 3) Volume-spike PUMP above EMA20
#   -> 4) RETEST of EMA 6/12/20 cluster -> 5) HOLD (higher low, no
#   close below EMA20) and MOVE UPSIDE
# Crypto pairs only | alert once per 24h per pair | ALERT ONLY IF SCORE >= 62
# Charts: 4H + 1D + 7D (MEXC light theme)
# Rate-limit safe: global throttle + retry passes for failed fetches
# DEBUG: reject reasons + near misses + API error counts.
#        TEST_SYMBOL=CFG_USDT runs one pair, sends nothing.
# ============================================================
import os, io, re, json, time, math, threading, warnings
from collections import Counter
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

# ===================== SETTINGS =====================
def envf(name, default):
    return type(default)(os.getenv(name, default))

MIN_SCORE           = envf("MIN_SCORE", 62.0)      # alert only if score >= this
USE_CLOSED_ONLY     = os.getenv("USE_CLOSED_ONLY", "true").lower() == "true"
WORKERS             = envf("WORKERS", 4)
REQ_INTERVAL        = envf("REQ_INTERVAL", 0.12)   # seconds between requests (global)
RETRY_PASSES        = envf("RETRY_PASSES", 2)      # extra passes for pairs with no data
MAX_ALERTS_PER_RUN  = envf("MAX_ALERTS_PER_RUN", 15)
COOLDOWN_HOURS      = envf("COOLDOWN_HOURS", 24)
EXTRA_EXCLUDE       = {x.strip().upper() for x in os.getenv("EXTRA_EXCLUDE", "").split(",") if x.strip()}
TEST_SYMBOL         = os.getenv("TEST_SYMBOL", "").strip().upper()   # e.g. CFG_USDT

# --- 1) Downtrend ---
DROP_LOOKBACK        = 120
MIN_DROP_PCT         = 30.0
MIN_BARS_HIGH_TO_LOW = 10
MIN_BARS_SINCE_LOW   = 4
MAX_BARS_SINCE_LOW   = 45

# --- 2) Base around the low (before the pump) ---
BASE_BARS_BEFORE     = 3
MAX_BASE_RANGE_PCT   = 25.0

# --- 3) Pump (volume spike) ---
MIN_BOUNCE_PCT       = 15.0
MIN_BARS_SINCE_PEAK  = 2
MAX_BARS_SINCE_PEAK  = 30
MIN_VOL_SPIKE        = 2.5
REQUIRE_GREEN_SPIKE  = True
REQUIRE_SPIKE_ABOVE_EMA20 = True

# --- 4) Retest ---
MIN_RETRACE          = 0.15
MAX_RETRACE          = 0.90
RETEST_TOUCH_PCT     = 4.0
MAX_EMA_SPREAD_PCT   = 3.5
MIN_DIST_EMA20_PCT   = -3.0
MAX_DIST_EMA20_PCT   = 6.0

# --- 5) Hold + move upside ---
HOLD_TOL_PCT         = 2.0
MIN_HIGHER_LOW_PCT   = 3.0
MIN_EMA20_SLOPE_PCT  = -1.5
REQUIRE_LAST_GREEN   = False
REQUIRE_ABOVE_EMA12  = True
REQUIRE_EMA6_OVER_20 = True

PLOT_CANDLES        = 100
STATE_FILE          = "sent_alerts.json"

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# timeframe label -> (MEXC interval, seconds per candle, candles to fetch, x-label format)
TF = {
    "4H": ("Hour4", 4 * 3600,   300, "%m-%d %H:%M"),
    "1D": ("Day1",  86400,      200, "%y-%m-%d"),
    "7D": ("Week1", 7 * 86400,  120, "%y-%m-%d"),
}

REASON_ORDER = ["no_data", "too_short", "low_window", "drop_small", "ema_stack_low",
                "bounce_small", "peak_age", "vol_base", "vol_spike", "spike_not_green",
                "spike_below_ema20", "base_range", "retrace", "retest_touch",
                "ema_spread", "dist_ema20", "hold_broke_ema20", "higher_low",
                "ema20_slope", "last_not_green", "below_ema12", "ema6_below_ema20"]
NEAR_MISS_FROM = REASON_ORDER.index("base_range")

# ===================== CRYPTO-ONLY FILTER =====================
NON_CRYPTO_BASES = {
    "XAU", "XAG", "GOLD", "SILVER", "OIL", "USOIL", "UKOIL", "BRENT", "WTI",
    "NATGAS", "COPPER", "PLATINUM", "PALLADIUM",
    "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "SPX", "SPX500", "NAS100",
    "NDX", "US30", "DJI", "DXY", "SP500", "NASDAQ",
    "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NFLX",
    "AMD", "INTC", "MSTR", "HOOD", "PLTR", "BABA", "UBER", "DIS", "NKE",
    "PYPL", "SHOP", "ORCL", "CRM", "ADBE", "QCOM", "AVGO", "TSM", "SPY", "QQQ",
    "IWM", "ARKK", "CRCL", "SNOW", "BA", "JPM", "V", "MA", "WMT", "KO", "PEP",
}
NON_CRYPTO_KW = ["stock", "forex", "commodit", "metal", "equit", "etf", "tradfi"]

def is_crypto(d):
    base = str(d.get("baseCoin", "")).upper()
    if base in NON_CRYPTO_BASES or base in EXTRA_EXCLUDE:
        return False
    if re.search(r"(STOCK|ETF)$", base):
        return False
    plates = " ".join(str(x) for x in (d.get("conceptPlate") or [])).lower()
    if any(k in plates for k in NON_CRYPTO_KW):
        return False
    return True

# ===================== MEXC API (rate-limit safe) =====================
BASE = "https://contract.mexc.com/api/v1/contract"
HDR = {"User-Agent": "Mozilla/5.0"}
SESSION = requests.Session()
SESSION.headers.update(HDR)
ERRS = Counter()
_rl_lock = threading.Lock()
_last_req = [0.0]

def throttle():
    with _rl_lock:
        wait = _last_req[0] + REQ_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()

def get_json(url, params=None, tries=5):
    for t in range(tries):
        throttle()
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 429 or r.status_code >= 500:
                ERRS[f"http_{r.status_code}"] += 1
                time.sleep(2 * (t + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            ERRS[type(e).__name__] += 1
            time.sleep(1 + t)
    return None

def get_symbols():
    j = get_json(f"{BASE}/detail")
    out, excluded = [], []
    if j and j.get("data"):
        for d in j["data"]:
            if d.get("quoteCoin") != "USDT" or d.get("state", 0) != 0:
                continue
            (out if is_crypto(d) else excluded).append(d["symbol"])
    print(f"Crypto pairs: {len(out)} | excluded non-crypto: {len(excluded)}")
    return sorted(set(out))

def get_klines(symbol, tf):
    interval, secs, n, _ = TF[tf]
    end = int(time.time())
    j = get_json(f"{BASE}/kline/{symbol}",
                 {"interval": interval, "start": end - n * secs, "end": end})
    if not j or not j.get("data") or not j["data"].get("time"):
        if j is not None:
            ERRS["empty_kline"] += 1
        return None
    d = j["data"]
    df = pd.DataFrame({
        "time": pd.to_datetime(d["time"], unit="s"),
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close"], "vol": d["vol"],
    }).astype({"open": float, "high": float, "low": float,
               "close": float, "vol": float})
    df = df.reset_index(drop=True)
    df["e6"]  = df.close.ewm(span=6,  adjust=False).mean()
    df["e12"] = df.close.ewm(span=12, adjust=False).mean()
    df["e20"] = df.close.ewm(span=20, adjust=False).mean()
    return df

# ===================== PATTERN DETECTION =====================
# returns (info or None, reason, metrics)
def detect(df):
    m = {}
    n = len(df)
    if n < 80:
        return None, "too_short", m
    o, h, l, c, v = (df.open.values, df.high.values, df.low.values,
                     df.close.values, df.vol.values)
    e6, e12, e20 = df.e6.values, df.e12.values, df.e20.values

    # ---------- 1) DOWNTREND -> LOW ----------
    w0 = max(0, n - 1 - MAX_BARS_SINCE_LOW)
    i_low = w0 + int(np.argmin(l[w0:]))
    since_low = n - 1 - i_low
    m["bars_since_low"] = since_low
    if since_low < MIN_BARS_SINCE_LOW or since_low > MAX_BARS_SINCE_LOW:
        return None, "low_window", m

    p0 = max(0, i_low - DROP_LOOKBACK)
    if i_low - p0 < 20:
        return None, "low_window", m
    i_high = p0 + int(np.argmax(h[p0:i_low + 1]))
    if i_low - i_high < MIN_BARS_HIGH_TO_LOW:
        return None, "drop_small", m
    drop = (h[i_high] - l[i_low]) / h[i_high] * 100
    m["drop%"] = round(float(drop), 1)
    if drop < MIN_DROP_PCT:
        return None, "drop_small", m
    if not (e6[i_low] < e12[i_low] < e20[i_low]):
        return None, "ema_stack_low", m

    # ---------- 3) PUMP ----------
    i_peak = i_low + int(np.argmax(h[i_low:]))
    if i_peak <= i_low:
        return None, "bounce_small", m
    bounce = (h[i_peak] / l[i_low] - 1) * 100
    since_peak = n - 1 - i_peak
    m["bounce%"] = round(float(bounce), 1); m["bars_since_peak"] = since_peak
    if bounce < MIN_BOUNCE_PCT:
        return None, "bounce_small", m
    if since_peak < MIN_BARS_SINCE_PEAK or since_peak > MAX_BARS_SINCE_PEAK:
        return None, "peak_age", m

    b0 = max(0, i_low - 30)
    base_vol = v[b0:i_low].mean() if i_low > b0 else 0
    if base_vol <= 0:
        return None, "vol_base", m
    s_end = min(n, i_peak + 2)
    i_spike = i_low + int(np.argmax(v[i_low:s_end]))
    spike = v[i_spike] / base_vol
    m["vol_spike_x"] = round(float(spike), 1)
    if spike < MIN_VOL_SPIKE:
        return None, "vol_spike", m
    if REQUIRE_GREEN_SPIKE and c[i_spike] <= o[i_spike]:
        return None, "spike_not_green", m
    if REQUIRE_SPIKE_ABOVE_EMA20 and c[i_spike] <= e20[i_spike]:
        return None, "spike_below_ema20", m

    # ---------- 2) BASE ----------
    bs = max(0, i_low - BASE_BARS_BEFORE)
    be = i_spike
    base_rng = 0.0
    if be - bs >= 3:
        lo_b = l[bs:be].min()
        base_rng = (h[bs:be].max() - lo_b) / lo_b * 100
    m["base_range%"] = round(float(base_rng), 1)
    if base_rng > MAX_BASE_RANGE_PCT:
        return None, "base_range", m

    # ---------- 4) RETEST ----------
    rng = h[i_peak] - l[i_low]
    retrace = (h[i_peak] - c[-1]) / rng if rng > 0 else 0
    m["retrace%"] = round(float(retrace) * 100, 1)
    if retrace < MIN_RETRACE or retrace > MAX_RETRACE:
        return None, "retrace", m

    k_rl = i_peak + int(np.argmin(l[i_peak:]))
    retest_low = l[k_rl]
    touch = (retest_low / e20[k_rl] - 1) * 100
    m["retest_touch%"] = round(float(touch), 2)
    if touch > RETEST_TOUCH_PCT:
        return None, "retest_touch", m

    ema_vals = [e6[-1], e12[-1], e20[-1]]
    spread = (max(ema_vals) - min(ema_vals)) / c[-1] * 100
    m["ema_spread%"] = round(float(spread), 2)
    if spread > MAX_EMA_SPREAD_PCT:
        return None, "ema_spread", m

    dist = (c[-1] - e20[-1]) / e20[-1] * 100
    m["dist_ema20%"] = round(float(dist), 2)
    if dist < MIN_DIST_EMA20_PCT or dist > MAX_DIST_EMA20_PCT:
        return None, "dist_ema20", m

    # ---------- 5) HOLD ----------
    if n - 1 > i_spike:
        if (c[i_spike + 1:] < e20[i_spike + 1:] * (1 - HOLD_TOL_PCT / 100)).any():
            return None, "hold_broke_ema20", m
    higher_low = (retest_low / l[i_low] - 1) * 100
    m["higher_low%"] = round(float(higher_low), 1)
    if higher_low < MIN_HIGHER_LOW_PCT:
        return None, "higher_low", m

    slope = (e20[-1] / e20[-7] - 1) * 100
    m["ema20_slope%"] = round(float(slope), 2)
    if slope < MIN_EMA20_SLOPE_PCT:
        return None, "ema20_slope", m

    # ---------- 5) MOVE UPSIDE ----------
    if REQUIRE_LAST_GREEN and c[-1] <= o[-1]:
        return None, "last_not_green", m
    if REQUIRE_ABOVE_EMA12 and c[-1] < e12[-1]:
        return None, "below_ema12", m
    if REQUIRE_EMA6_OVER_20 and e6[-1] < e20[-1] * 0.995:
        return None, "ema6_below_ema20", m

    score = (min(drop, 70) / 70 * 20 +
             min(bounce, 60) / 60 * 15 +
             min(spike, 10) / 10 * 20 +
             max(0, 1 - spread / MAX_EMA_SPREAD_PCT) * 20 +
             max(0, 1 - abs(dist) / MAX_DIST_EMA20_PCT) * 10 +
             min(higher_low, 15) / 15 * 15)
    info = dict(score=round(float(score), 1), price=float(c[-1]),
                drop_pct=round(float(drop), 1), bounce_pct=round(float(bounce), 1),
                vol_spike=round(float(spike), 1), retrace=round(float(retrace) * 100, 1),
                dist_ema20_pct=round(float(dist), 2), ema_spread_pct=round(float(spread), 2),
                ema20_slope_pct=round(float(slope), 2),
                base_range_pct=round(float(base_rng), 1),
                higher_low_pct=round(float(higher_low), 1),
                bars_since_low=int(since_low), bars_since_peak=int(since_peak),
                i_low=int(i_low), i_peak=int(i_peak), i_spike=int(i_spike))
    return info, "match", m

# ===================== CHART (MEXC light theme) =====================
UP, DOWN = "#2EBD85", "#F6465D"
C6, C12, C20 = "#D98FD0", "#F5A142", "#7B61FF"
VMA5, VMA10 = "#F7A8F0", "#6EC1FF"
GRID, TXT = "#EEF0F3", "#8A9099"

def fmt_p(p):
    dec = max(2, 3 - int(math.floor(math.log10(abs(p) if p else 1))))
    return f"{p:.{dec}f}"

def make_chart(symbol, df, tf, info=None):
    d = df.tail(PLOT_CANDLES).reset_index(drop=True)
    off = len(df) - len(d)
    x = np.arange(len(d))
    fig = plt.figure(figsize=(7.2, 7.4), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0]); axv = fig.add_subplot(gs[1], sharex=ax)
    for a in (ax, axv):
        a.set_facecolor("white"); a.grid(True, color=GRID, lw=0.8)
        a.yaxis.tick_right()
        for s in a.spines.values(): s.set_visible(False)
        a.tick_params(colors=TXT, labelsize=9, length=0)

    for i, r in d.iterrows():
        col = UP if r.close >= r.open else DOWN
        ax.vlines(i, r.low, r.high, color=col, lw=1)
        ax.add_patch(Rectangle((i - 0.32, min(r.open, r.close)), 0.64,
                               max(abs(r.close - r.open), r.close * 1e-6),
                               color=col, lw=0))
    ax.plot(x, d.e6, color=C6, lw=1.3)
    ax.plot(x, d.e12, color=C12, lw=1.3)
    ax.plot(x, d.e20, color=C20, lw=1.5)

    ih, il = int(d.high.values.argmax()), int(d.low.values.argmin())
    ax.annotate(fmt_p(d.high[ih]), (ih, d.high[ih]), xytext=(ih + 4, d.high[ih]),
                fontsize=9, color="#222",
                arrowprops=dict(arrowstyle="-", color="#222", lw=0.8))
    ax.annotate(fmt_p(d.low[il]), (il, d.low[il]), xytext=(max(il - 14, 0), d.low[il]),
                fontsize=9, color="#222",
                arrowprops=dict(arrowstyle="-", color="#222", lw=0.8))

    last = d.close.iloc[-1]
    ax.axhline(last, color=UP, ls="--", lw=0.9)
    ax.text(1.005, last, fmt_p(last), transform=ax.get_yaxis_transform(),
            fontsize=9, color=UP, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=UP))

    ax.text(0.01, 1.05, f"{symbol}  {tf}  (UTC)", transform=ax.transAxes,
            fontsize=11, weight="bold", color="#222")
    ax.text(0.01, 1.01, f"EMA6:{fmt_p(d.e6.iloc[-1])}", transform=ax.transAxes, fontsize=9, color=C6)
    ax.text(0.30, 1.01, f"EMA12:{fmt_p(d.e12.iloc[-1])}", transform=ax.transAxes, fontsize=9, color=C12)
    ax.text(0.60, 1.01, f"EMA20:{fmt_p(d.e20.iloc[-1])}", transform=ax.transAxes, fontsize=9, color=C20)

    if info and tf == "4H":
        for key, mk in (("i_low", "^"), ("i_peak", "v")):
            k = info[key] - off
            if 0 <= k < len(d):
                y = d.low[k] if key == "i_low" else d.high[k]
                ax.plot(k, y, marker=mk, color="#444", ms=5)
        ks = info.get("i_spike", -1) - off
        if 0 <= ks < len(d):
            axv.plot(ks, d.vol[ks], marker="o", color="#444", ms=4)

    vc = [UP if o <= c_ else DOWN for o, c_ in zip(d.open, d.close)]
    axv.bar(x, d.vol, color=vc, width=0.7)
    axv.plot(x, d.vol.rolling(5).mean(), color=VMA5, lw=1)
    axv.plot(x, d.vol.rolling(10).mean(), color=VMA10, lw=1)
    axv.set_ylim(0, d.vol.max() * 1.15 if d.vol.max() > 0 else 1)

    step = max(1, len(d) // 7)
    ticks = list(range(0, len(d), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([d.time[i].strftime(TF[tf][3]) for i in ticks])
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xlim(-1, len(d) + 2)
    ax.margins(y=0.08)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ===================== TELEGRAM =====================
def tg_post(method, data, files=None, tries=3):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    for t in range(tries):
        try:
            r = requests.post(url, data=data, files=files, timeout=60)
            if r.status_code == 429:
                wait = r.json().get("parameters", {}).get("retry_after", 5)
                time.sleep(wait + 1)
                if files:
                    for f in files.values(): f[1].seek(0)
                continue
            if r.ok:
                return True
            print("Telegram error:", r.status_code, r.text[:200])
            return False
        except Exception as e:
            print("Telegram exception:", e)
            time.sleep(2)
    return False

def send_alert(symbol, info, images):
    coin = symbol.replace("_USDT", "")
    caption = (
        f"🚨 <b>{symbol}</b> | 4H Base-Pump-Retest-Hold\n"
        f"Price: <code>{fmt_p(info['price'])}</code>  |  Score: {info['score']}\n"
        f"✅ Downtrend: -{info['drop_pct']}%\n"
        f"✅ Base: range {info['base_range_pct']}%\n"
        f"✅ Pump: +{info['bounce_pct']}%  |  Vol x{info['vol_spike']}\n"
        f"✅ Retest: retrace {info['retrace']}%  |  EMA spread {info['ema_spread_pct']}%\n"
        f"✅ Hold: higher low +{info['higher_low_pct']}%  |  EMA20 slope {info['ema20_slope_pct']}%\n"
        f"✅ Upside: dist EMA20 {info['dist_ema20_pct']}%\n"
        f"⏱ Bars since low: {info['bars_since_low']}   since peak: {info['bars_since_peak']}\n"
        f"Charts: {' / '.join(k for k, _ in images)}\n"
        f"https://futures.mexc.com/exchange/{coin}_USDT"
    )
    if len(images) == 1:
        name, buf = images[0]
        return tg_post("sendPhoto", {"chat_id": CHAT_ID, "caption": caption,
                                     "parse_mode": "HTML"},
                       {"photo": (f"{symbol}_{name}.png", buf, "image/png")})
    media, files = [], {}
    for i, (name, buf) in enumerate(images):
        key = f"p{i}"
        item = {"type": "photo", "media": f"attach://{key}"}
        if i == 0:
            item["caption"] = caption
            item["parse_mode"] = "HTML"
        media.append(item)
        files[key] = (f"{symbol}_{name}.png", buf, "image/png")
    return tg_post("sendMediaGroup",
                   {"chat_id": CHAT_ID, "media": json.dumps(media)}, files)

# ===================== STATE (24h dedupe) =====================
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)

# ===================== MAIN =====================
def scan_one(sym):
    df = get_klines(sym, "4H")
    if df is None:
        return sym, None, None, "no_data", {}
    det_df = df.iloc[:-1] if (USE_CLOSED_ONLY and len(df) > 1) else df
    info, reason, m = detect(det_df)
    return sym, df, info, reason, m

def run_pass(symbols, workers, outcome):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(scan_one, s) for s in symbols]):
            sym, df, info, reason, m = f.result()
            outcome[sym] = (df, info, reason, m)

def main():
    now = time.time()
    state = {k: v for k, v in load_state().items() if now - v < 48 * 3600}
    save_state(state)

    symbols = get_symbols()
    if TEST_SYMBOL:
        symbols = [TEST_SYMBOL]
        print(f"TEST MODE: only {TEST_SYMBOL}")
    print(f"Scanning {len(symbols)} pairs on 4H (workers={WORKERS}, interval={REQ_INTERVAL}s, min score={MIN_SCORE})...")

    outcome = {}
    run_pass(symbols, WORKERS, outcome)

    # retry pairs whose data could not be fetched
    for p in range(1, RETRY_PASSES + 1):
        missing = [s for s, o in outcome.items() if o[2] == "no_data"]
        if not missing:
            break
        print(f"Retry pass {p}: {len(missing)} pairs without data...")
        time.sleep(5)
        run_pass(missing, 2, outcome)

    reasons = Counter(o[2] for o in outcome.values())
    results, cache, near = [], {}, []
    for sym, (df, info, reason, m) in outcome.items():
        if TEST_SYMBOL:
            print(f"{sym}: {reason}  {m}")
        if info:
            cache[sym] = df
            results.append({"symbol": sym, **info})
        elif reason in REASON_ORDER and REASON_ORDER.index(reason) >= NEAR_MISS_FROM:
            near.append((REASON_ORDER.index(reason), sym, reason, m))

    print("\n--- REJECT REASONS ---")
    for k in REASON_ORDER + ["match"]:
        if reasons.get(k):
            print(f"  {k:20s} {reasons[k]}")

    print("\n--- API ERRORS ---")
    if ERRS:
        for k, v in ERRS.most_common():
            print(f"  {k:20s} {v}")
    else:
        print("  none")

    near.sort(key=lambda t: -t[0])
    if near:
        print(f"\n--- NEAR MISSES (passed pump stage, failed later) top {min(20, len(near))} ---")
        for _, sym, reason, m in near[:20]:
            print(f"  {sym:18s} failed: {reason:18s} {m}")

    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"\nMatches: {len(results)}")
    for r in results:
        tag = "" if r["score"] >= MIN_SCORE else f"  [below min score {MIN_SCORE}]"
        print(f"  {r['symbol']}  score={r['score']}  drop={r['drop_pct']}%  "
              f"bounce={r['bounce_pct']}%  vol x{r['vol_spike']}  "
              f"dist={r['dist_ema20_pct']}%  spread={r['ema_spread_pct']}%  "
              f"hl=+{r['higher_low_pct']}%{tag}")

    eligible = [r for r in results if r["score"] >= MIN_SCORE]
    print(f"Eligible for alert (score >= {MIN_SCORE}): {len(eligible)}")

    if TEST_SYMBOL or not eligible:
        return
    if not BOT_TOKEN or not CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets are missing.")
        raise SystemExit(1)

    sent = 0
    for r in eligible:
        sym = r["symbol"]
        if now - state.get(sym, 0) < COOLDOWN_HOURS * 3600:
            print(f"  skip {sym} (alerted within {COOLDOWN_HOURS}h)")
            continue
        if sent >= MAX_ALERTS_PER_RUN:
            print("Max alerts per run reached.")
            break
        images = [("4H", make_chart(sym, cache[sym], "4H", r))]
        for tf in ("1D", "7D"):
            dft = get_klines(sym, tf)
            if dft is not None and len(dft) >= 10:
                images.append((tf, make_chart(sym, dft, tf)))
        if send_alert(sym, r, images):
            state[sym] = time.time()
            save_state(state)
            sent += 1
            print(f"  sent {sym}")
        time.sleep(1.5)
    print(f"Alerts sent: {sent}")

if __name__ == "__main__":
    main()
