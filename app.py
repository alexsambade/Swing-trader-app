# Swing Trader: Multi-Timeframe Technical + News Sentiment (Streamlit)
# -------------------------------------------------------------------
# Technical: Trend (MA slopes + MACD), Momentum (RSI/BB), Breakouts, Volume confirmation,
# Risk (ADX + realized vol), 52-week tilt
# Patterns (1D / 1W / 1M): support/resistance, channels, wedges, simple head-and-shoulders
# Plus: Dark Mode, ATR-based entry/exit checklist, earnings flags
# NEW: Optional News Sentiment (NewsAPI headlines + FinBERT if available)
# Educational use only. Not investment advice.

from __future__ import annotations
import math, datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import requests

from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# ---------- Optional FinBERT (heavy). App still runs without it ----------
HAS_FINBERT = True
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except Exception:
    HAS_FINBERT = False

st.set_page_config(page_title="Swing Trader — Tech + News", layout="centered")
st.title("📈 Swing Trader — Market Mapper (Tech + News)")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Scan Settings")

    dark_mode = st.toggle("🌙 Dark Mode", value=True)
    plotly_tpl = "plotly_dark" if dark_mode else "plotly"

    universe_choice = st.selectbox("Universe", ["S&P 500", "NASDAQ-100", "Custom list"], index=1)
    custom_list = st.text_area("Custom tickers (comma/space separated)")
    lookback_years = st.slider("Price history (years)", 1, 5, 2)
    max_tickers = st.number_input("Max tickers (speed)", min_value=0, value=150, step=25, help="0 = unlimited")

    st.markdown("---")
    st.subheader("📊 Filters")
    min_price = st.number_input("Min price ($)", value=3.0, step=0.5)
    min_dollar_vol = st.number_input("Min avg dollar volume ($)", value=5_000_000, step=1_000_000)
    avoid_earn_days = st.slider("Avoid earnings within (days)", 0, 21, 7)

    st.markdown("---")
    st.subheader("🧮 Weights")
    w_trend = st.slider("Trend", 0.0, 1.0, 0.30, 0.05)
    w_momo  = st.slider("Momentum", 0.0, 1.0, 0.20, 0.05)
    w_break = st.slider("Breakout", 0.0, 1.0, 0.15, 0.05)
    w_volc  = st.slider("Volume confirm", 0.0, 1.0, 0.10, 0.05)
    w_risk  = st.slider("Risk (ADX & vol)", 0.0, 1.0, 0.10, 0.05)
    w_valu  = st.slider("52w tilt", 0.0, 1.0, 0.05, 0.05)
    w_mtf   = st.slider("Multi-TF patterns", 0.0, 1.0, 0.10, 0.05)
    w_news  = st.slider("News sentiment", 0.0, 1.0, 0.10, 0.05)

    st.caption(f"Sum of weights: **{w_trend + w_momo + w_break + w_volc + w_risk + w_valu + w_mtf + w_news:.2f}** (used as-is)")

    st.markdown("---")
    st.subheader("📰 News Sentiment (optional)")
    enable_news = st.toggle("Enable News Sentiment", value=False, help="Needs a NewsAPI key; FinBERT is optional.")
    news_api_key = st.text_input("NewsAPI.org key", type="password", help="Leave empty to disable headlines.")
    max_headlines = st.slider("Max headlines per ticker", 5, 30, 15, help="Fewer = faster")

    st.markdown("---")
    st.subheader("🏷️ Labels")
    buy_thr  = st.slider("BUY threshold", 0.0, 1.0, 0.60, 0.05)
    sell_thr = st.slider("SHORT threshold (≤)", -1.0, 0.0, -0.40, 0.05)

    st.markdown("---")
    st.subheader("💵 Risk & Sizing (checklist)")
    acct_equity = st.number_input("Account equity ($)", value=10_000, step=500)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 3.0, 1.0, 0.1)
    atr_mult = st.slider("Initial stop = ATR ×", 0.5, 4.0, 1.5, 0.1)

    run_scan = st.button("🚀 Run Scan", type="primary", use_container_width=True)

AS_OF = dt.date.today().isoformat()

# ---------------- Helpers ----------------
def _clean_name(x: str) -> str:
    bad = ["Inc.", "Inc", "Corporation", "Corp.", "Corp", "Ltd.", "Ltd", "PLC", "Plc",
           "Holdings", "Holding", "S.A.", "N.V.", "Class A", "Class B"]
    for b in bad:
        x = (x or "").replace(b, "")
    return " ".join(x.split()).strip()

@st.cache_data(show_spinner=False)
def sp500() -> pd.DataFrame:
    t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    df = t.rename(columns={"Symbol": "ticker", "Security":"name", "GICS Sector":"sector"})
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["name"] = df["name"].apply(_clean_name)
    return df[["ticker","name","sector"]]

@st.cache_data(show_spinner=False)
def nasdaq100() -> pd.DataFrame:
    tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    table = [t for t in tables if ("Ticker" in t.columns) or ("Ticker symbol" in t.columns)][0]
    col = "Ticker" if "Ticker" in table.columns else "Ticker symbol"
    name_col = "Company" if "Company" in table.columns else "Company name"
    df = table.rename(columns={col:"ticker", name_col:"name"})
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
    df["name"] = df["name"].apply(_clean_name); df["sector"] = ""
    return df[["ticker","name","sector"]]

def parse_custom(s: str) -> List[str]:
    s = (s or "").replace("\n"," ").replace(";", " ")
    return list(dict.fromkeys([p.strip().upper().replace(".","-") for p in s.replace(","," ").split() if p.strip()]))

@st.cache_data(show_spinner=False)
def fetch_history(tickers: List[str], years: int) -> Dict[str, pd.DataFrame]:
    if not tickers: return {}
    end = dt.date.today(); start = end - dt.timedelta(days=int(365.25*years))
    data = yf.download(tickers, start=start, end=end+dt.timedelta(days=1),
                       group_by="ticker", auto_adjust=True, threads=True)
    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.levels[0]:
                df = data[t].dropna(); df.columns = [c.lower() for c in df.columns]; out[t]=df
    else:
        df = data.dropna(); df.columns = [c.lower() for c in df.columns]; out[tickers[0]]=df
    return out

@st.cache_data(show_spinner=False)
def get_next_earnings(ticker: str) -> Optional[pd.Timestamp]:
    try:
        ek = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if ek is None or ek.empty: 
            return None
        today = pd.Timestamp.today(tz=ek.index.tz if ek.index.tz else None).normalize()
        future = ek.index[ek.index >= today]
        return future[0].tz_localize(None) if len(future)>0 else None
    except Exception:
        return None

# ---------------- Indicators ----------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    o,h,l,c,v = df["open"],df["high"],df["low"],df["close"],df["volume"]
    for w in [20,50,100,200]: df[f"sma{w}"]=c.rolling(w).mean()

    def slope(series, window):
        y = series.tail(window).values
        if len(y)<window: return np.nan
        x = np.arange(len(y))
        b1 = np.polyfit(x,y,1)[0]
        return float(b1/(np.mean(y)+1e-9))
    df["slope50"]  = df["sma50"].rolling(50).apply(lambda s: slope(pd.Series(s),50), raw=False)
    df["slope200"] = df["sma200"].rolling(200).apply(lambda s: slope(pd.Series(s),200), raw=False)

    rsi = RSIIndicator(close=c, window=14); df["rsi14"]=rsi.rsi()
    macd = MACD(close=c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]=macd.macd(); df["macd_signal"]=macd.macd_signal(); df["macd_hist"]=macd.macd_diff()

    adx = ADXIndicator(high=h, low=l, close=c, window=14); df["adx14"]=adx.adx()

    bb = BollingerBands(close=c, window=20, window_dev=2)
    df["bb_h"]=bb.bollinger_hband(); df["bb_l"]=bb.bollinger_lband()
    df["bb_p"]=(c-df["bb_l"])/((df["bb_h"]-df["bb_l"]).replace(0,np.nan))

    df["hi20"]=h.rolling(20).max(); df["lo20"]=l.rolling(20).min()
    df["hi55"]=h.rolling(55).max(); df["lo55"]=l.rolling(55).min()
    df["bo20_up"]=(c>df["hi20"].shift(1)).astype(int); df["bo55_up"]=(c>df["hi55"].shift(1)).astype(int)
    df["bo20_dn"]=(c<df["lo20"].shift(1)).astype(int); df["bo55_dn"]=(c<df["lo55"].shift(1)).astype(int)

    df["vol20"]=v.rolling(20).mean(); df["vol_ratio"]=v/(df["vol20"]+1)

    df["ret1"]=c.pct_change()
    df["vol_20d"]=df["ret1"].rolling(20).std()*np.sqrt(252)
    df["hi252"]=h.rolling(252).max(); df["lo252"]=l.rolling(252).min()
    df["pct_to_hi52"]=(c-df["hi252"])/(df["hi252"]+1e-9)

    atr = AverageTrueRange(high=h, low=l, close=c, window=14); df["atr14"]=atr.average_true_range()
    return df

def subscores_last(x: pd.Series) -> Dict[str,float]:
    trend  = np.clip(0.6*np.tanh(10*(x.get("slope50",0))) + 0.2*np.tanh(10*(x.get("slope200",0))) +
                     (0.2 if x.get("macd",0)>x.get("macd_signal",0) else -0.2), -1, 1)
    rsi    = x.get("rsi14",50.0)
    momo   = np.clip(0.7*((rsi-50)/50) + 0.3*((x.get("bb_p",0.5)-0.5)*2), -1, 1)
    bo     = np.clip(0.6*x.get("bo20_up",0)+0.4*x.get("bo55_up",0)-0.6*x.get("bo20_dn",0)-0.4*x.get("bo55_dn",0), -1, 1)
    volc   = float(np.clip(np.tanh(x.get("vol_ratio",1.0)-1.0), -1, 1))
    adx    = x.get("adx14",20.0)
    adx_pref = -abs((adx-27.5)/27.5)+1
    vol20  = x.get("vol_20d",0.25)
    risk   = np.clip(0.6*adx_pref + 0.4*(1-np.tanh(max(0,vol20-0.30)*3)), -1, 1)
    valu   = float(np.clip(-x.get("pct_to_hi52",0.0), -1, 1))
    return {"trend":trend,"momentum":momo,"breakout":bo,"vol_confirm":volc,"risk":risk,"valuation_proxy":valu}

# ------------- Levels & Patterns -------------
def pivot_points(series: pd.Series, left:int=3, right:int=3) -> pd.Series:
    piv = np.zeros(len(series), dtype=int)
    for i in range(left, len(series)-right):
        win = series[i-left:i+right+1]
        if series[i]==win.max(): piv[i]=+1
        if series[i]==win.min(): piv[i]=-1
    return pd.Series(piv, index=series.index)

def cluster_levels(pvals: pd.Series, tol: float=0.006, top:int=5) -> List[float]:
    vals = pvals.values
    if len(vals)==0: return []
    vals = np.sort(vals)
    clusters = []
    cluster = [vals[0]]
    for p in vals[1:]:
        if abs(p-cluster[-1])/cluster[-1] <= tol:
            cluster.append(p)
        else:
            clusters.append(np.mean(cluster)); cluster=[p]
    clusters.append(np.mean(cluster))
    return clusters[-top:]

def fit_trendline(idx: np.ndarray, y: np.ndarray) -> Tuple[float,float]:
    if len(y)<2: return 0.0, y[-1] if len(y) else 0.0
    x = (idx - idx.min())/(idx.max()-idx.min() + 1e-9)
    a,b = np.polyfit(x, y, 1)
    return float(a), float(b)

def pattern_scores(df: pd.DataFrame) -> Tuple[float, Dict[str,float], Dict[str,float]]:
    n = min(len(df), 180)
    d = df.tail(n).copy()
    close, high, low = d["close"], d["high"], d["low"]

    pivH = pivot_points(high, 3, 3); pivL = pivot_points(low, 3, 3)
    highs = high[pivH==+1]; lows = low[pivL==-1]
    res_levels = cluster_levels(highs, 0.006, 4)
    sup_levels = cluster_levels(lows,  0.006, 4)

    last = close.iloc[-1]
    near_sup = any(abs(last - s)/s < 0.006 for s in sup_levels)
    near_res = any(abs(last - r)/r < 0.006 for r in res_levels)
    lvl_score = (0.5 if near_sup else 0.0) + (-0.5 if near_res else 0.0)

    # Channel / wedge
    idx = np.arange(len(d))
    if len(highs)>=2 and len(lows)>=2:
        hi_idx = d.index.get_indexer(highs.index)
        lo_idx = d.index.get_indexer(lows.index)
        a_hi, b_hi = fit_trendline(idx[hi_idx], highs.values)
        a_lo, b_lo = fit_trendline(idx[lo_idx], lows.values)
        x1 = 1.0
        up_line  = a_hi*x1 + b_hi
        low_line = a_lo*x1 + b_lo
        width = max(1e-9, up_line - low_line)
        channel_up   = (a_hi>0) and (a_lo>0)
        channel_down = (a_hi<0) and (a_lo<0)
        x0 = 0.0
        width0 = (a_hi*x0 + b_hi) - (a_lo*x0 + b_lo)
        wedge = width < width0 * 0.7
    else:
        channel_up=channel_down=wedge=False

    ch_score = (0.25 if channel_up else 0.0) + (-0.25 if channel_down else 0.0) + (0.15 if wedge else 0.0)

    # Head & Shoulders (simple heuristic on last 3 peaks/troughs)
    hs_score = 0.0
    peaks = highs.tail(10).values
    if len(peaks) >= 3:
        L, H, R = peaks[-3], peaks[-2], peaks[-1]
        if H > L*1.01 and H > R*1.01 and abs(L-R)/max(L,R) < 0.03: hs_score -= 0.3
    troughs = lows.tail(10).values
    if len(troughs) >= 3:
        L, H, R = troughs[-3], troughs[-2], troughs[-1]
        if H < L*0.99 and H < R*0.99 and abs(L-R)/max(L,R) < 0.03: hs_score += 0.3

    breakout = 0.0
    if res_levels and last > max(res_levels) * 1.003: breakout += 0.25
    if sup_levels and last < min(sup_levels) * 0.997: breakout -= 0.25

    parts = {"levels": lvl_score, "channel_wedge": ch_score, "head_shoulders": hs_score, "level_breakout": breakout}
    score = float(np.clip(sum(parts.values()), -1, 1))
    ctx = {"last": float(last), "supports": sup_levels, "resistances": res_levels,
           "channel_up": bool(channel_up), "channel_down": bool(channel_down), "wedge": bool(wedge)}
    return score, parts, ctx

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.concat([o,h,l,c,v], axis=1); out.columns=["open","high","low","close","volume"]
    return out.dropna()

def mtf_score(df_daily: pd.DataFrame) -> Tuple[float, Dict[str,float], Dict[str,Dict]]:
    d1 = df_daily.copy()
    w1 = resample_ohlc(df_daily, "W")
    m1 = resample_ohlc(df_daily, "M")
    s1, p1, c1 = pattern_scores(d1)
    s2, p2, c2 = pattern_scores(w1)
    s3, p3, c3 = pattern_scores(m1)
    score = float(np.clip(0.5*s1 + 0.3*s2 + 0.2*s3, -1, 1))
    parts = {"1D": s1, "1W": s2, "1M": s3}
    ctx = {"1D": c1, "1W": c2, "1M": c3}
    return score, parts, ctx

# ---------------- News sentiment ----------------
class FinBERT:
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        self.ok = HAS_FINBERT
        if not HAS_FINBERT:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def score_texts(self, texts: List[str]) -> float:
        if (not texts) or (not self.ok): return 0.0
        with torch.no_grad():
            pos = neg = 0.0; n = 0
            for t in texts[:20]:
                toks = self.tokenizer(t[:512], return_tensors="pt", truncation=True).to(self.device)
                logits = self.model(**toks).logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [neutral, positive, negative]
                pos += float(probs[1]); neg += float(probs[2]); n += 1
        n = max(n, 1)
        return float((pos/n) - (neg/n))  # [-1,1]-ish

@st.cache_data(show_spinner=False)
def newsapi_headlines(query: str, api_key: str, days: int = 7, max_items: int = 15) -> List[str]:
    if not api_key: return []
    base = "https://newsapi.org/v2/everything"
    from_date = (dt.datetime.utcnow() - dt.timedelta(days=days)).date().isoformat()
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": max_items,
        "apiKey": api_key,
    }
    try:
        r = requests.get(base, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        out = []
        for a in js.get("articles", []):
            t = (a.get("title") or "").strip()
            d = (a.get("description") or "").strip()
            txt = (t + " — " + d).strip(" —")
            if txt:
                out.append(txt)
        return out
    except Exception:
        return []

def composite(sub: Dict[str,float], mtf: float, news: float, weights: Dict[str,float]) -> float:
    return float(np.clip(
        weights["trend"]*sub["trend"] +
        weights["momentum"]*sub["momentum"] +
        weights["breakout"]*sub["breakout"] +
        weights["vol_confirm"]*sub["vol_confirm"] +
        weights["risk"]*sub["risk"] +
        weights["valuation_proxy"]*sub["valuation_proxy"] +
        weights["mtf"]*mtf +
        weights["news"]*news, -1, 1))

def label_for(score: float, buy_thr: float, sell_thr: float) -> str:
    if score >= buy_thr: return "BUY"
    if score <= sell_thr: return "SHORT"
    return "HOLD"

@dataclass
class Idea:
    ticker:str; name:str; sector:str; price:float; dollar_vol:float
    sub:Dict[str,float]; mtf:float; news:float; score:float; label:str
    mtf_parts:Dict[str,float]; mtf_ctx:Dict[str,Dict]; next_earn: Optional[pd.Timestamp]

# ---------------- Run Scan ----------------
if run_scan:
    # Universe
    if universe_choice=="S&P 500":
        uni = sp500()
    elif universe_choice=="NASDAQ-100":
        uni = nasdaq100()
    else:
        tickers = parse_custom(custom_list)
        if not tickers:
            st.error("Enter at least one ticker in Custom list."); st.stop()
        uni = pd.DataFrame({"ticker": tickers, "name": tickers, "sector": ""})

    if max_tickers>0: uni = uni.head(int(max_tickers))

    with st.status("⏬ Fetching price history…", expanded=False):
        data = fetch_history(uni["ticker"].tolist(), lookback_years)

    finbert = FinBERT() if (enable_news and news_api_key and HAS_FINBERT) else None

    ideas: List[Idea] = []
    pbar = st.progress(0.0)
    today = pd.Timestamp.today().normalize()
    for i,(idx,row) in enumerate(uni.reset_index(drop=True).iterrows(), start=1):
        pbar.progress(i/len(uni))
        tkr, nm, sec = row["ticker"], row["name"], row["sector"]
        if tkr not in data or data[tkr].empty: continue
        df = compute_features(data[tkr].copy()).dropna()
        if df.empty: continue
        last = df.iloc[-1]
        price = float(last["close"]); vol20=float(last.get("vol20", np.nan))
        dollar_vol = price*(vol20 if not math.isnan(vol20) else 0.0)
        if price < min_price or dollar_vol < min_dollar_vol: continue

        sub = subscores_last(last)
        mtf, mtf_parts, mtf_ctx = mtf_score(df)

        # Earnings date + small safety penalty if close
        next_earn = get_next_earnings(tkr)
        earns_penalty = 0.0
        if next_earn is not None and avoid_earn_days>0:
            days = (next_earn.normalize() - today).days
            if 0 <= days <= avoid_earn_days:
                earns_penalty = -0.2

        # News sentiment
        news_score = 0.0
        if enable_news and news_api_key:
            q = f"\"{nm}\" OR {tkr} stock"
            headlines = newsapi_headlines(q, news_api_key, days=7, max_items=int(max_headlines))
            if finbert and finbert.ok:
                news_score = finbert.score_texts(headlines)
            else:
                # Lightweight proxy without FinBERT: +0.1 if many obviously positive words, -0.1 if negative (very crude)
                pos_kw = ("beats", "surges", "record", "strong", "upgrade", "bullish", "outperforms")
                neg_kw = ("misses", "plunge", "downgrade", "lawsuit", "bearish", "weak", "recession")
                pos = sum(any(k in h.lower() for k in pos_kw) for h in headlines)
                neg = sum(any(k in h.lower() for k in neg_kw) for h in headlines)
                if pos+neg > 0:
                    news_score = float(np.clip((pos - neg) / (pos + neg), -1, 1)) * 0.5  # scale down

        weights = {"trend": w_trend,"momentum": w_momo,"breakout": w_break,
                   "vol_confirm": w_volc,"risk": w_risk,"valuation_proxy": w_valu,
                   "mtf": w_mtf, "news": w_news}
        score = composite(sub, mtf, news_score, weights) + earns_penalty
        score = float(np.clip(score, -1, 1))
        lab = label_for(score, buy_thr, sell_thr)

        ideas.append(Idea(tkr, nm, sec, price, dollar_vol, sub, mtf, news_score, score, lab, mtf_parts, mtf_ctx, next_earn))

    if not ideas:
        st.warning("No symbols passed your filters. Loosen volume/price floors.")
        st.stop()

    df_out = pd.DataFrame([{
        "ticker":I.ticker,"name":I.name,"sector":I.sector,"price":round(I.price,2),
        "avg_dollar_vol":int(I.dollar_vol),"score":round(I.score,3),"label":I.label,
        "trend":round(I.sub["trend"],3),"momentum":round(I.sub["momentum"],3),
        "breakout":round(I.sub["breakout"],3),"vol_confirm":round(I.sub["vol_confirm"],3),
        "risk":round(I.sub["risk"],3),"tilt_52w":round(I.sub["valuation_proxy"],3),
        "mtf":round(I.mtf,3),"news":round(I.news,3),
        "earnings": I.next_earn.date().isoformat() if I.next_earn is not None else ""
    } for I in ideas]).sort_values("score", ascending=False).reset_index(drop=True)

    st.success(f"Scan complete — {len(df_out)} ideas")
    st.dataframe(df_out, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download CSV", df_out.to_csv(index=False), "swing_signals.csv", "text/csv")

    st.markdown("---")
    st.subheader("🖼️ Chart & MTF Patterns")
    pick = st.selectbox("Select a ticker", df_out["ticker"].tolist())
    I = next(x for x in ideas if x.ticker==pick)
    dfd = compute_features(data[pick].copy()).dropna()

    # Price + SMAs + BB
    fig = go.Figure()
    fig.update_layout(template=plotly_tpl, height=480, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
    fig.add_trace(go.Candlestick(x=dfd.index, open=dfd["open"], high=dfd["high"], low=dfd["low"], close=dfd["close"], name="Price"))
    for w in [20,50,100,200]:
        fig.add_trace(go.Scatter(x=dfd.index, y=dfd[f"sma{w}"], name=f"SMA{w}", mode="lines"))
    fig.add_trace(go.Scatter(x=dfd.index, y=dfd["bb_h"], name="BB High", mode="lines"))
    fig.add_trace(go.Scatter(x=dfd.index, y=dfd["bb_l"], name="BB Low", mode="lines"))
    st.plotly_chart(fig, use_container_width=True)

    # RSI / MACD
    rsi = RSIIndicator(close=dfd["close"], window=14).rsi()
    st.plotly_chart(px.line(x=dfd.index, y=rsi, labels={"x":"Date","y":"RSI(14)"}, title="RSI(14)",
                            template=plotly_tpl).update_layout(height=260, margin=dict(l=10,r=10,t=30,b=10)),
                    use_container_width=True)
    macd = MACD(close=dfd["close"], window_slow=26, window_fast=12, window_sign=9)
    macd_df = pd.DataFrame({"date": dfd.index, "macd": macd.macd(), "signal": macd.macd_signal(), "hist": macd.macd_diff()})
    fig2 = go.Figure()
    fig2.update_layout(template=plotly_tpl, height=300, margin=dict(l=10,r=10,t=30,b=10))
    fig2.add_trace(go.Scatter(x=macd_df["date"], y=macd_df["macd"], name="MACD"))
    fig2.add_trace(go.Scatter(x=macd_df["date"], y=macd_df["signal"], name="Signal"))
    fig2.add_trace(go.Bar(x=macd_df["date"], y=macd_df["hist"], name="Hist"))
    st.plotly_chart(fig2, use_container_width=True)

    # --- Entry/Exit checklist & sizing ---
    last = dfd.iloc[-1]
    atr14 = float(last["atr14"])
    entry = float(last["close"])
    stop = max(0.01, entry - atr14*atr_mult)
    risk_per_share = entry - stop
    dollars_risk = acct_equity * (risk_pct/100.0)
    size = int(dollars_risk / max(0.01, risk_per_share))
    st.markdown("### ✅ Entry/Exit checklist")
    st.write(f"- Price: **${entry:.2f}**  \n"
             f"- ATR(14): **${atr14:.2f}**  \n"
             f"- Suggested initial stop (ATR × {atr_mult}): **${stop:.2f}**  \n"
             f"- Position size for {risk_pct:.1f}% risk and ${acct_equity:,} equity: **{size} shares** "
             f"(risk/share ${risk_per_share:.2f}, total risk ≈ ${dollars_risk:.0f})")
    if I.next_earn is not None:
        days = (I.next_earn - pd.Timestamp.today().normalize()).days
        st.warning(f"📅 Earnings: **{I.next_earn.date().isoformat()}** (in {days} days). Consider avoiding new swings near earnings.")

    # MTF context
    ctx = I.mtf_ctx
    def fmt_levels(xs): return ", ".join([f"{x:.2f}" for x in xs]) if xs else "—"
    with st.expander("🔎 MTF levels & patterns"):
        for tf in ["1D","1W","1M"]:
            c = ctx[tf]
            st.write(f"**{tf}** — Supports: {fmt_levels(c['supports'])} | "
                     f"Resistances: {fmt_levels(c['resistances'])} | "
                     f"Channel↑ {c['channel_up']} | Channel↓ {c['channel_down']} | Wedge {c['wedge']}")

else:
    st.info("Set preferences and tap **Run Scan**. Enable News and paste a NewsAPI key if you want sentiment included.")
    st.caption("Dark Mode affects charts; earnings near your window reduce score slightly.")
