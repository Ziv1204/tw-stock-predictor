
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(
    page_title="台股智慧分析站",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# 基本工具
# -----------------------------
def normalize_ticker(code: str) -> str:
    code = str(code).strip().upper()
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code
    if code.isdigit():
        return code + ".TW"
    return code


def display_code(ticker: str) -> str:
    return ticker.replace(".TW", "").replace(".TWO", "")


def safe_div(a, b):
    if isinstance(b, pd.Series):
        b = b.replace(0, np.nan)
    return a / b


@st.cache_data(ttl=900, show_spinner=False)
def download_daily(ticker: str, period="2y"):
    df = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


@st.cache_data(ttl=300, show_spinner=False)
def download_intraday(ticker: str, interval="5m", period="5d"):
    df = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


@st.cache_data(ttl=900, show_spinner=False)
def download_market(period="2y"):
    df = yf.download(
        "^TWII",
        period=period,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


def rsi(s, period=14):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = safe_div(ag, al)
    return 100 - 100/(1+rs)


def add_pressure_features(x: pd.DataFrame) -> pd.DataFrame:
    hl = (x["High"] - x["Low"]).replace(0, np.nan)
    mfm = ((x["Close"] - x["Low"]) - (x["High"] - x["Close"])) / hl
    x["money_flow_volume"] = mfm * x["Volume"]

    x["cmf20"] = (
        x["money_flow_volume"].rolling(20).sum()
        / x["Volume"].rolling(20).sum().replace(0, np.nan)
    )

    direction = np.sign(x["Close"].diff()).fillna(0)
    x["obv"] = (direction * x["Volume"]).cumsum()
    x["obv_change_5"] = x["obv"].diff(5)

    up_vol = x["Volume"].where(x["Close"] > x["Close"].shift(1), 0.0)
    down_vol = x["Volume"].where(x["Close"] < x["Close"].shift(1), 0.0)
    x["up_down_vol_ratio_10"] = (
        up_vol.rolling(10).sum()
        / down_vol.rolling(10).sum().replace(0, np.nan)
    )

    x["close_location"] = 2*(x["Close"]-x["Low"]) / hl - 1
    ret = x["Close"].pct_change(fill_method=None)
    x["pv_pressure"] = ret * np.log1p(x["Volume"])
    return x


def make_daily_features(stock, market):
    x = stock.copy()

    x["ret_1"] = x["Close"].pct_change(fill_method=None)
    x["ret_3"] = x["Close"].pct_change(3, fill_method=None)
    x["ret_5"] = x["Close"].pct_change(5, fill_method=None)
    x["ret_10"] = x["Close"].pct_change(10, fill_method=None)

    x["vol_5"] = x["ret_1"].rolling(5).std()
    x["vol_10"] = x["ret_1"].rolling(10).std()
    x["vol_20"] = x["ret_1"].rolling(20).std()

    for n in [5, 10, 20, 60]:
        ma = x["Close"].rolling(n).mean()
        x[f"close_to_ma_{n}"] = safe_div(x["Close"], ma) - 1

    x["rsi_14"] = rsi(x["Close"])

    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    sig = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - sig

    low9 = x["Low"].rolling(9).min()
    high9 = x["High"].rolling(9).max()
    rsv = safe_div(x["Close"] - low9, high9 - low9) * 100
    x["k"] = rsv.ewm(alpha=1/3, adjust=False).mean()
    x["d"] = x["k"].ewm(alpha=1/3, adjust=False).mean()

    mid = x["Close"].rolling(20).mean()
    std = x["Close"].rolling(20).std()
    upper = mid + 2*std
    lower = mid - 2*std
    x["bb_width"] = safe_div(upper-lower, mid)
    x["bb_pos"] = safe_div(x["Close"]-lower, upper-lower)

    x["range_pct"] = safe_div(x["High"]-x["Low"], x["Close"])
    x["open_gap"] = safe_div(x["Open"], x["Close"].shift(1)) - 1
    x["close_pos"] = safe_div(x["Close"]-x["Low"], x["High"]-x["Low"])
    x["vol_ratio_20"] = safe_div(x["Volume"], x["Volume"].rolling(20).mean())

    x = add_pressure_features(x)

    m = market[["Close"]].rename(columns={"Close":"market_close"}).copy()
    m["market_ret_1"] = m["market_close"].pct_change(fill_method=None)
    m["market_ret_5"] = m["market_close"].pct_change(5, fill_method=None)
    m["market_ma20"] = m["market_close"].rolling(20).mean()
    m["market_to_ma20"] = safe_div(m["market_close"], m["market_ma20"]) - 1

    x = x.join(
        m[["market_ret_1","market_ret_5","market_to_ma20"]],
        how="left"
    )
    x["relative_strength_5"] = x["ret_5"] - x["market_ret_5"]

    return x.replace([np.inf, -np.inf], np.nan)


DAILY_FEATURES = [
    "ret_1","ret_3","ret_5","ret_10",
    "vol_5","vol_10","vol_20",
    "close_to_ma_5","close_to_ma_10","close_to_ma_20","close_to_ma_60",
    "rsi_14","macd","macd_hist","k","d","bb_width","bb_pos",
    "range_pct","open_gap","close_pos","vol_ratio_20",
    "cmf20","obv_change_5","up_down_vol_ratio_10","close_location","pv_pressure",
    "market_ret_1","market_ret_5","market_to_ma20","relative_strength_5"
]


def pressure_score(row):
    score = 0.0

    cmf = row.get("cmf20", np.nan)
    if pd.notna(cmf):
        score += np.clip(cmf * 120, -25, 25)

    cl = row.get("close_location", np.nan)
    if pd.notna(cl):
        score += np.clip(cl * 20, -20, 20)

    obv = row.get("obv_change_5", np.nan)
    if pd.notna(obv):
        score += 12 if obv > 0 else -12 if obv < 0 else 0

    uv = row.get("up_down_vol_ratio_10", np.nan)
    if pd.notna(uv):
        if uv > 1.5:
            score += 18
        elif uv > 1.1:
            score += 8
        elif uv < 0.67:
            score -= 18
        elif uv < 0.9:
            score -= 8

    return float(np.clip(score, -100, 100))


# -----------------------------
# 首頁候選股掃描
# -----------------------------
DEFAULT_WATCHLIST = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2454.TW": "聯發科",
    "2308.TW": "台達電",
    "2382.TW": "廣達",
    "3231.TW": "緯創",
    "2357.TW": "華碩",
    "2376.TW": "技嘉",
    "2881.TW": "富邦金",
    "2882.TW": "國泰金",
    "2886.TW": "兆豐金",
    "2891.TW": "中信金",
    "2603.TW": "長榮",
    "2609.TW": "陽明",
    "2615.TW": "萬海",
    "2303.TW": "聯電",
    "3711.TW": "日月光投控",
    "3034.TW": "聯詠",
    "6669.TW": "緯穎",
    "3017.TW": "奇鋐",
}


@st.cache_data(ttl=900, show_spinner=False)
def quick_scan_one(ticker: str):
    stock = download_daily(ticker, "6mo")
    market = download_market("6mo")
    if stock.empty or len(stock) < 70:
        return None

    df = make_daily_features(stock, market)
    valid = df.dropna(subset=[
        "rsi_14","macd_hist","vol_ratio_20","cmf20",
        "close_to_ma_20","ret_5","relative_strength_5",
        "close_location"
    ])
    if valid.empty:
        return None

    r = valid.iloc[-1]
    score = 50.0

    # 趨勢
    score += np.clip(r["close_to_ma_20"] * 350, -12, 12)
    score += np.clip(r["ret_5"] * 250, -10, 10)
    score += np.clip(r["relative_strength_5"] * 250, -10, 10)

    # MACD
    if r["macd_hist"] > 0:
        score += 8
    else:
        score -= 8

    # RSI：避免過熱也避免太弱
    if 45 <= r["rsi_14"] <= 68:
        score += 8
    elif r["rsi_14"] > 78:
        score -= 8
    elif r["rsi_14"] < 35:
        score -= 5

    # 量能
    if r["vol_ratio_20"] >= 1.5:
        score += 8
    elif r["vol_ratio_20"] >= 1.1:
        score += 4

    # 資金流
    score += np.clip(r["cmf20"] * 60, -10, 10)
    score += np.clip(r["close_location"] * 5, -5, 5)

    score = float(np.clip(score, 0, 100))
    pscore = pressure_score(r)

    return {
        "ticker": ticker,
        "close": float(r["Close"]),
        "score": score,
        "pressure": pscore,
        "rsi": float(r["rsi_14"]),
        "vol_ratio": float(r["vol_ratio_20"]),
        "ret5": float(r["ret_5"]) * 100,
    }


@st.cache_data(ttl=900, show_spinner=False)
def scan_watchlist():
    rows = []
    for ticker, name in DEFAULT_WATCHLIST.items():
        try:
            r = quick_scan_one(ticker)
            if r:
                r["name"] = name
                rows.append(r)
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["score", "pressure"], ascending=False).reset_index(drop=True)


# -----------------------------
# 詳細分析：未來多日
# -----------------------------
@st.cache_data(ttl=1200, show_spinner=False)
def analyze_daily(ticker: str, horizon: int, period: str):
    stock = download_daily(ticker, period)
    market = download_market(period)

    if stock.empty or len(stock) < 200:
        raise ValueError("歷史資料不足。")

    df = make_daily_features(stock, market)

    future_high = pd.concat(
        [df["High"].shift(-i) for i in range(1, horizon+1)],
        axis=1
    ).max(axis=1)
    future_low = pd.concat(
        [df["Low"].shift(-i) for i in range(1, horizon+1)],
        axis=1
    ).min(axis=1)

    df["target_high_ret"] = safe_div(future_high, df["Close"]) - 1
    df["target_low_ret"] = safe_div(future_low, df["Close"]) - 1
    df = df.replace([np.inf, -np.inf], np.nan)

    train = df.dropna(
        subset=DAILY_FEATURES + ["target_high_ret","target_low_ret"]
    ).copy()
    latest = df.dropna(subset=DAILY_FEATURES).iloc[[-1]].copy()

    if len(train) < 150:
        raise ValueError("可用訓練資料不足。")

    X = train[DAILY_FEATURES].astype("float64")
    yh = train["target_high_ret"].astype("float64")
    yl = train["target_low_ret"].astype("float64")
    X_now = latest[DAILY_FEATURES].astype("float64")

    cv = TimeSeriesSplit(n_splits=3)
    he, le = [], []

    for tr, va in cv.split(X):
        mh = ExtraTreesRegressor(
            n_estimators=60, min_samples_leaf=4,
            max_features=0.7, random_state=42, n_jobs=-1
        )
        ml = ExtraTreesRegressor(
            n_estimators=60, min_samples_leaf=4,
            max_features=0.7, random_state=43, n_jobs=-1
        )
        mh.fit(X.iloc[tr], yh.iloc[tr])
        ml.fit(X.iloc[tr], yl.iloc[tr])
        he.append(mean_absolute_error(yh.iloc[va], mh.predict(X.iloc[va])))
        le.append(mean_absolute_error(yl.iloc[va], ml.predict(X.iloc[va])))

    mh = ExtraTreesRegressor(
        n_estimators=100, min_samples_leaf=4,
        max_features=0.7, random_state=42, n_jobs=-1
    )
    ml = ExtraTreesRegressor(
        n_estimators=100, min_samples_leaf=4,
        max_features=0.7, random_state=43, n_jobs=-1
    )
    mh.fit(X, yh)
    ml.fit(X, yl)

    hret = float(mh.predict(X_now)[0])
    lret = float(ml.predict(X_now)[0])

    close = float(latest["Close"].iloc[0])
    ph = max(close * (1+hret), close)
    pl = min(close * (1+lret), close)

    row = latest.iloc[0]
    pscore = pressure_score(row)

    return {
        "last_close": close,
        "pred_high": ph,
        "pred_low": pl,
        "high_mae": float(np.mean(he)*100),
        "low_mae": float(np.mean(le)*100),
        "pressure_score": pscore,
        "pressure_label": (
            "買壓偏強" if pscore >= 25 else
            "買壓略強" if pscore >= 8 else
            "賣壓偏強" if pscore <= -25 else
            "賣壓略強" if pscore <= -8 else
            "多空接近"
        ),
        "cmf20": float(row["cmf20"]),
        "up_down_vol_ratio_10": float(row["up_down_vol_ratio_10"]),
        "rsi14": float(row["rsi_14"]),
        "macd_hist": float(row["macd_hist"]),
        "chart": stock[["Close"]].tail(180),
    }


# -----------------------------
# 詳細分析：當日盤中
# -----------------------------
def make_intraday_training(df: pd.DataFrame):
    x = df.copy()
    if x.empty:
        return x

    idx = x.index
    try:
        local_dates = idx.tz_convert("Asia/Taipei").date
    except Exception:
        local_dates = idx.date

    x["trade_date"] = local_dates
    x["bar_return"] = x["Close"].pct_change(fill_method=None)
    x["bar_range"] = safe_div(x["High"]-x["Low"], x["Close"])

    typical = (x["High"] + x["Low"] + x["Close"]) / 3
    cum_pv = (typical * x["Volume"]).groupby(x["trade_date"]).cumsum()
    cum_v = x["Volume"].groupby(x["trade_date"]).cumsum().replace(0, np.nan)
    x["vwap_proxy"] = cum_pv / cum_v

    x["day_open"] = x.groupby("trade_date")["Open"].transform("first")
    x["day_high_so_far"] = x.groupby("trade_date")["High"].cummax()
    x["day_low_so_far"] = x.groupby("trade_date")["Low"].cummin()

    x["ret_from_open"] = safe_div(x["Close"], x["day_open"]) - 1
    x["high_from_open"] = safe_div(x["day_high_so_far"], x["day_open"]) - 1
    x["low_from_open"] = safe_div(x["day_low_so_far"], x["day_open"]) - 1
    x["dist_vwap"] = safe_div(x["Close"], x["vwap_proxy"]) - 1
    x["rv_6"] = x["bar_return"].rolling(6).std()
    x["vol_ratio_6"] = safe_div(x["Volume"], x["Volume"].rolling(6).mean())

    hl = (x["High"] - x["Low"]).replace(0, np.nan)
    x["bar_close_location"] = 2*(x["Close"]-x["Low"])/hl - 1
    x["bar_pressure"] = x["bar_close_location"] * np.log1p(x["Volume"])
    x["pressure_6"] = x["bar_pressure"].rolling(6).mean()

    x["bar_no"] = x.groupby("trade_date").cumcount()
    max_bar = x.groupby("trade_date")["bar_no"].transform("max").replace(0, np.nan)
    x["day_progress"] = safe_div(x["bar_no"], max_bar)

    final_high = x.groupby("trade_date")["High"].transform("max")
    final_low = x.groupby("trade_date")["Low"].transform("min")
    x["target_remaining_high_ret"] = safe_div(final_high, x["Close"]) - 1
    x["target_remaining_low_ret"] = safe_div(final_low, x["Close"]) - 1

    return x.replace([np.inf, -np.inf], np.nan)


INTRADAY_FEATURES = [
    "bar_return","bar_range","ret_from_open","high_from_open","low_from_open",
    "dist_vwap","rv_6","vol_ratio_6","bar_close_location","pressure_6","day_progress"
]


@st.cache_data(ttl=240, show_spinner=False)
def analyze_today(ticker: str):
    intra = download_intraday(ticker, "5m", "5d")
    if intra.empty or len(intra) < 40:
        raise ValueError("目前沒有足夠的 5 分鐘盤中資料。")

    df = make_intraday_training(intra)
    usable = df.dropna(
        subset=INTRADAY_FEATURES + [
            "target_remaining_high_ret","target_remaining_low_ret"
        ]
    ).copy()

    latest_candidates = df.dropna(subset=INTRADAY_FEATURES)
    if len(usable) < 30 or latest_candidates.empty:
        raise ValueError("盤中可用資料不足。")

    latest = latest_candidates.iloc[[-1]].copy()
    X = usable[INTRADAY_FEATURES].astype("float64")
    yh = usable["target_remaining_high_ret"].astype("float64")
    yl = usable["target_remaining_low_ret"].astype("float64")

    mh = ExtraTreesRegressor(
        n_estimators=80, min_samples_leaf=3,
        max_features=0.8, random_state=51, n_jobs=-1
    )
    ml = ExtraTreesRegressor(
        n_estimators=80, min_samples_leaf=3,
        max_features=0.8, random_state=52, n_jobs=-1
    )
    mh.fit(X, yh)
    ml.fit(X, yl)

    xn = latest[INTRADAY_FEATURES].astype("float64")
    hret = float(mh.predict(xn)[0])
    lret = float(ml.predict(xn)[0])

    close = float(latest["Close"].iloc[0])
    current_high = float(latest["day_high_so_far"].iloc[0])
    current_low = float(latest["day_low_so_far"].iloc[0])

    pred_high = max(current_high, close*(1+hret))
    pred_low = min(current_low, close*(1+lret))

    p = float(latest["pressure_6"].iloc[0])
    pressure_text = (
        "盤中買壓較強" if p > 0.15 else
        "盤中賣壓較強" if p < -0.15 else
        "盤中多空接近"
    )

    return {
        "current_price": close,
        "current_high": current_high,
        "current_low": current_low,
        "pred_today_high": pred_high,
        "pred_today_low": pred_low,
        "pressure_text": pressure_text,
        "intraday": intra[["Close"]].tail(120),
    }


# -----------------------------
# Session state
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "首頁"

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "2330.TW"


def go_detail(ticker):
    st.session_state.selected_ticker = ticker
    st.session_state.page = "個股分析"


# -----------------------------
# Sidebar navigation
# -----------------------------
with st.sidebar:
    st.title("台股智慧分析站")

    if st.button("🏠 首頁", use_container_width=True):
        st.session_state.page = "首頁"

    if st.button("🔎 個股分析", use_container_width=True):
        st.session_state.page = "個股分析"

    st.divider()
    st.caption("首頁候選股是模型篩選結果，不代表投資建議。")


# -----------------------------
# 首頁
# -----------------------------
if st.session_state.page == "首頁":
    st.title("今日台股候選股")
    st.caption(
        "依趨勢、相對大盤強弱、MACD、RSI、成交量與資金流快速篩選。"
        "這是量化掃描結果，不是保證上漲的推薦。"
    )

    with st.spinner("正在更新今日候選股..."):
        scan = scan_watchlist()

    if scan.empty:
        st.warning("目前無法取得候選股資料，請稍後重新整理。")
    else:
        top = scan.head(6)

        # 前三名大卡
        st.subheader("今日前 3 名")
        cols = st.columns(3)

        for i, (_, row) in enumerate(top.head(3).iterrows()):
            with cols[i]:
                st.markdown(f"### {row['name']} ({display_code(row['ticker'])})")
                st.metric("綜合分數", f"{row['score']:.0f}/100")
                st.write(f"最新價：**{row['close']:.2f}**")
                st.write(f"5 日漲跌：**{row['ret5']:+.2f}%**")
                st.write(f"買賣壓分數：**{row['pressure']:+.0f}**")
                st.write(f"RSI：**{row['rsi']:.1f}**")
                if st.button(
                    "查看詳細資料",
                    key=f"top_{row['ticker']}",
                    use_container_width=True
                ):
                    go_detail(row["ticker"])
                    st.rerun()

        st.divider()
        st.subheader("其他候選股")

        table = top[[
            "name","ticker","close","score","pressure","rsi","vol_ratio","ret5"
        ]].copy()
        table["ticker"] = table["ticker"].apply(display_code)
        table.columns = [
            "名稱","代碼","最新價","綜合分數",
            "買賣壓","RSI","20日量比","5日漲跌%"
        ]
        st.dataframe(
            table.round({
                "最新價":2,"綜合分數":1,"買賣壓":1,
                "RSI":1,"20日量比":2,"5日漲跌%":2
            }),
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "目前首頁使用一組大型、成交活躍台股清單做快速掃描，"
            "之後也可以擴充成全上市櫃掃描。"
        )


# -----------------------------
# 個股分析頁
# -----------------------------
elif st.session_state.page == "個股分析":
    st.title("個股詳細分析")

    default_code = display_code(st.session_state.selected_ticker)
    code_input = st.text_input("股票代碼", value=default_code)
    ticker = normalize_ticker(code_input)

    mode = st.radio("分析模式", ["當日盤中", "未來多日"], horizontal=True)

    if mode == "未來多日":
        c1, c2 = st.columns(2)
        with c1:
            horizon = st.slider("預測未來交易日", 1, 30, 10)
        with c2:
            period = st.selectbox("歷史資料", ["2y","5y","10y"], index=0)

    run = st.button("開始分析", type="primary")

    if run:
        if mode == "當日盤中":
            with st.spinner(f"正在分析 {ticker} 當日盤中資料..."):
                try:
                    r = analyze_today(ticker)
                except Exception as e:
                    st.error(f"當日分析失敗：{e}")
                    st.stop()

            st.subheader(f"{ticker} 當日盤中")
            a,b,c,d = st.columns(4)
            a.metric("目前價", f"{r['current_price']:.2f}")
            b.metric("目前盤中低", f"{r['current_low']:.2f}")
            c.metric("預估今日低", f"{r['pred_today_low']:.2f}")
            d.metric("預估今日高", f"{r['pred_today_high']:.2f}")
            st.info(r["pressure_text"])
            st.line_chart(r["intraday"])

        else:
            with st.spinner(f"正在分析 {ticker} 未來 {horizon} 個交易日..."):
                try:
                    r = analyze_daily(ticker, horizon, period)
                except Exception as e:
                    st.error(f"多日分析失敗：{e}")
                    st.stop()

            st.subheader(f"{ticker} 未來 {horizon} 日分析")
            a,b,c,d = st.columns(4)
            a.metric("最新收盤", f"{r['last_close']:.2f}")
            b.metric("預估區間低", f"{r['pred_low']:.2f}")
            c.metric("預估區間高", f"{r['pred_high']:.2f}")
            d.metric("買賣壓分數", f"{r['pressure_score']:+.0f}")

            st.info(f"壓力判讀：{r['pressure_label']}")

            e,f,g,h = st.columns(4)
            e.metric("CMF(20)", f"{r['cmf20']:.3f}")
            f.metric("10日上/下量比", f"{r['up_down_vol_ratio_10']:.2f}")
            g.metric("RSI(14)", f"{r['rsi14']:.1f}")
            h.metric("MACD Hist", f"{r['macd_hist']:.3f}")

            st.line_chart(r["chart"])

            st.subheader("歷史驗證")
            i,j = st.columns(2)
            i.metric("高點報酬 MAE", f"{r['high_mae']:.2f}%")
            j.metric("低點報酬 MAE", f"{r['low_mae']:.2f}%")
