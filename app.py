
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


st.set_page_config(
    page_title="台股高低點區間預測",
    page_icon="📈",
    layout="wide",
)


def normalize_ticker(code: str) -> str:
    code = code.strip().upper()
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code
    if code.isdigit():
        return code + ".TW"
    return code


@st.cache_data(ttl=3600)
def download_price(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


@st.cache_data(ttl=3600)
def download_market(period: str = "5y") -> pd.DataFrame:
    df = yf.download(
        "^TWII",
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # Returns / volatility
    x["ret_1"] = x["Close"].pct_change()
    x["ret_3"] = x["Close"].pct_change(3)
    x["ret_5"] = x["Close"].pct_change(5)
    x["ret_10"] = x["Close"].pct_change(10)
    x["vol_5"] = x["ret_1"].rolling(5).std()
    x["vol_10"] = x["ret_1"].rolling(10).std()
    x["vol_20"] = x["ret_1"].rolling(20).std()

    # Moving averages
    for n in [5, 10, 20, 60]:
        x[f"ma_{n}"] = x["Close"].rolling(n).mean()
        x[f"close_to_ma_{n}"] = x["Close"] / x[f"ma_{n}"] - 1

    # RSI
    x["rsi_14"] = rsi(x["Close"], 14)

    # MACD
    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]

    # KD stochastic
    low9 = x["Low"].rolling(9).min()
    high9 = x["High"].rolling(9).max()
    rsv = (x["Close"] - low9) / (high9 - low9).replace(0, np.nan) * 100
    x["k"] = rsv.ewm(alpha=1/3, adjust=False).mean()
    x["d"] = x["k"].ewm(alpha=1/3, adjust=False).mean()

    # Bollinger Bands
    bb_mid = x["Close"].rolling(20).mean()
    bb_std = x["Close"].rolling(20).std()
    x["bb_mid"] = bb_mid
    x["bb_upper"] = bb_mid + 2 * bb_std
    x["bb_lower"] = bb_mid - 2 * bb_std
    x["bb_width"] = (x["bb_upper"] - x["bb_lower"]) / bb_mid
    x["bb_pos"] = (x["Close"] - x["bb_lower"]) / (x["bb_upper"] - x["bb_lower"]).replace(0, np.nan)

    # Candlestick / range
    x["range_pct"] = (x["High"] - x["Low"]) / x["Close"]
    x["open_gap"] = x["Open"] / x["Close"].shift(1) - 1
    x["close_pos"] = (x["Close"] - x["Low"]) / (x["High"] - x["Low"]).replace(0, np.nan)

    # Volume
    x["vol_chg_1"] = x["Volume"].pct_change()
    x["vol_ma_5"] = x["Volume"].rolling(5).mean()
    x["vol_ma_20"] = x["Volume"].rolling(20).mean()
    x["vol_ratio_5"] = x["Volume"] / x["vol_ma_5"]
    x["vol_ratio_20"] = x["Volume"] / x["vol_ma_20"]

    return x


def merge_market_features(stock: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    x = stock.copy()
    m = market[["Close"]].rename(columns={"Close": "market_close"}).copy()
    m["market_ret_1"] = m["market_close"].pct_change()
    m["market_ret_5"] = m["market_close"].pct_change(5)
    m["market_ma20"] = m["market_close"].rolling(20).mean()
    m["market_to_ma20"] = m["market_close"] / m["market_ma20"] - 1

    x = x.join(m[["market_ret_1", "market_ret_5", "market_to_ma20"]], how="left")
    x["relative_strength_5"] = x["ret_5"] - x["market_ret_5"]
    return x


def make_dataset(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    x = df.copy()

    future_high = pd.concat(
        [x["High"].shift(-i) for i in range(1, horizon + 1)], axis=1
    ).max(axis=1)

    future_low = pd.concat(
        [x["Low"].shift(-i) for i in range(1, horizon + 1)], axis=1
    ).min(axis=1)

    x["target_high_ret"] = future_high / x["Close"] - 1
    x["target_low_ret"] = future_low / x["Close"] - 1
    return x


FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10",
    "vol_5", "vol_10", "vol_20",
    "close_to_ma_5", "close_to_ma_10", "close_to_ma_20", "close_to_ma_60",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "k", "d",
    "bb_width", "bb_pos",
    "range_pct", "open_gap", "close_pos",
    "vol_chg_1", "vol_ratio_5", "vol_ratio_20",
    "market_ret_1", "market_ret_5", "market_to_ma20", "relative_strength_5",
]


def train_predict(ticker: str, horizon: int = 10, period: str = "5y"):
    raw = download_price(ticker, period)
    market = download_market(period)

    if raw.empty or len(raw) < 250:
        raise ValueError("歷史資料不足，建議至少 1 年以上資料。")

    df = add_technical_features(raw)
    df = merge_market_features(df, market)
    df = make_dataset(df, horizon)

    train = df.dropna(subset=FEATURES + ["target_high_ret", "target_low_ret"]).copy()
    latest = df.dropna(subset=FEATURES).iloc[[-1]].copy()

    if len(train) < 200:
        raise ValueError("可訓練資料不足。")

    X = train[FEATURES]
    yh = train["target_high_ret"]
    yl = train["target_low_ret"]

    params = dict(
        n_estimators=600,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    high_model = RandomForestRegressor(**params)
    low_model = RandomForestRegressor(**{**params, "random_state": 43})

    # Time-series validation
    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    high_resid = []
    low_resid = []

    for tr_idx, va_idx in tscv.split(X):
        high_model.fit(X.iloc[tr_idx], yh.iloc[tr_idx])
        low_model.fit(X.iloc[tr_idx], yl.iloc[tr_idx])

        ph = high_model.predict(X.iloc[va_idx])
        pl = low_model.predict(X.iloc[va_idx])

        fold_metrics.append((
            mean_absolute_error(yh.iloc[va_idx], ph),
            mean_absolute_error(yl.iloc[va_idx], pl),
        ))
        high_resid.extend((yh.iloc[va_idx].values - ph).tolist())
        low_resid.extend((yl.iloc[va_idx].values - pl).tolist())

    high_model.fit(X, yh)
    low_model.fit(X, yl)

    pred_high_ret = float(high_model.predict(latest[FEATURES])[0])
    pred_low_ret = float(low_model.predict(latest[FEATURES])[0])

    last_close = float(latest["Close"].iloc[0])
    pred_high = max(last_close * (1 + pred_high_ret), last_close)
    pred_low = min(last_close * (1 + pred_low_ret), last_close)

    # 80% empirical uncertainty interval around predicted returns
    hq10, hq90 = np.quantile(high_resid, [0.10, 0.90])
    lq10, lq90 = np.quantile(low_resid, [0.10, 0.90])

    pred_high_interval = (
        last_close * (1 + pred_high_ret + hq10),
        last_close * (1 + pred_high_ret + hq90),
    )
    pred_low_interval = (
        last_close * (1 + pred_low_ret + lq10),
        last_close * (1 + pred_low_ret + lq90),
    )

    high_mae = np.mean([m[0] for m in fold_metrics]) * 100
    low_mae = np.mean([m[1] for m in fold_metrics]) * 100

    # Simple confidence score based on validation MAE; not a probability
    avg_mae = (high_mae + low_mae) / 2
    confidence = max(0, min(100, 100 - avg_mae * 8))

    latest_row = latest.iloc[0]

    technical_summary = {
        "RSI(14)": float(latest_row["rsi_14"]),
        "MACD Hist": float(latest_row["macd_hist"]),
        "K": float(latest_row["k"]),
        "D": float(latest_row["d"]),
        "BB Position": float(latest_row["bb_pos"]),
        "Volume Ratio(20)": float(latest_row["vol_ratio_20"]),
        "Relative Strength 5D": float(latest_row["relative_strength_5"]),
    }

    return {
        "ticker": ticker,
        "date": latest.index[-1],
        "last_close": last_close,
        "pred_low": pred_low,
        "pred_high": pred_high,
        "pred_low_interval": pred_low_interval,
        "pred_high_interval": pred_high_interval,
        "low_mae": low_mae,
        "high_mae": high_mae,
        "confidence": confidence,
        "technical_summary": technical_summary,
        "price_df": raw,
    }


st.title("台股高低點區間預測")
st.caption("以歷史價格、技術指標與台灣加權指數特徵估計未來區間。這不是保證的最高點或最低點，也不是投資建議。")

with st.sidebar:
    st.header("分析設定")
    ticker_input = st.text_input("股票代碼", value="2330", help="例如 2330、2454、6488.TWO")
    horizon = st.slider("預測未來交易日", 3, 30, 10)
    period = st.selectbox("歷史資料", ["2y", "5y", "10y"], index=1)
    run = st.button("開始分析", type="primary", use_container_width=True)

if run:
    ticker = normalize_ticker(ticker_input)

    with st.spinner(f"正在分析 {ticker} ..."):
        try:
            r = train_predict(ticker, horizon, period)
        except Exception as e:
            st.error(f"分析失敗：{e}")
            st.stop()

    st.subheader(f"{ticker} 分析結果")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新收盤", f"{r['last_close']:.2f}")
    c2.metric(
        f"預估 {horizon} 日最低",
        f"{r['pred_low']:.2f}",
        f"{(r['pred_low']/r['last_close']-1)*100:.2f}%"
    )
    c3.metric(
        f"預估 {horizon} 日最高",
        f"{r['pred_high']:.2f}",
        f"{(r['pred_high']/r['last_close']-1)*100:.2f}%"
    )
    c4.metric("模型信心分數", f"{r['confidence']:.0f}/100")

    st.info(
        f"經驗式 80% 不確定區間："
        f"最低點估計約 {r['pred_low_interval'][0]:.2f}～{r['pred_low_interval'][1]:.2f}；"
        f"最高點估計約 {r['pred_high_interval'][0]:.2f}～{r['pred_high_interval'][1]:.2f}。"
        "這是依歷史驗證誤差推估的區間，不代表發生機率。"
    )

    st.subheader("技術指標")
    tech = r["technical_summary"]
    tech_df = pd.DataFrame({
        "指標": list(tech.keys()),
        "數值": [round(v, 4) for v in tech.values()]
    })
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

    st.subheader("近期價格")
    chart_df = r["price_df"][["Close"]].tail(180)
    st.line_chart(chart_df)

    st.subheader("歷史驗證")
    v1, v2 = st.columns(2)
    v1.metric("最高點報酬預測 MAE", f"{r['high_mae']:.2f}%")
    v2.metric("最低點報酬預測 MAE", f"{r['low_mae']:.2f}%")

    st.caption(
        "提醒：股價會受財報、新聞、政策、產業循環與突發事件影響。"
        "模型只使用可取得的歷史市場資料與技術特徵，因此可能大幅失準。"
    )

else:
    st.write("在左側輸入股票代碼後按「開始分析」。")
