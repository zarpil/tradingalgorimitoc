"""
strategies.py
─────────────
Las 5 estrategias como funciones que generan señales sobre un DataFrame OHLCV.
Sin dependencias externas — solo pandas y numpy.

Cada función recibe un DataFrame con columnas Open/High/Low/Close/Volume
y devuelve el mismo DataFrame con columnas extra:
  signal      →  1 = LONG, -1 = SHORT, 0 = sin señal
  sl_price    →  precio stop loss
  tp_price    →  precio take profit
"""

import pandas as pd
import numpy as np


# ── Indicadores base ──────────────────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss
    return 100 - 100 / (1 + rs)

def calc_bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid, mid + std * sigma, mid - std * sigma

def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = calc_ema(series, fast)
    ema_slow   = calc_ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calc_stochastic(df: pd.DataFrame, k=14, d=3):
    low_k  = df["Low"].rolling(k).min()
    high_k = df["High"].rolling(k).max()
    k_line = 100 * (df["Close"] - low_k) / (high_k - low_k)
    d_line = k_line.rolling(d).mean()
    return k_line, d_line

def calc_vwap(df: pd.DataFrame) -> pd.Series:
    hlc3 = (df["High"] + df["Low"] + df["Close"]) / 3
    return (hlc3 * df["Volume"]).cumsum() / df["Volume"].cumsum()


# ── Estrategia 1: EMA 9 / EMA 21 ─────────────────────────────────────────────

def strategy_ema_cross(df: pd.DataFrame, sl_pct=0.008, tp_pct=0.016,
                       fast=9, slow=21, min_sep=0.001) -> pd.DataFrame:
    """Cruce de EMA rápida / lenta con filtro de separación mínima."""
    df = df.copy()
    df["ema_fast"] = calc_ema(df["Close"], fast)
    df["ema_slow"] = calc_ema(df["Close"], slow)

    sep = (df["ema_fast"] - df["ema_slow"]).abs() / df["Close"]

    cross_up   = (df["ema_fast"] > df["ema_slow"]) & \
                 (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & \
                 (sep > min_sep)
    cross_down = (df["ema_fast"] < df["ema_slow"]) & \
                 (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & \
                 (sep > min_sep)

    df["signal"]   = np.where(cross_up, 1, np.where(cross_down, -1, 0))
    df["sl_price"] = np.where(df["signal"] == 1,
                              df["Close"] * (1 - sl_pct),
                              df["Close"] * (1 + sl_pct))
    df["tp_price"] = np.where(df["signal"] == 1,
                              df["Close"] * (1 + tp_pct),
                              df["Close"] * (1 - tp_pct))
    return df


# ── Estrategia 2: RSI + Bollinger Bands ──────────────────────────────────────

def strategy_rsi_bollinger(df: pd.DataFrame, sl_pct=0.007, rsi_lo=30, rsi_hi=70) -> pd.DataFrame:
    """Reversión a la media: toque de banda + confirmación RSI."""
    df = df.copy()
    df["rsi"] = calc_rsi(df["Close"])
    bb_mid, bb_up, bb_lo = calc_bollinger(df["Close"])
    df["bb_mid"] = bb_mid
    df["bb_up"]  = bb_up
    df["bb_lo"]  = bb_lo

    long_cond  = (df["Close"] <= df["bb_lo"]) & (df["rsi"] < rsi_lo)
    short_cond = (df["Close"] >= df["bb_up"]) & (df["rsi"] > rsi_hi)

    df["signal"]   = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    df["sl_price"] = np.where(df["signal"] == 1,
                              df["bb_lo"] * (1 - sl_pct),
                              df["bb_up"] * (1 + sl_pct))
    df["tp_price"] = df["bb_mid"]  # TP = vuelta a la media
    return df


# ── Estrategia 3: MACD + EMA 50 ──────────────────────────────────────────────

def strategy_macd_ema50(df: pd.DataFrame, sl_pct=0.009, tp_pct=0.019) -> pd.DataFrame:
    """MACD con filtro de tendencia EMA50."""
    df = df.copy()
    df["ema50"]   = calc_ema(df["Close"], 50)
    macd, signal  = calc_macd(df["Close"])
    df["macd"]    = macd
    df["signal_l"]= signal

    bull = df["Close"] > df["ema50"]
    bear = df["Close"] < df["ema50"]

    macd_cross_up  = (df["macd"] > df["signal_l"]) & \
                     (df["macd"].shift(1) <= df["signal_l"].shift(1))
    macd_cross_dn  = (df["macd"] < df["signal_l"]) & \
                     (df["macd"].shift(1) >= df["signal_l"].shift(1))

    long_cond  = bull & macd_cross_up
    short_cond = bear & macd_cross_dn

    df["signal"]   = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    df["sl_price"] = np.where(df["signal"] == 1,
                              df["Close"] * (1 - sl_pct),
                              df["Close"] * (1 + sl_pct))
    df["tp_price"] = np.where(df["signal"] == 1,
                              df["Close"] * (1 + tp_pct),
                              df["Close"] * (1 - tp_pct))
    return df


# ── Estrategia 4: VWAP + RSI ─────────────────────────────────────────────────

def strategy_vwap_rsi(df: pd.DataFrame, sl_pct=0.007, tp_pct=0.014) -> pd.DataFrame:
    """Cruce del VWAP con confirmación RSI > / < 50."""
    df = df.copy()
    df["vwap"] = calc_vwap(df)
    df["rsi"]  = calc_rsi(df["Close"])

    crossed_above = (df["Close"] > df["vwap"]) & (df["Close"].shift(1) <= df["vwap"].shift(1))
    crossed_below = (df["Close"] < df["vwap"]) & (df["Close"].shift(1) >= df["vwap"].shift(1))

    long_cond  = crossed_above & (df["rsi"] > 50)
    short_cond = crossed_below & (df["rsi"] < 50)

    df["signal"]   = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    df["sl_price"] = np.where(df["signal"] == 1,
                              df["Close"] * (1 - sl_pct),
                              df["Close"] * (1 + sl_pct))
    df["tp_price"] = np.where(df["signal"] == 1,
                              df["Close"] * (1 + tp_pct),
                              df["Close"] * (1 - tp_pct))
    return df


# ── Estrategia 5: Estocástico + EMA 200 ──────────────────────────────────────

def strategy_stoch_ema200(df: pd.DataFrame, sl_pct=0.010, tp_pct=0.020,
                          stoch_lo=20, stoch_hi=80) -> pd.DataFrame:
    """Filtro de tendencia macro (EMA200) + timing estocástico."""
    df = df.copy()
    df["ema200"] = calc_ema(df["Close"], 200)
    k, d         = calc_stochastic(df)
    df["k"]      = k
    df["d"]      = d

    bull = df["Close"] > df["ema200"]
    bear = df["Close"] < df["ema200"]

    k_cross_up = (df["k"] > df["d"]) & (df["k"].shift(1) <= df["d"].shift(1))
    k_cross_dn = (df["k"] < df["d"]) & (df["k"].shift(1) >= df["d"].shift(1))

    long_cond  = bull & k_cross_up & (df["k"] < stoch_lo)
    short_cond = bear & k_cross_dn & (df["k"] > stoch_hi)

    df["signal"]   = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    df["sl_price"] = np.where(df["signal"] == 1,
                              df["Close"] * (1 - sl_pct),
                              df["Close"] * (1 + sl_pct))
    df["tp_price"] = np.where(df["signal"] == 1,
                              df["Close"] * (1 + tp_pct),
                              df["Close"] * (1 - tp_pct))
    return df


# ── S6: VWAP + Volumen (ganadora nueva IA — PF 5.0, WR 71%) ──────────────────

def strategy_vwap_volume(df: pd.DataFrame,
                          sl_pct: float = 0.007,
                          tp_pct: float = 0.014,
                          vol_mult: float = 1.5) -> pd.DataFrame:
    """
    VWAP + confirmacion de volumen.
    Descubierta en la comparativa: PF 5.0, WR 71%, DD solo -2% en SOL.

    Logica:
      - LONG:  precio cierra POR ENCIMA del VWAP + volumen > media*1.5
      - SHORT: precio cierra POR DEBAJO del VWAP + volumen > media*1.5

    El volumen como filtro es superior al RSI en este mercado:
    selecciona movimientos institucionales reales, no solo oscilaciones.
    """
    df   = df.copy()
    vwap = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() \
           / df["Volume"].cumsum()
    vol_ma   = df["Volume"].rolling(20).mean()
    high_vol = df["Volume"] > vol_ma * vol_mult

    crossed_above = (df["Close"] > vwap) & (df["Close"].shift(1) <= vwap.shift(1))
    crossed_below = (df["Close"] < vwap) & (df["Close"].shift(1) >= vwap.shift(1))

    long_sig  = crossed_above & high_vol
    short_sig = crossed_below & high_vol

    df["signal"]   = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    df["sl_price"] = np.where(df["signal"] == 1,
                              df["Close"] * (1 - sl_pct),
                              df["Close"] * (1 + sl_pct))
    df["tp_price"] = np.where(df["signal"] == 1,
                              df["Close"] * (1 + tp_pct),
                              df["Close"] * (1 - tp_pct))
    return df


# ── S7: Stoch + EMA50 (mejor ratio riesgo/beneficio que EMA200) ───────────────

def strategy_stoch_ema50(df: pd.DataFrame,
                          sl_pct: float = 0.015,
                          tp_pct: float = 0.030,
                          stoch_lo: int = 20,
                          stoch_hi: int = 80) -> pd.DataFrame:
    """
    Estocastico + EMA50 como filtro de tendencia.
    Descubierta en comparativa: ETH +35%, DD -17.8% (vs EMA200: +54% pero DD -33%).

    Ventaja vs EMA200:
      - Genera menos trades (150 vs 254)
      - Drawdown casi a la mitad (-18% vs -33%)
      - Ratio retorno/riesgo superior para capital pequeno

    Ideal para: cartera conservadora donde preservar capital importa.
    """
    df    = df.copy()
    ema50 = calc_ema(df["Close"], 50)

    lo_k  = df["Low"].rolling(14).min()
    hi_k  = df["High"].rolling(14).max()
    k_pct = 100 * (df["Close"] - lo_k) / (hi_k - lo_k + 1e-10)
    k_sma = k_pct.rolling(3).mean()

    bull = df["Close"] > ema50
    bear = df["Close"] < ema50

    k_cross_up = (k_pct > k_sma) & (k_pct.shift(1) <= k_sma.shift(1))
    k_cross_dn = (k_pct < k_sma) & (k_pct.shift(1) >= k_sma.shift(1))

    long_sig  = bull & k_cross_up & (k_pct < stoch_lo)
    short_sig = bear & k_cross_dn & (k_pct > stoch_hi)

    df["signal"]   = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    df["sl_price"] = np.where(df["signal"] == 1,
                              df["Close"] * (1 - sl_pct),
                              df["Close"] * (1 + sl_pct))
    df["tp_price"] = np.where(df["signal"] == 1,
                              df["Close"] * (1 + tp_pct),
                              df["Close"] * (1 - tp_pct))
    return df
