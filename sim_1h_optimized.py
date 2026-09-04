"""
sim_1h_optimized.py
────────────────────
Simulacion completa en timeframe 1H con comisiones REALES de Bybit EU:
  - Maker: 0.10% por lado (0.20% round-trip)
  - Taker: 0.25% por lado (0.50% round-trip)

Capital: 100 EUR
Estrategias probadas:
  1. Stoch(14,3,3) + EMA200
  2. Stoch(14,3,3) + EMA50
  3. VWAP diario + Volumen
  4. RSI(14) + EMA200 (nueva para 1H)
  5. EMA 20/50 crossover + ADX filtro (nueva para 1H)
"""

import sys
import io

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import os
import ccxt
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

console = Console(force_terminal=True, legacy_windows=False)

# ── Comisiones reales Bybit EU ────────────────────────────────────────────────
MAKER_FEE = 0.0010   # 0.10% por operacion
TAKER_FEE = 0.0025   # 0.25% por operacion (usamos Maker siempre que podamos)
FEE = MAKER_FEE      # el bot usara ordenes limite (Maker)

CAPITAL     = 100.0
RISK_PCT    = 0.10   # arriesgar 10% del slot por operacion
SYMBOLS     = ["ETH/USDT", "SOL/USDT", "BTC/USDT"]
TIMEFRAME   = "1h"
LIMIT       = 8760   # 1 año de velas 1H

# ══════════════════════════════════════════════════════════════════════════════
#  DESCARGA DE DATOS 1H DESDE BYBIT EN VIVO
# ══════════════════════════════════════════════════════════════════════════════
def download_1h(symbol: str) -> pd.DataFrame:
    """Descarga 1H desde Bybit publico y guarda en data/SYMBOL_1h.csv"""
    cached = os.path.join("data", f"{symbol.replace('/', '')}_1h.csv")
    if os.path.exists(cached):
        df = pd.read_csv(cached)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        console.print(f"  [green]Cargando cache[/green]: {symbol} → {len(df)} velas 1H")
        return df

    console.print(f"  [yellow]Descargando[/yellow] {symbol} 1H desde Bybit...")
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    all_ohlcv = []
    since = ex.parse8601("2025-09-04T00:00:00Z")
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 3600 * 1000
        if len(ohlcv) < 1000:
            break
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop_duplicates("timestamp", inplace=True)
    df.to_csv(cached, index=False)
    console.print(f"  [green]OK[/green]: {len(df)} velas guardadas")
    return df

# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR DE BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
def backtest(df: pd.DataFrame, sl_pct: float, tp_pct: float) -> dict:
    """Backtest completo con posicion sizing basado en riesgo fijo."""
    capital = CAPITAL / len(SYMBOLS)  # cada simbolo recibe capital/3 aprox
    trades = []
    in_pos = False
    entry_price = 0.0
    pos_size_eur = 0.0
    peak = capital

    for i in range(len(df)):
        sig = df["signal"].iloc[i]
        hi  = df["high"].iloc[i]
        lo  = df["low"].iloc[i]
        cls = df["close"].iloc[i]

        if in_pos:
            # comprobar SL y TP con high/low de la vela (mas realista)
            hit_sl = lo <= entry_price * (1 - sl_pct)
            hit_tp = hi >= entry_price * (1 + tp_pct)

            if hit_sl or hit_tp:
                exit_pct = tp_pct if hit_tp else -sl_pct
                pnl_eur = pos_size_eur * exit_pct - pos_size_eur * FEE * 2
                capital += pnl_eur
                capital = max(0.0, capital)
                trades.append({
                    "result": "TP" if hit_tp else "SL",
                    "pnl_eur": round(pnl_eur, 3),
                    "pnl_pct": round(exit_pct * 100, 2),
                    "capital": round(capital, 2),
                })
                in_pos = False
                peak = max(peak, capital)

        if not in_pos and sig == 1 and capital > 1:
            risk_eur = capital * RISK_PCT
            pos_size_eur = risk_eur / sl_pct
            pos_size_eur = min(pos_size_eur, capital * 0.5)
            entry_price = cls
            capital -= pos_size_eur * FEE  # comision entrada
            in_pos = True

    if not trades:
        return {"n": 0, "wr": 0, "pnl": 0, "pf": 0, "dd": 0, "cap": capital}

    wins  = [t["pnl_eur"] for t in trades if t["pnl_eur"] > 0]
    loses = [t["pnl_eur"] for t in trades if t["pnl_eur"] <= 0]
    wr    = len(wins) / len(trades) * 100
    pnl   = (capital - CAPITAL / len(SYMBOLS)) / (CAPITAL / len(SYMBOLS)) * 100
    pf    = abs(sum(wins) / sum(loses)) if loses and sum(loses) != 0 else 99.0
    # max drawdown
    all_caps = [t["capital"] for t in trades]
    pk = all_caps[0]
    dd = 0
    for c in all_caps:
        if c > pk: pk = c
        dd = min(dd, c - pk)
    dd_pct = dd / pk * 100 if pk > 0 else 0

    return {"n": len(trades), "wr": wr, "pnl": pnl, "pf": pf, "dd": dd_pct, "cap": capital}

# ══════════════════════════════════════════════════════════════════════════════
#  GENERADORES DE SEÑALES 1H
# ══════════════════════════════════════════════════════════════════════════════
def sig_stoch_ema200(df):
    d = df.copy()
    lo14 = d["low"].rolling(14).min(); hi14 = d["high"].rolling(14).max()
    d["k"] = 100 * (d["close"] - lo14) / (hi14 - lo14 + 1e-9)
    d["d"] = d["k"].rolling(3).mean()
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()
    d["signal"] = 0
    cond = (d["k"] > d["d"]) & (d["k"].shift(1) <= d["d"].shift(1)) & (d["k"] < 35) & (d["close"] > d["ema200"])
    d.loc[cond, "signal"] = 1
    return d

def sig_stoch_ema50(df):
    d = df.copy()
    lo14 = d["low"].rolling(14).min(); hi14 = d["high"].rolling(14).max()
    d["k"] = 100 * (d["close"] - lo14) / (hi14 - lo14 + 1e-9)
    d["d"] = d["k"].rolling(3).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["signal"] = 0
    cond = (d["k"] > d["d"]) & (d["k"].shift(1) <= d["d"].shift(1)) & (d["k"] < 30) & (d["close"] > d["ema50"])
    d.loc[cond, "signal"] = 1
    return d

def sig_rsi_ema200(df):
    d = df.copy()
    delta = d["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d["rsi"] = 100 - 100 / (1 + rs)
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()
    d["signal"] = 0
    # Compra: RSI sale de zona sobreventa (<35) con precio sobre EMA200
    cond = (d["rsi"] > 35) & (d["rsi"].shift(1) <= 35) & (d["close"] > d["ema200"])
    d.loc[cond, "signal"] = 1
    return d

def sig_ema_cross_adx(df):
    d = df.copy()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    # ADX manual
    d["h_l"] = d["high"] - d["low"]
    d["h_pc"] = (d["high"] - d["close"].shift(1)).abs()
    d["l_pc"] = (d["low"] - d["close"].shift(1)).abs()
    d["tr"] = d[["h_l","h_pc","l_pc"]].max(axis=1)
    d["atr14"] = d["tr"].rolling(14).mean()
    d["dm_pos"] = (d["high"] - d["high"].shift(1)).clip(lower=0)
    d["dm_neg"] = (d["low"].shift(1) - d["low"]).clip(lower=0)
    d["di_pos"] = 100 * d["dm_pos"].rolling(14).mean() / (d["atr14"] + 1e-9)
    d["di_neg"] = 100 * d["dm_neg"].rolling(14).mean() / (d["atr14"] + 1e-9)
    d["dx"] = 100 * (d["di_pos"] - d["di_neg"]).abs() / (d["di_pos"] + d["di_neg"] + 1e-9)
    d["adx"] = d["dx"].rolling(14).mean()
    d["signal"] = 0
    # Compra: cruce EMA20 > EMA50 con ADX > 25 (tendencia fuerte)
    cond = (d["ema20"] > d["ema50"]) & (d["ema20"].shift(1) <= d["ema50"].shift(1)) & (d["adx"] > 25)
    d.loc[cond, "signal"] = 1
    return d

def sig_vwap_vol(df):
    d = df.copy()
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    # VWAP se reinicia cada 24 velas (1 dia en 1H)
    d["day_idx"] = d.index // 24
    d["cum_pv"] = (tp * d["volume"]).groupby(d["day_idx"]).cumsum()
    d["cum_v"]  = d["volume"].groupby(d["day_idx"]).cumsum()
    d["vwap"] = d["cum_pv"] / (d["cum_v"] + 1e-9)
    d["vol_sma"] = d["volume"].rolling(20).mean()
    d["signal"] = 0
    cond = (d["close"] > d["vwap"]) & (d["close"].shift(1) <= d["vwap"].shift(1)) & (d["volume"] > d["vol_sma"] * 1.5)
    d.loc[cond, "signal"] = 1
    return d

# ══════════════════════════════════════════════════════════════════════════════
#  GRID SEARCH DE SL / TP para cada estrategia
# ══════════════════════════════════════════════════════════════════════════════
SL_OPTIONS = [0.015, 0.020, 0.025, 0.030]
TP_MULTS   = [2.0, 2.5, 3.0, 4.0]

STRATEGIES = [
    ("Stoch + EMA200", sig_stoch_ema200),
    ("Stoch + EMA50",  sig_stoch_ema50),
    ("RSI + EMA200",   sig_rsi_ema200),
    ("EMA20/50 + ADX", sig_ema_cross_adx),
    ("VWAP + Vol",     sig_vwap_vol),
]

def run_all():
    console.rule("[bold cyan]SIMULACION 1H — BYBIT EU SPOT — COMISION MAKER 0.10%[/bold cyan]")
    console.print(f"Capital total: {CAPITAL} EUR | Comision/operacion: {FEE*100:.2f}% (Maker)\n")

    all_data = {}
    for sym in SYMBOLS:
        df = download_1h(sym)
        all_data[sym] = df

    best_overall = []

    for sym in SYMBOLS:
        console.rule(f"[yellow]{sym}[/yellow]")
        df_raw = all_data[sym]
        results = []

        for strat_name, sig_fn in STRATEGIES:
            df_sig = sig_fn(df_raw.copy())
            best_pf = -1
            best_res = None
            for sl in SL_OPTIONS:
                for tp_m in TP_MULTS:
                    tp = sl * tp_m
                    res = backtest(df_sig, sl, tp)
                    if res["n"] >= 3 and res["pf"] > best_pf:
                        best_pf = res["pf"]
                        best_res = {**res, "sl": sl, "tp": tp, "rr": f"1:{tp_m:.1f}",
                                    "strat": strat_name, "sym": sym}
            if best_res:
                results.append(best_res)

        results.sort(key=lambda x: x["pf"], reverse=True)

        table = Table(box=box.SIMPLE_HEAVY)
        table.add_column("Estrategia", style="cyan")
        table.add_column("SL", justify="right")
        table.add_column("TP", justify="right")
        table.add_column("R:R", justify="center")
        table.add_column("Trades", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("PnL", justify="right")
        table.add_column("Profit Factor", justify="right")
        table.add_column("Max DD", justify="right")

        for r in results[:5]:
            pnl_col = f"[green]{r['pnl']:+.1f}%[/green]" if r['pnl'] >= 0 else f"[red]{r['pnl']:+.1f}%[/red]"
            table.add_row(
                r["strat"], f"{r['sl']*100:.1f}%", f"{r['tp']*100:.1f}%", r["rr"],
                str(r["n"]), f"{r['wr']:.0f}%", pnl_col,
                f"{r['pf']:.2f}", f"{r['dd']:.1f}%"
            )
        console.print(table)
        best_overall.extend(results[:2])

    # Tabla final combinada
    console.rule("[bold green]TOP COMBINACIONES GLOBALES — PORTFOLIO OPTIMO[/bold green]")
    best_overall.sort(key=lambda x: (x["pf"] * (x["pnl"] > 0)), reverse=True)
    top = [r for r in best_overall if r["pnl"] > 0][:6]

    tbl = Table(title="Las mejores estrategias 1H para tus 100 EUR en Bybit EU", box=box.DOUBLE)
    tbl.add_column("Rank", justify="center")
    tbl.add_column("Activo", justify="center")
    tbl.add_column("Estrategia", style="cyan")
    tbl.add_column("SL", justify="right")
    tbl.add_column("TP", justify="right")
    tbl.add_column("R:R", justify="center")
    tbl.add_column("Trades/año", justify="right")
    tbl.add_column("Win Rate", justify="right")
    tbl.add_column("PnL Neto", justify="right")
    tbl.add_column("Profit Factor", justify="right")
    tbl.add_column("Max DD", justify="right")

    for i, r in enumerate(top, 1):
        pnl_c = f"[bold green]{r['pnl']:+.1f}%[/bold green]"
        tbl.add_row(str(i), r["sym"], r["strat"],
                    f"{r['sl']*100:.1f}%", f"{r['tp']*100:.1f}%", r["rr"],
                    str(r["n"]), f"{r['wr']:.0f}%", pnl_c,
                    f"{r['pf']:.2f}", f"{r['dd']:.1f}%")
    console.print(tbl)

    # Proyeccion de cartera
    console.rule("[bold magenta]PROYECCION CARTERA 100 EUR[/bold magenta]")
    if top:
        avg_annual = sum(r["pnl"] for r in top[:3]) / min(3, len(top))
        console.print(f"\nRendimiento medio anual esperado (top 3 combinadas): [bold green]{avg_annual:+.1f}%[/bold green]")
        for h in [1, 2, 3, 5]:
            proj = CAPITAL * ((1 + avg_annual/100) ** h)
            console.print(f"  {h} año{'s' if h > 1 else ''}: [cyan]{proj:.0f} EUR[/cyan]  ({proj-CAPITAL:+.0f} EUR)")

if __name__ == "__main__":
    run_all()
