"""
optimizer_grid.py
─────────────────
Optimizador de parámetros sobre el histórico anual para encontrar
los mejores ratios SL/TP y umbrales para las estrategias clave (ETH y SOL).
"""

import os
import sys
import io
import pandas as pd
import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from rich.console import Console
from rich.table import Table

console = Console(force_terminal=True, legacy_windows=False)

def load_data(symbol: str) -> pd.DataFrame:
    path = os.path.join("data", f"{symbol}_15m.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def simulate_trades(df: pd.DataFrame, sl_pct: float, tp_pct: float, fee: float = 0.0006):
    trades = []
    in_pos = False
    entry_price = 0.0
    direction = 0 # 1 long, -1 short
    
    for i in range(len(df)):
        signal = df["signal"].iloc[i]
        price = df["close"].iloc[i]
        
        if not in_pos:
            if signal == 1:
                in_pos = True
                direction = 1
                entry_price = price
            elif signal == -1:
                in_pos = True
                direction = -1
                entry_price = price
        else:
            pnl_raw = (price - entry_price) / entry_price if direction == 1 else (entry_price - price) / entry_price
            if pnl_raw >= tp_pct:
                trades.append(tp_pct - 2 * fee)
                in_pos = False
            elif pnl_raw <= -sl_pct:
                trades.append(-sl_pct - 2 * fee)
                in_pos = False
                
    if not trades:
        return 0, 0, 0, 0
    trades = np.array(trades)
    win_rate = np.mean(trades > 0) * 100
    total_pnl = np.sum(trades) * 100
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    pf = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else 99.0
    
    # Max DD
    cum = np.cumsum(trades)
    peak = np.maximum.accumulate(cum)
    dd = np.min(cum - peak) * 100
    return len(trades), win_rate, total_pnl, pf, dd

def optimize_eth_stoch():
    console.print("\n[bold yellow]🔍 Optimizando ETH: Stoch (14,3,3) + EMA200 / EMA50[/bold yellow]")
    df = load_data("ETH")
    
    # Calcular indicadores base
    low_min = df["low"].rolling(14).min()
    high_max = df["high"].rolling(14).max()
    df["k"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
    df["d"] = df["k"].rolling(3).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    
    grid_results = []
    
    # Probar diferentes SL y TP
    for sl in [0.006, 0.008, 0.010, 0.012]:
        for tp_mult in [1.5, 2.0, 2.5, 3.0]:
            tp = sl * tp_mult
            
            # Generar señales
            df["signal"] = 0
            long_cond = (df["k"] > df["d"]) & (df["k"].shift(1) <= df["d"].shift(1)) & (df["k"] < 30) & (df["close"] > df["ema200"])
            short_cond = (df["k"] < df["d"]) & (df["k"].shift(1) >= df["d"].shift(1)) & (df["k"] > 70) & (df["close"] < df["ema200"])
            df.loc[long_cond, "signal"] = 1
            df.loc[short_cond, "signal"] = -1
            
            count, wr, pnl, pf, dd = simulate_trades(df, sl, tp)
            grid_results.append({
                "sl": sl,
                "tp": tp,
                "rr": f"1:{tp_mult:.1f}",
                "trades": count,
                "wr": wr,
                "pnl": pnl,
                "pf": pf,
                "dd": dd
            })
            
    grid_results.sort(key=lambda x: x["pf"], reverse=True)
    
    table = Table(title="Top 5 Ratios SL/TP para ETH Stoch+EMA200")
    table.add_column("SL %", justify="right")
    table.add_column("TP %", justify="right")
    table.add_column("R:R", justify="center")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("PnL Acum", justify="right")
    table.add_column("Profit Factor", justify="right")
    table.add_column("Max DD", justify="right")
    
    for r in grid_results[:5]:
        table.add_row(
            f"{r['sl']*100:.1f}%",
            f"{r['tp']*100:.1f}%",
            r["rr"],
            str(r["trades"]),
            f"{r['wr']:.1f}%",
            f"{r['pnl']:+.1f}%",
            f"{r['pf']:.2f}",
            f"{r['dd']:.1f}%"
        )
    console.print(table)

def optimize_sol_vwap():
    console.print("\n[bold yellow]🔍 Optimizando SOL: VWAP + Volumen[/bold yellow]")
    df = load_data("SOL")
    
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_pv = (typical_price * df["volume"]).cumsum()
    cum_v = df["volume"].cumsum()
    df["vwap"] = cum_pv / (cum_v + 1e-9)
    df["vol_sma"] = df["volume"].rolling(20).mean()
    
    grid_results = []
    for sl in [0.005, 0.007, 0.010]:
        for tp_mult in [1.5, 2.0, 2.5, 3.0]:
            tp = sl * tp_mult
            
            df["signal"] = 0
            long_cond = (df["close"] > df["vwap"]) & (df["close"].shift(1) <= df["vwap"].shift(1)) & (df["volume"] > df["vol_sma"] * 1.5)
            short_cond = (df["close"] < df["vwap"]) & (df["close"].shift(1) >= df["vwap"].shift(1)) & (df["volume"] > df["vol_sma"] * 1.5)
            df.loc[long_cond, "signal"] = 1
            df.loc[short_cond, "signal"] = -1
            
            count, wr, pnl, pf, dd = simulate_trades(df, sl, tp)
            grid_results.append({
                "sl": sl,
                "tp": tp,
                "rr": f"1:{tp_mult:.1f}",
                "trades": count,
                "wr": wr,
                "pnl": pnl,
                "pf": pf,
                "dd": dd
            })
            
    grid_results.sort(key=lambda x: x["pf"], reverse=True)
    table = Table(title="Top 5 Ratios SL/TP para SOL VWAP+Volumen")
    table.add_column("SL %", justify="right")
    table.add_column("TP %", justify="right")
    table.add_column("R:R", justify="center")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("PnL Acum", justify="right")
    table.add_column("Profit Factor", justify="right")
    table.add_column("Max DD", justify="right")
    
    for r in grid_results[:5]:
        table.add_row(
            f"{r['sl']*100:.1f}%",
            f"{r['tp']*100:.1f}%",
            r["rr"],
            str(r["trades"]),
            f"{r['wr']:.1f}%",
            f"{r['pnl']:+.1f}%",
            f"{r['pf']:.2f}",
            f"{r['dd']:.1f}%"
        )
    console.print(table)

if __name__ == "__main__":
    optimize_eth_stoch()
    optimize_sol_vwap()
