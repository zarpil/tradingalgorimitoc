"""
walk_forward_test.py
────────────────────
Validación walk-forward out-of-sample de las 3 estrategias elegidas.

Metodología:
  - In-sample (optimización):  Sep 2025 → Mar 2026  (6 meses = ~4380 velas 1H)
  - Out-of-sample (validación): Mar 2026 → Sep 2026  (6 meses = ~4380 velas 1H)

Parámetros FIJADOS en in-sample (los del backtest original):
  ETH: VWAP+Vol    SL=2.0%  TP=5.0%
  SOL: EMA+ADX     SL=2.0%  TP=8.0%
  BTC: VWAP+Vol    SL=1.5%  TP=6.0%

Si el PF out-of-sample >= 1.10 y DD <= 20%, los parámetros son válidos.
Si no, hay overfitting y hay que revisar.
"""

import sys, io
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import os
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

console = Console(force_terminal=True, legacy_windows=False)

MAKER_FEE    = 0.0010
CAPITAL_SLOT = 33.33   # EUR por activo
RISK_PCT     = 0.10

# Parámetros FIJADOS (no se tocan en out-of-sample)
FIXED_PARAMS = {
    "ETH/USDT": {"fn": "vwap_vol", "sl": 0.020, "tp": 0.050},
    "SOL/USDT": {"fn": "ema_adx",  "sl": 0.020, "tp": 0.080},
    "BTC/USDT": {"fn": "vwap_vol", "sl": 0.015, "tp": 0.060},
}

# ── Cargar datos ─────────────────────────────────────────────────────────────
def load(symbol):
    path = os.path.join("data", f"{symbol.replace('/', '')}_1h.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# ── Señales ──────────────────────────────────────────────────────────────────
def sig_vwap_vol(df):
    d = df.copy().reset_index(drop=True)
    tp_p = (d["high"] + d["low"] + d["close"]) / 3.0
    d["day_idx"] = d.index // 24
    d["cum_pv"]  = (tp_p * d["volume"]).groupby(d["day_idx"]).cumsum()
    d["cum_v"]   = d["volume"].groupby(d["day_idx"]).cumsum()
    d["vwap"]    = d["cum_pv"] / (d["cum_v"] + 1e-9)
    d["vol_sma"] = d["volume"].rolling(20).mean()
    d["signal"]  = 0
    cond = ((d["close"] > d["vwap"]) &
            (d["close"].shift(1) <= d["vwap"].shift(1)) &
            (d["volume"] > d["vol_sma"] * 1.5))
    d.loc[cond, "signal"] = 1
    return d["signal"]

def sig_ema_adx(df):
    d = df.copy().reset_index(drop=True)
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["h_l"]   = d["high"] - d["low"]
    d["h_pc"]  = (d["high"] - d["close"].shift(1)).abs()
    d["l_pc"]  = (d["low"]  - d["close"].shift(1)).abs()
    d["tr"]    = d[["h_l","h_pc","l_pc"]].max(axis=1)
    d["atr14"] = d["tr"].rolling(14).mean()
    d["dm_p"]  = (d["high"] - d["high"].shift(1)).clip(lower=0)
    d["dm_n"]  = (d["low"].shift(1) - d["low"]).clip(lower=0)
    d["di_p"]  = 100 * d["dm_p"].rolling(14).mean() / (d["atr14"] + 1e-9)
    d["di_n"]  = 100 * d["dm_n"].rolling(14).mean() / (d["atr14"] + 1e-9)
    d["dx"]    = 100 * (d["di_p"] - d["di_n"]).abs() / (d["di_p"] + d["di_n"] + 1e-9)
    d["adx"]   = d["dx"].rolling(14).mean()
    d["signal"] = 0
    cond = ((d["ema20"] > d["ema50"]) &
            (d["ema20"].shift(1) <= d["ema50"].shift(1)) &
            (d["adx"] > 25))
    d.loc[cond, "signal"] = 1
    return d["signal"]

SIG_FNS = {"vwap_vol": sig_vwap_vol, "ema_adx": sig_ema_adx}

# ── Backtest (igual que sim_1h_optimized) ────────────────────────────────────
def backtest(df, sl_pct, tp_pct, signals):
    capital = CAPITAL_SLOT
    trades  = []
    in_pos  = False
    entry   = 0.0
    pos_eur = 0.0
    peak    = capital

    for i in range(len(df)):
        sig = signals.iloc[i] if hasattr(signals, 'iloc') else signals[i]
        hi  = df["high"].iloc[i]
        lo  = df["low"].iloc[i]
        cls = df["close"].iloc[i]

        if in_pos:
            hit_sl = lo <= entry * (1 - sl_pct)
            hit_tp = hi >= entry * (1 + tp_pct)
            if hit_sl or hit_tp:
                exit_pct = tp_pct if hit_tp else -sl_pct
                pnl = pos_eur * exit_pct - pos_eur * MAKER_FEE * 2
                capital = max(0.0, capital + pnl)
                trades.append({"TP": hit_tp, "pnl": pnl, "cap": capital})
                in_pos = False
                peak = max(peak, capital)

        if not in_pos and sig == 1 and capital > 1:
            risk    = capital * RISK_PCT
            pos_eur = min(risk / sl_pct, capital * 0.5)
            entry   = cls
            capital -= pos_eur * MAKER_FEE
            in_pos  = True

    if not trades:
        return {"n": 0, "wr": 0, "pnl_pct": 0, "pf": 0, "dd": 0, "final": capital}

    wins  = [t["pnl"] for t in trades if t["pnl"] > 0]
    loses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    wr    = len(wins) / len(trades) * 100
    pnl   = (capital - CAPITAL_SLOT) / CAPITAL_SLOT * 100
    pf    = abs(sum(wins) / sum(loses)) if loses and sum(loses) != 0 else 99.0
    caps  = [t["cap"] for t in trades]
    pk    = caps[0]; dd = 0
    for c in caps:
        if c > pk: pk = c
        dd = min(dd, c - pk)
    dd_pct = dd / pk * 100 if pk > 0 else 0
    return {"n": len(trades), "wr": wr, "pnl_pct": pnl, "pf": pf, "dd": dd_pct, "final": capital}

# ── Ejecutar walk-forward ─────────────────────────────────────────────────────
def run():
    console.rule("[bold cyan]WALK-FORWARD TEST — VALIDACIÓN OUT-OF-SAMPLE[/bold cyan]")
    console.print("Metodología: in-sample (6 meses) → parámetros fijos → out-of-sample (6 meses)\n")

    symbols = list(FIXED_PARAMS.keys())
    results = {}

    for sym in symbols:
        df   = load(sym)
        p    = FIXED_PARAMS[sym]
        fn   = SIG_FNS[p["fn"]]
        sl   = p["sl"]
        tp   = p["tp"]

        # Split temporal (50/50 por índice de vela)
        mid  = len(df) // 2
        df_in  = df.iloc[:mid].reset_index(drop=True)
        df_out = df.iloc[mid:].reset_index(drop=True)

        date_start = df["timestamp"].iloc[0].strftime("%Y-%m-%d")
        date_mid   = df["timestamp"].iloc[mid].strftime("%Y-%m-%d")
        date_end   = df["timestamp"].iloc[-1].strftime("%Y-%m-%d")

        console.print(f"[cyan]{sym}[/cyan]")
        console.print(f"  In-sample:    {date_start} → {date_mid}  ({len(df_in)} velas)")
        console.print(f"  Out-of-sample:{date_mid} → {date_end}  ({len(df_out)} velas)")

        sig_in  = fn(df_in)
        sig_out = fn(df_out)

        r_in  = backtest(df_in,  sl, tp, sig_in)
        r_out = backtest(df_out, sl, tp, sig_out)

        results[sym] = {"in": r_in, "out": r_out, "sl": sl, "tp": tp, "strategy": p["fn"]}

    # Tabla comparativa
    console.rule("[bold green]RESULTADOS IN-SAMPLE vs OUT-OF-SAMPLE[/bold green]")
    tbl = Table(box=box.DOUBLE)
    tbl.add_column("Activo",     style="cyan")
    tbl.add_column("Estrategia")
    tbl.add_column("Fase",       justify="center")
    tbl.add_column("Trades",     justify="right")
    tbl.add_column("Win Rate",   justify="right")
    tbl.add_column("PnL",        justify="right")
    tbl.add_column("PF",         justify="right")
    tbl.add_column("Max DD",     justify="right")
    tbl.add_column("Válido?",    justify="center")

    portfolio_out_pnls = []

    for sym, res in results.items():
        for fase, r in [("IN-SAMPLE", res["in"]), ("OUT-SAMPLE", res["out"])]:
            color     = "green" if r["pnl_pct"] >= 0 else "red"
            pnl_txt   = f"[{color}]{r['pnl_pct']:+.1f}%[/{color}]"
            pf_txt    = f"[{'green' if r['pf'] >= 1.1 else 'red'}]{r['pf']:.2f}[/{'green' if r['pf'] >= 1.1 else 'red'}]"
            valid     = "✅" if (r["pf"] >= 1.10 and r["dd"] >= -20) else "❌"
            if fase == "OUT-SAMPLE":
                portfolio_out_pnls.append(r["pnl_pct"])
            tbl.add_row(
                sym if fase == "IN-SAMPLE" else "",
                res["strategy"] if fase == "IN-SAMPLE" else "",
                f"[dim]{fase}[/dim]" if fase == "IN-SAMPLE" else f"[bold]{fase}[/bold]",
                str(r["n"]), f"{r['wr']:.0f}%", pnl_txt, pf_txt,
                f"{r['dd']:.1f}%", valid
            )

    console.print(tbl)

    # Veredicto de portfolio
    console.rule("[bold magenta]VEREDICTO PORTFOLIO OUT-OF-SAMPLE[/bold magenta]")
    avg_out = sum(portfolio_out_pnls) / len(portfolio_out_pnls) if portfolio_out_pnls else 0
    validos = sum(1 for r in results.values()
                  if r["out"]["pf"] >= 1.10 and r["out"]["dd"] >= -20)

    if validos == 3:
        v_txt = "[bold green]✅ LOS 3 ACTIVOS PASAN EL FORWARD TEST — Estrategias válidas[/bold green]"
    elif validos >= 1:
        v_txt = f"[yellow]⚠ {validos}/3 activos pasan. Revisar los que fallan antes de subir capital.[/yellow]"
    else:
        v_txt = "[bold red]❌ NINGÚN ACTIVO PASA — Posible overfitting. NO usar en real con capital alto.[/bold red]"

    console.print(f"\n{v_txt}")
    console.print(f"PnL medio out-of-sample (6 meses): [bold]{avg_out:+.1f}%[/bold]")
    console.print(f"PnL anualizado estimado:            [bold]{avg_out*2:+.1f}%[/bold]")
    console.print(f"\nCriterios de validación: PF >= 1.10 y Max DD >= -20%")

if __name__ == "__main__":
    run()
