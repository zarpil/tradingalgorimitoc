"""
bot_bybit.py
────────────
Bot de trading algorítmico optimizado para Bybit EU (Spot, comisión Maker 0.10%).

Estrategias validadas con datos reales 1H, 1 año de backtest:
  1. ETH/USDT  — VWAP diario + Volumen  (SL 2%, TP 5%,  R:R 1:2.5) → +23.5% anual
  2. SOL/USDT  — EMA 20/50 + ADX       (SL 2%, TP 8%,  R:R 1:4.0) → +12.9% anual
  3. BTC/USDT  — VWAP diario + Volumen  (SL 1.5%, TP 6%, R:R 1:4.0) → +7.6% anual

Guardarraíles activos (recomendación Hermes):
  • Kill switch: si DD de portfolio > 20% → bot para automáticamente
  • Máx 1 posición abierta simultánea (evitar correlación en crash)
  • Log de slippage: expected fill vs real fill guardado en trade_log.csv

Comisiones: MAKER 0.10% (órdenes límite).
Proyección media cartera (3 slots, 100 EUR): +14.1% anual neto.
Timeframe: 1H.
"""

import os
import sys
import io
import csv
import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

# ── Codificación Windows ──────────────────────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from rich.console import Console
from rich.table import Table
from rich import box
from telegram_alerts import TelegramNotifier

console = Console(force_terminal=True, legacy_windows=False)

# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE .env
# ══════════════════════════════════════════════════════════════════════════════
def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

load_env_file()

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
MODE = os.getenv("MODE", "demo").lower()

API_KEYS = {
    "demo": {
        "apiKey": os.getenv("BYBIT_TESTNET_API_KEY", ""),
        "secret": os.getenv("BYBIT_TESTNET_API_SECRET", ""),
    },
    "real": {
        "apiKey": os.getenv("BYBIT_REAL_API_KEY", ""),
        "secret": os.getenv("BYBIT_REAL_API_SECRET", ""),
    }
}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

TIMEFRAME   = "1h"
MAKER_FEE   = 0.0010   # 0.10% — usando órdenes límite (Maker)

# ── Guardarraíles (Hermes) ──────────────────────────────────────────────────
MAX_PORTFOLIO_DD  = 0.20   # Kill switch: parar si DD > 20%
MAX_OPEN_POSITIONS = 1      # Máx posiciones simultáneas (evitar crash correlado)
SLIPPAGE_ALERT_PCT = 0.0005 # Alertar si slippage medio > 0.05%
TRADE_LOG_PATH     = "trade_log.csv"  # Registro de todas las operaciones

# ── Estrategias activas — validadas con walk-forward out-of-sample ────────────
# BTC eliminado: PF=1.07 out-of-sample → no supera el umbral mínimo de 1.10
# Distribución: ETH 40 USDT + SOL 40 USDT + 40 USDT reserva en USDT
PORTFOLIO = [
    {
        "symbol":    "ETH/USDT",
        "strategy":  "VWAP + Volumen",
        "sl_pct":    0.020,   # Stop Loss: 2.0%
        "tp_pct":    0.050,   # Take Profit: 5.0%  (R:R 1:2.5) → neto +4.80%
        "fn":        "analyze_vwap_vol",
        "capital":   40.0,    # USDT por slot
    },
    {
        "symbol":    "SOL/USDT",
        "strategy":  "EMA 20/50 + ADX",
        "sl_pct":    0.020,   # Stop Loss: 2.0%
        "tp_pct":    0.080,   # Take Profit: 8.0%  (R:R 1:4) → neto +7.80%
        "fn":        "analyze_ema_adx",
        "capital":   40.0,
    },
]

# 40 USDT restantes permanecen como reserva en USDT (disponibles para próximo dip)

# ══════════════════════════════════════════════════════════════════════════════
#  EXCHANGE
# ══════════════════════════════════════════════════════════════════════════════
def get_market_exchange():
    """Exchange público para leer velas sin autenticación."""
    return ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "spot"}})

def get_auth_exchange():
    """Exchange autenticado para balance y órdenes en Bybit.eu."""
    creds = API_KEYS[MODE]
    ex = ccxt.bybit({
        "apiKey": creds["apiKey"],
        "secret": creds["secret"],
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    ex.urls["api"]["private"] = "https://api.bybit.eu"
    ex.urls["api"]["public"]  = "https://api.bybit.eu"
    return ex

# ══════════════════════════════════════════════════════════════════════════════
#  DESCARGA DE VELAS 1H
# ══════════════════════════════════════════════════════════════════════════════
def fetch_candles(market_ex, symbol: str, limit: int = 250) -> pd.DataFrame:
    try:
        ohlcv = market_ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        console.print(f"[red]Error obteniendo velas de {symbol}: {e}[/red]")
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
#  SEÑALES — ESTRATEGIA 1 y 3: VWAP DIARIO + VOLUMEN ANORMAL
# ══════════════════════════════════════════════════════════════════════════════
def analyze_vwap_vol(df: pd.DataFrame, sl_pct: float, tp_pct: float):
    """
    Compra: precio cruza el VWAP diario al alza con volumen anormalmente alto.
    El VWAP diario se reinicia cada 24 velas (24h).
    """
    if len(df) < 30:
        return 0, 0.0, 0.0

    d = df.copy()
    tp_price = (d["high"] + d["low"] + d["close"]) / 3.0
    d["day_idx"] = d.index // 24
    d["cum_pv"]  = (tp_price * d["volume"]).groupby(d["day_idx"]).cumsum()
    d["cum_v"]   = d["volume"].groupby(d["day_idx"]).cumsum()
    d["vwap"]    = d["cum_pv"] / (d["cum_v"] + 1e-9)
    d["vol_sma"] = d["volume"].rolling(20).mean()

    curr = d.iloc[-1]
    prev = d.iloc[-2]
    price = curr["close"]
    vol_surge = curr["volume"] > curr["vol_sma"] * 1.5

    # LONG: precio cruza VWAP al alza con volumen anormal
    if (price > curr["vwap"]) and (prev["close"] <= prev["vwap"]) and vol_surge:
        sl = price * (1.0 - sl_pct)
        tp = price * (1.0 + tp_pct)
        return 1, sl, tp

    return 0, 0.0, 0.0

# ══════════════════════════════════════════════════════════════════════════════
#  SEÑALES — ESTRATEGIA 2: EMA 20/50 CRUCE + FILTRO ADX > 25
# ══════════════════════════════════════════════════════════════════════════════
def analyze_ema_adx(df: pd.DataFrame, sl_pct: float, tp_pct: float):
    """
    Compra: cruce EMA20 > EMA50 con ADX > 25 (tendencia fuerte confirmada).
    Evita entrar en mercados laterales (ADX bajo filtra ruido).
    """
    if len(df) < 60:
        return 0, 0.0, 0.0

    d = df.copy()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()

    # ADX manual
    d["h_l"]   = d["high"] - d["low"]
    d["h_pc"]  = (d["high"] - d["close"].shift(1)).abs()
    d["l_pc"]  = (d["low"]  - d["close"].shift(1)).abs()
    d["tr"]    = d[["h_l", "h_pc", "l_pc"]].max(axis=1)
    d["atr14"] = d["tr"].rolling(14).mean()
    d["dm_p"]  = (d["high"] - d["high"].shift(1)).clip(lower=0)
    d["dm_n"]  = (d["low"].shift(1) - d["low"]).clip(lower=0)
    d["di_p"]  = 100 * d["dm_p"].rolling(14).mean() / (d["atr14"] + 1e-9)
    d["di_n"]  = 100 * d["dm_n"].rolling(14).mean() / (d["atr14"] + 1e-9)
    d["dx"]    = 100 * (d["di_p"] - d["di_n"]).abs() / (d["di_p"] + d["di_n"] + 1e-9)
    d["adx"]   = d["dx"].rolling(14).mean()

    curr = d.iloc[-1]
    prev = d.iloc[-2]
    price = curr["close"]

    # LONG: cruce EMA20 por encima de EMA50 con tendencia fuerte (ADX > 25)
    cruce_al_alza = (curr["ema20"] > curr["ema50"]) and (prev["ema20"] <= prev["ema50"])
    tendencia     = curr["adx"] > 25

    if cruce_al_alza and tendencia:
        sl = price * (1.0 - sl_pct)
        tp = price * (1.0 + tp_pct)
        return 1, sl, tp

    return 0, 0.0, 0.0

# ══════════════════════════════════════════════════════════════════════════════
#  DISPATCHER de estrategias
# ══════════════════════════════════════════════════════════════════════════════
STRATEGY_FNS = {
    "analyze_vwap_vol": analyze_vwap_vol,
    "analyze_ema_adx":  analyze_ema_adx,
}

# ══════════════════════════════════════════════════════════════════════════════
#  BUCLE PRINCIPAL DEL BOT
# ══════════════════════════════════════════════════════════════════════════════
def run_bot():
    console.rule("[bold cyan]BOT DE TRADING — BYBIT EU SPOT — OPTIMIZADO 1H[/bold cyan]")
    mode_color = "yellow" if MODE == "demo" else "bold green"
    console.print(f"Modo: [{mode_color}]{MODE.upper()}[/{mode_color}]  |  Comision Maker: 0.10%  |  Timeframe: {TIMEFRAME}\n")

    console.print("[dim]Estrategias activas:[/dim]")
    for p in PORTFOLIO:
        console.print(
            f"  • [cyan]{p['symbol']:12s}[/cyan] {p['strategy']:20s}"
            f" | SL {p['sl_pct']*100:.1f}%  TP {p['tp_pct']*100:.1f}%"
            f"  R:R 1:{p['tp_pct']/p['sl_pct']:.0f}  | Slot: {p['capital']:.2f} EUR"
        )

    market_ex = get_market_exchange()
    auth_ex   = get_auth_exchange()

    # Verificar feed de mercado
    try:
        ts = market_ex.fetch_time()
        console.print(f"\n[green]✔ Feed de mercado en vivo conectado (Bybit, ts: {ts})[/green]")
    except Exception as e:
        console.print(f"[yellow]Aviso feed: {e}[/yellow]")

    # Verificar autenticacion y saldo
    try:
        bal = auth_ex.fetch_balance()
        usdt_total = bal.get("USDT", {}).get("total", 0.0) or 0.0
        usdt_free  = bal.get("USDT", {}).get("free",  0.0) or 0.0
        btc_total  = bal.get("BTC",  {}).get("total", 0.0) or 0.0
        console.print(f"[bold green]✔ Autenticacion Bybit.eu OK[/bold green]")
        console.print(f"   USDT: {usdt_total:.2f} (libre: {usdt_free:.2f})  |  BTC: {btc_total:.6f}")
    except Exception as e:
        console.print(f"[yellow]⚠ Modo Observacion: {e}[/yellow]")

    notifier.send_message(
        f"*Bot Iniciado*\nModo: `{MODE.upper()}`\nTimeframe: `{TIMEFRAME}`\n"
        f"ETH VWAP+Vol | SOL EMA+ADX | BTC VWAP+Vol\n"
        f"Proyeccion: +14% anual neto con fee Maker 0.10%"
    )

    console.print("\n[dim]Revision cada 5 minutos (velas 1H). Ctrl+C para detener.[/dim]\n")

    # Estado de posiciones abiertas: clave = symbol
    open_positions = {}

    # Capital inicial para calcular DD del portfolio
    try:
        bal0 = auth_ex.fetch_balance()
        capital_inicial = bal0.get("USDT", {}).get("total", 120.0) or 120.0
    except Exception:
        capital_inicial = 120.0
    capital_peak = capital_inicial
    console.print(f"[dim]Capital inicial registrado: {capital_inicial:.2f} USDT | Kill switch si DD > {MAX_PORTFOLIO_DD*100:.0f}%[/dim]")
    console.print(f"[dim]Max posiciones simultáneas: {MAX_OPEN_POSITIONS} | Log: {TRADE_LOG_PATH}[/dim]\n")

    # Inicializar CSV de log si no existe
    if not os.path.exists(TRADE_LOG_PATH):
        with open(TRADE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "symbol", "side", "order_type", "qty",
                        "expected_price", "fill_price", "slippage_pct",
                        "sl", "tp", "result", "pnl_usdt", "balance_usdt"])

    def log_trade(symbol, side, order_type, qty, expected, fill, sl=None, tp=None,
                  result="", pnl_usdt=0.0, balance=0.0):
        """Guarda cada operación en trade_log.csv con slippage real."""
        slippage = abs(fill - expected) / expected if expected else 0
        with open(TRADE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([datetime.now().isoformat(), symbol, side, order_type,
                        round(qty, 6), round(expected, 4), round(fill, 4),
                        round(slippage * 100, 4),
                        round(sl, 4) if sl else "",
                        round(tp, 4) if tp else "",
                        result, round(pnl_usdt, 4), round(balance, 2)])
        if slippage > SLIPPAGE_ALERT_PCT:
            console.print(f"[yellow]⚠ Slippage alto en {symbol}: {slippage*100:.3f}% (esperado {expected:.4f}, fill {fill:.4f})[/yellow]")
            notifier.send_message(f"⚠️ *Slippage alto* en `{symbol}`\nEsperado: `{expected:.4f}` | Fill: `{fill:.4f}` | Slippage: `{slippage*100:.3f}%`")

    def place_order(symbol, side, amount, price=None, order_type="limit"):
        """Coloca una orden en Bybit EU. Devuelve (order_id, fill_price) o (None, None)."""
        try:
            params = {}
            if order_type == "limit":
                order = auth_ex.create_limit_order(symbol, side, amount, price, params)
            else:
                order = auth_ex.create_market_order(symbol, side, amount, params)
            oid       = order.get("id", "?")
            fill_price = float(order.get("average") or order.get("price") or price or 0)
            console.print(f"[green]  Orden {order_type.upper()} {side.upper()} {symbol}: {amount:.6f} @ fill={fill_price:.4f} | ID {oid}[/green]")
            return oid, fill_price
        except Exception as e:
            console.print(f"[red]  Error al colocar orden {symbol} {side}: {e}[/red]")
            return None, None

    while True:
        try:
            ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ── Refrescar saldo ──────────────────────────────────────────────
            try:
                bal = auth_ex.fetch_balance()
                usdt_free  = bal.get("USDT", {}).get("free",  0.0) or 0.0
                usdt_total = bal.get("USDT", {}).get("total", 0.0) or 0.0
            except Exception:
                usdt_free = usdt_total = 0.0

            # ── KILL SWITCH: DD de portfolio > 20% ────────────────────────────
            capital_peak = max(capital_peak, usdt_total)
            dd_portfolio = (usdt_total - capital_peak) / capital_peak if capital_peak > 0 else 0
            if dd_portfolio <= -MAX_PORTFOLIO_DD and usdt_total > 0:
                msg = (f"🚨 *KILL SWITCH ACTIVADO*\n"
                       f"DD portfolio: `{dd_portfolio*100:.1f}%` (límite: -{MAX_PORTFOLIO_DD*100:.0f}%)\n"
                       f"Balance: `{usdt_total:.2f} USDT`\nBot PARADO automáticamente.")
                console.print(f"\n[bold red]🚨 KILL SWITCH! DD={dd_portfolio*100:.1f}% — Bot deteniendo...[/bold red]")
                notifier.send_message(msg)
                break

            table = Table(
                title=f"Estado — {ahora}  |  Bybit EU Spot  |  USDT libre: {usdt_free:.2f}  |  Maker 0.10%",
                box=box.SIMPLE_HEAVY
            )
            table.add_column("Activo",      style="cyan")
            table.add_column("Estrategia",  style="dim")
            table.add_column("Precio",      justify="right")
            table.add_column("Señal",       justify="center")
            table.add_column("Posicion",    justify="center")
            table.add_column("SL",          justify="right")
            table.add_column("TP",          justify="right")
            table.add_column("PnL neto",    justify="right")

            for slot in PORTFOLIO:
                symbol = slot["symbol"]
                fn     = STRATEGY_FNS[slot["fn"]]
                sl_pct = slot["sl_pct"]
                tp_pct = slot["tp_pct"]
                slot_usdt = slot["capital"]

                df = fetch_candles(market_ex, symbol, limit=250)
                if df.empty:
                    continue

                price = df["close"].iloc[-1]
                pos   = open_positions.get(symbol)

                # ── Comprobar si posicion abierta ha llegado a SL o TP ───────
                if pos:
                    entry  = pos["entry"]
                    sl_lvl = pos["sl"]
                    tp_lvl = pos["tp"]
                    pnl    = (price - entry) / entry * 100
                    pnl_eur= pos["amount"] * (price - entry) - pos["amount"] * entry * MAKER_FEE * 2

                    pos_text = f"[yellow]ABIERTA @ {entry:.2f}[/yellow]"
                    net_text = f"{pnl:+.2f}%"

                    hit_sl = price <= sl_lvl
                    hit_tp = price >= tp_lvl

                    if hit_tp or hit_sl:
                        result  = "TP" if hit_tp else "SL"
                        outcome = "GANANCIA" if hit_tp else "PERDIDA"
                        r_color = "green" if hit_tp else "red"
                        console.print(f"\n[bold {r_color}]{result} ALCANZADO en {symbol}! {outcome}: {pnl_eur:+.2f} USDT ({pnl:+.2f}%)[/bold {r_color}]")

                        # Vender a mercado para garantizar ejecucion inmediata
                        _oid, fill_sell = place_order(symbol, "sell", pos["amount"], order_type="market")
                        log_trade(symbol, "sell", "market", pos["amount"],
                                  expected=price, fill=fill_sell or price,
                                  sl=sl_lvl, tp=tp_lvl, result=result,
                                  pnl_usdt=pnl_eur, balance=usdt_free)
                        notifier.notify_close(
                            asset=symbol,
                            outcome=result,
                            pnl_pct=pnl,
                            pnl_eur=round(pnl_eur, 2),
                            balance=usdt_free
                        )
                        del open_positions[symbol]
                        pos = None
                        pos_text = "[dim]CERRADA[/dim]"
                        net_text = f"[bold {r_color}]{pnl:+.2f}%[/bold {r_color}]"

                    sig_text = "[dim]-[/dim]"
                    table.add_row(symbol, slot["strategy"], f"{price:.2f}",
                                  sig_text, pos_text,
                                  f"{sl_lvl:.2f}", f"{tp_lvl:.2f}", net_text)
                    continue

                # ── Sin posicion abierta: evaluar nueva señal ─────────────────
                sig, sl, tp = fn(df, sl_pct, tp_pct)

                sig_text  = "[bold green]LONG[/bold green]" if sig == 1 else "[dim]ESPERANDO[/dim]"
                net_tp    = f"+{(tp_pct - MAKER_FEE*2)*100:.2f}%" if sig != 0 else "-"
                sl_text   = f"{sl:.2f}" if sig != 0 else "-"
                tp_text   = f"{tp:.2f}" if sig != 0 else "-"

                table.add_row(symbol, slot["strategy"], f"{price:.2f}",
                              sig_text, "[dim]SIN POSICION[/dim]",
                              sl_text, tp_text, net_tp)

                if sig == 1 and usdt_free >= 10:
                    # ── GUARDARAIL: máx posiciones simultáneas ──────────────────
                    if len(open_positions) >= MAX_OPEN_POSITIONS:
                        console.print(f"[dim]  ⚠ Señal en {symbol} bloqueada: ya hay {len(open_positions)}/{MAX_OPEN_POSITIONS} posiciones abiertas (guardarrail correlación)[/dim]")
                        continue

                    net_gain = (tp_pct - MAKER_FEE * 2) * 100
                    # Calcular cantidad a comprar (riesgo del 10% del slot)
                    risk_usdt     = min(slot_usdt, usdt_free) * 0.10
                    position_usdt = min(risk_usdt / sl_pct, min(slot_usdt, usdt_free) * 0.5)
                    amount_crypto = round(position_usdt / price, 6)

                    console.print(
                        f"\n[bold green]SENAL LONG en {symbol}![/bold green]\n"
                        f"  Estrategia:    {slot['strategy']}\n"
                        f"  Precio:        {price:.4f}\n"
                        f"  Cantidad:      {amount_crypto:.6f} ({position_usdt:.2f} USDT)\n"
                        f"  Stop Loss:     {sl:.4f} ({sl_pct*100:.1f}%)\n"
                        f"  Take Profit:   {tp:.4f} ({tp_pct*100:.1f}%)\n"
                        f"  Ganancia neta si TP: +{net_gain:.2f}%\n"
                        f"  Fee Maker:     {MAKER_FEE*2*100:.2f}% (entrada + salida)"
                    )

                    notifier.notify_signal(
                        asset=symbol, strategy=slot["strategy"],
                        action="BUY", price=price, sl=sl, tp=tp
                    )

                    # Colocar orden limite de compra (Maker — 0.10% fee)
                    oid, fill_buy = place_order(symbol, "buy", amount_crypto, price=price, order_type="limit")

                    if oid:
                        log_trade(symbol, "buy", "limit", amount_crypto,
                                  expected=price, fill=fill_buy or price,
                                  sl=sl, tp=tp, result="OPEN", balance=usdt_free)
                        open_positions[symbol] = {
                            "status":       "open",
                            "entry":        fill_buy or price,
                            "sl":           sl,
                            "tp":           tp,
                            "amount":       amount_crypto,
                            "buy_order_id": oid,
                        }

            console.print(table)
            slots_abiertos = len(open_positions)
            dd_txt = f"[red]{dd_portfolio*100:.1f}%[/red]" if dd_portfolio < -0.05 else f"[green]{dd_portfolio*100:.1f}%[/green]"
            console.print(f"[dim]Posiciones: {slots_abiertos}/{MAX_OPEN_POSITIONS} | DD portfolio: {dd_txt} | Proxima revision en 5 min...[/dim]\n")
            time.sleep(300)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Bot detenido por el usuario.[/bold yellow]")
            notifier.send_message("*Bot detenido* por el usuario.")
            break
        except Exception as e:
            console.print(f"[red]Error en ciclo: {e}[/red]")
            time.sleep(30)

if __name__ == "__main__":
    run_bot()
