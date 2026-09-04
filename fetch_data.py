"""
fetch_data.py  (fixed: UTF-8, rate-limit, datetime)
"""

import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
import sys

# forzar UTF-8 en stdout para Windows
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SYMBOLS = {
    "BTC": "BTC/USDT:USDT",
    "ETH": "ETH/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
    "LTC": "LTC/USDT:USDT",
}

TIMEFRAME = "15m"
DAYS_BACK = 365
DATA_DIR  = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def fetch_ohlcv(symbol: str, timeframe: str = "15m", days: int = 365) -> pd.DataFrame:
    exchange = ccxt.bybit({"options": {"defaultType": "linear"}})

    # Fix: datetime.now(timezone.utc) en lugar del deprecado utcnow()
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_bars = []
    limit    = 200   # reducido para evitar rate-limit

    print(f"  Descargando {symbol} [{timeframe}] {days} dias...", flush=True)

    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                        since=since_ms, limit=limit)
        except ccxt.RateLimitExceeded:
            print("  Rate limit - esperando 10s...", flush=True)
            time.sleep(10)
            continue
        except Exception as e:
            raise e

        if not bars:
            break
        all_bars.extend(bars)
        since_ms = bars[-1][0] + 1
        if len(bars) < limit:
            break
        time.sleep(1.0)   # 1s entre llamadas = seguro con Bybit

    df = pd.DataFrame(all_bars, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    print(f"  OK: {len(df)} velas descargadas", flush=True)
    return df


def main():
    print("=" * 50)
    print("  Descarga de datos historicos - Bybit Futures")
    print("=" * 50)

    for name, symbol in SYMBOLS.items():
        csv_path = DATA_DIR / f"{name}_{TIMEFRAME}.csv"

        if csv_path.exists():
            existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            print(f"  {name}: ya existe ({len(existing)} velas) - omitiendo.")
            continue

        try:
            df = fetch_ohlcv(symbol, TIMEFRAME, DAYS_BACK)
            df.to_csv(csv_path)
            print(f"  {name}: guardado en {csv_path}")
        except Exception as e:
            print(f"  ERROR en {name}: {e}")

        time.sleep(2)   # pausa extra entre activos

    print("\nDescarga completa. Archivos en /data/")


if __name__ == "__main__":
    main()
