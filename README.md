# 🤖 Trading Bot — Bybit EU Spot

Bot de trading algorítmico para Bybit EU (mercado Spot, sin derivados) optimizado para cuentas reguladas en la UE.

## Estrategias activas

Validadas con **walk-forward out-of-sample** sobre 1 año de datos reales (8782 velas 1H):

| Activo | Estrategia | SL | TP | R:R | Win Rate | PnL neto/año | Profit Factor |
|:------:|:----------:|:--:|:--:|:---:|:--------:|:------------:|:-------------:|
| ETH/USDT | VWAP diario + Volumen (×1.5) | 2% | 5% | 1:2.5 | 39% | +9.9% | 1.38 |
| SOL/USDT | EMA 20/50 cruce + ADX > 25 | 2% | 8% | 1:4 | 27% | +4.6% | 1.27 |

> Rendimiento anual estimado neto con comisiones Maker 0.10%: **~+9.6%**

## Características

- ✅ **Comisión Maker 0.10%** — solo órdenes límite (nunca Taker)
- ✅ **Walk-forward validado** — parámetros fijos, no overfitting
- ✅ **Kill switch DD > 20%** — el bot para automáticamente
- ✅ **Máx. 1 posición simultánea** — evita correlación en crash
- ✅ **Log de slippage** en `trade_log.csv` (expected vs fill real)
- ✅ **Alertas Telegram** en cada señal, apertura y cierre
- ✅ **Timeframe 1H** — reduce ruido vs 15m y baja impacto de comisiones

## Instalación

```bash
git clone https://github.com/TU_USUARIO/tradingalgorimitoc.git
cd tradingalgorimitoc
pip install -r requirements.txt
```

## Configuración

Copia el archivo de ejemplo y rellena tus credenciales:

```bash
cp .env.example .env
```

Edita `.env`:

```ini
MODE=real                          # "demo" para testnet, "real" para cuenta real
BYBIT_REAL_API_KEY=TU_API_KEY
BYBIT_REAL_API_SECRET=TU_API_SECRET
TELEGRAM_BOT_TOKEN=TU_BOT_TOKEN   # Opcional
TELEGRAM_CHAT_ID=TU_CHAT_ID       # Opcional
```

> ⚠️ **Nunca subas tu `.env` a GitHub.** Está en `.gitignore` por defecto.

## Uso

```bash
# Ejecutar el bot (modo configurado en .env)
python bot_bybit.py

# Descargar datos históricos 1H
python fetch_data.py

# Ejecutar simulación completa (5 estrategias × 3 activos)
python sim_1h_optimized.py

# Validación walk-forward out-of-sample
python walk_forward_test.py
```

## Estructura del proyecto

```
tradingalgorimitoc/
├── bot_bybit.py            # Bot principal — lógica completa de trading
├── sim_1h_optimized.py     # Grid search 1H (5 estrategias × 240 combinaciones)
├── walk_forward_test.py    # Validación out-of-sample (split 50/50 por tiempo)
├── strategies.py           # Funciones de señal (referencia)
├── telegram_alerts.py      # Módulo de alertas Telegram
├── fetch_data.py           # Descarga de datos OHLCV desde Bybit
├── optimizer_grid.py       # Optimizador de parámetros SL/TP
├── Pinescripts/            # Scripts de TradingView equivalentes
├── .env.example            # Plantilla de configuración
├── .gitignore
├── requirements.txt
└── trade_log.csv           # Generado en ejecución (ignorado por git)
```

## Metodología de backtesting

1. **Descarga** — datos 1H desde Bybit (1 año, ~8782 velas por activo)
2. **Grid search** — 240 combinaciones (5 estrategias × 4 SL × 4 multiplicadores TP × 3 activos)
3. **Walk-forward** — split 50/50 temporal: primeros 6 meses = in-sample, segundos 6 = out-of-sample
4. **Criterio de validación** — PF ≥ 1.10 y Max DD ≥ -20% en out-of-sample
5. **BTC descartado** — PF 1.07 out-of-sample (no supera umbral)

## Gestión de riesgo

```
Capital total:        120 USDT
  ETH slot:           40 USDT  (operativo)
  SOL slot:           40 USDT  (operativo)
  Reserva USDT:       40 USDT  (esperando oportunidad)

Por operación:
  Riesgo máximo:      10% del slot = 4 USDT
  Tamaño posición:    máx 50% del slot = 20 USDT
  Fee round-trip:     0.20% (Maker entrada + Maker salida)
```

## Requisitos

- Python 3.10+
- Cuenta en [Bybit EU](https://bybit.eu) (regulada para la UE)
- API Key con permisos de **Trade** en cuenta Spot Unificada
- El saldo USDT debe estar en la **Cuenta Unificada de Trading** (no en Funding)

## ⚠️ Aviso legal

Este software es para uso educativo y de investigación. El trading de criptomonedas conlleva riesgo de pérdida de capital. Los resultados históricos no garantizan rendimientos futuros. Úsalo bajo tu propia responsabilidad.

---

*Desarrollado con datos reales de Bybit EU · Comisiones Maker 0.10% · Timeframe 1H · Walk-forward validado*
