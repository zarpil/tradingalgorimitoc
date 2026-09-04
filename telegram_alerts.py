"""
telegram_alerts.py
──────────────────
Módulo ligero para enviar alertas a Telegram vía API oficial de bots de Telegram.
No requiere dependencias complejas (usa urllib nativo de Python).
"""

import urllib.request
import urllib.parse
import json
import logging

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)

    def send_message(self, message: str) -> bool:
        if not self.enabled:
            # Si no está configurado, muestra por consola sin bloquear
            print(f"[TELEGRAM SIMULADO]: {message}")
            return True

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8')
            # Si falla por markdown, reintentar sin formato
            try:
                payload.pop("parse_mode", None)
                data_plain = json.dumps(payload).encode("utf-8")
                req_plain = urllib.request.Request(url, data=data_plain, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req_plain, timeout=10) as response2:
                    return response2.status == 200
            except Exception:
                logger.error(f"Error Telegram: {err_msg}")
                print(f"[Aviso Telegram]: {err_msg}")
                return False
        except Exception as e:
            logger.error(f"Error enviando mensaje a Telegram: {e}")
            return False

    def notify_signal(self, asset: str, strategy: str, action: str, price: float, sl: float, tp: float):
        emoji = "🟢 LONG" if action.upper() == "BUY" else "🔴 SHORT"
        msg = (
            f"⚡ *NUEVA SEÑAL DETECTADA*\n\n"
            f"*Activo:* `{asset}`\n"
            f"*Estrategia:* `{strategy}`\n"
            f"*Acción:* {emoji}\n"
            f"*Precio Entrada:* `{price:.4f}`\n"
            f"*Stop Loss (SL):* `{sl:.4f}`\n"
            f"*Take Profit (TP):* `{tp:.4f}`\n"
        )
        return self.send_message(msg)

    def notify_close(self, asset: str, outcome: str, pnl_pct: float, pnl_eur: float, balance: float):
        emoji = "🎯 TP ALCANZADO" if "TP" in outcome.upper() else "🛑 STOP LOSS"
        msg = (
            f"{emoji}\n\n"
            f"*Activo:* `{asset}`\n"
            f"*Resultado:* `{outcome}`\n"
            f"*PnL Operación:* `{pnl_pct:+.2f}%` ({pnl_eur:+.2f} EUR)\n"
            f"*Balance Cartera:* `{balance:.2f} EUR`\n"
        )
        return self.send_message(msg)
