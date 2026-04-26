# main.py
import os
import asyncio
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from deriv_api import DerivAPI
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

# ---------- Переменные окружения ----------
DERIV_TOKEN = os.getenv("DERIV_API_TOKEN")
DERIV_SYMBOL = os.getenv("DERIV_SYMBOL", "frxEURUSD")
DERIV_STAKE = float(os.getenv("DERIV_STAKE", "1.0"))
DERIV_DURATION = int(os.getenv("DERIV_DURATION", "5"))
DERIV_DURATION_UNIT = os.getenv("DERIV_DURATION_UNIT", "m")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MA_LENGTH = int(os.getenv("MA_LENGTH", "5"))
MA_TYPE = int(os.getenv("MA_TYPE", "1"))

if not DERIV_TOKEN:
    raise ValueError("❌ Не задан DERIV_API_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("❌ Не задан TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

admin_chat_id: Optional[int] = None
stats = {
    "total": 0,
    "wins": 0,
    "losses": 0,
    "pending": 0,
    "signals": []
}

# ---------- Индикаторы ----------
def mod_hull(src: pd.Series, length: int) -> pd.Series:
    half = length // 2
    sqrt_len = int(round(np.sqrt(length)))
    wma_half = src.rolling(window=half).apply(lambda x: np.average(x, weights=range(1, half+1)), raw=True)
    wma_full = src.rolling(window=length).apply(lambda x: np.average(x, weights=range(1, length+1)), raw=True)
    raw = 2 * wma_half - wma_full
    return raw.rolling(window=sqrt_len).mean()

def hull(src: pd.Series, length: int) -> pd.Series:
    half = length // 2
    sqrt_len = int(round(np.sqrt(length)))
    wma_half = src.rolling(window=half).apply(lambda x: np.average(x, weights=range(1, half+1)), raw=True)
    wma_full = src.rolling(window=length).apply(lambda x: np.average(x, weights=range(1, length+1)), raw=True)
    raw = 2 * wma_half - wma_full
    return raw.rolling(window=sqrt_len).apply(lambda x: np.average(x, weights=range(1, sqrt_len+1)), raw=True)

def calc_ma(series: pd.Series, ma_type: int, length: int) -> pd.Series:
    if ma_type == 1:
        return mod_hull(series, length)
    elif ma_type == 2:
        return hull(series, length)
    elif ma_type == 3:
        return series.ewm(span=length, adjust=False).mean()
    elif ma_type == 4:
        return series.rolling(window=length).apply(lambda x: np.average(x, weights=range(1, length+1)), raw=True)
    elif ma_type == 5:
        return series.ewm(alpha=1/length, adjust=False).mean()
    elif ma_type == 6:
        return series.rolling(window=length).apply(lambda x: np.average(x, weights=range(1, length+1)), raw=True)
    elif ma_type == 7:
        return series.rolling(window=length).mean()
    else:
        return series.rolling(window=length).mean()

def generate_signal(closes: pd.Series, ma_type: int, length: int) -> Optional[str]:
    if len(closes) < length + 2:
        return None
    ma = calc_ma(closes, ma_type, length)
    ma_prev, ma_curr = ma.iloc[-2], ma.iloc[-1]
    close_curr = closes.iloc[-1]
    close_prev = closes.iloc[-2]

    trend_up = ma_curr > ma_prev
    trend_down = ma_curr < ma_prev

    long_cond = trend_up and close_curr > ma_curr
    short_cond = trend_down and close_curr < ma_curr

    prev_trend_up = ma_prev > ma.iloc[-3] if len(ma) >= 3 else False
    prev_trend_down = ma_prev < ma.iloc[-3] if len(ma) >= 3 else False
    prev_long = prev_trend_up and close_prev > ma_prev
    prev_short = prev_trend_down and close_prev < ma_prev

    if long_cond and not prev_long:
        return "LONG"
    elif short_cond and not prev_short:
        return "SHORT"
    return None

# ---------- Работа с Deriv ----------
class DerivTrader:
    def __init__(self):
        self.api = DerivAPI(app_id=11780)  # Официальный объект, connect не нужен
        self.symbol = DERIV_SYMBOL
        self.running = False
        self.candles = pd.DataFrame(columns=["close"])

    async def authorize(self):
        authorize = await self.api.authorize(DERIV_TOKEN)
        if authorize.get("error"):
            raise Exception(f"Ошибка авторизации: {authorize['error']}")
        logger.info("✅ Авторизовано")

    async def get_history(self, count=50):
        response = await self.api.ticks_history({
            "ticks_history": self.symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "style": "candles",
            "granularity": 60
        })
        if response.get("error"):
            raise Exception(f"Ошибка истории: {response['error']}")
        candles = response["candles"]
        closes = [c["close"] for c in candles]
        self.candles = pd.DataFrame({"close": closes})
        logger.info(f"📊 Загружено {len(closes)} свечей")

    async def start_trading(self):
        await self.get_history(50)
        self.running = True
        last_candle_time = None
        while self.running:
            response = await self.api.ticks_history({
                "ticks_history": self.symbol,
                "adjust_start_time": 1,
                "count": 1,
                "end": "latest",
                "style": "candles",
                "granularity": 60
            })
            if response.get("error"):
                logger.error(f"Ошибка получения свечи: {response['error']}")
                await asyncio.sleep(5)
                continue

            candle = response["candles"][0]
            epoch = int(candle["epoch"])
            close = float(candle["close"])
            if last_candle_time == epoch:
                await asyncio.sleep(1)
                continue
            last_candle_time = epoch

            logger.info(f"Новая свеча: время={datetime.fromtimestamp(epoch)}, close={close}")
            self.candles = pd.concat([self.candles, pd.DataFrame({"close": [close]})], ignore_index=True)
            if len(self.candles) > 100:
                self.candles = self.candles.iloc[-100:]

            signal = generate_signal(self.candles["close"], MA_TYPE, MA_LENGTH)
            if signal:
                logger.info(f"🔔 Сигнал: {signal}")
                await send_telegram_message(f"📢 Сигнал {signal} по {DERIV_SYMBOL} в {datetime.now().strftime('%H:%M:%S')}")
                await self.place_trade(signal)
            await asyncio.sleep(2)

    async def place_trade(self, direction: str):
        contract_type = "CALL" if direction == "LONG" else "PUT"
        try:
            proposal = await self.api.proposal({
                "proposal": 1,
                "amount": DERIV_STAKE,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": DERIV_DURATION,
                "duration_unit": DERIV_DURATION_UNIT,
                "symbol": self.symbol,
            })
            if proposal.get("error"):
                logger.error(f"Ошибка proposal: {proposal['error']}")
                return

            proposal_id = proposal["proposal"]["id"]
            buy = await self.api.buy({
                "buy": proposal_id,
                "price": DERIV_STAKE
            })
            if buy.get("error"):
                logger.error(f"Ошибка покупки: {buy['error']}")
                return

            contract_id = buy["buy"]["contract_id"]
            logger.info(f"✅ Контракт {contract_id} {contract_type} открыт")
            stats["total"] += 1
            stats["pending"] += 1
            stats["signals"].append({
                "time": datetime.now(),
                "contract_id": contract_id,
                "direction": direction,
                "symbol": DERIV_SYMBOL,
                "stake": DERIV_STAKE,
                "duration": DERIV_DURATION,
                "unit": DERIV_DURATION_UNIT
            })

            await self.monitor_contract(contract_id)

        except Exception as e:
            logger.error(f"Исключение в place_trade: {e}")

    async def monitor_contract(self, contract_id):
        try:
            subscription = await self.api.subscribe({
                "proposal_open_contract": 1,
                "contract_id": contract_id
            })
            async for update in subscription:
                if update.get("proposal_open_contract"):
                    poc = update["proposal_open_contract"]
                    if poc.get("is_sold"):
                        profit = poc.get("profit", 0)
                        if profit > 0:
                            stats["wins"] += 1
                            result = "WIN ✅"
                        else:
                            stats["losses"] += 1
                            result = "LOSS ❌"
                        stats["pending"] -= 1
                        logger.info(f"🏁 Контракт {contract_id} завершён: {result}, профит={profit}")
                        await send_telegram_message(
                            f"🏁 Контракт {contract_id} завершён: {result}\n"
                            f"Направление: {poc.get('contract_type')}\n"
                            f"Профит: ${profit:.2f}"
                        )
                        break
        except Exception as e:
            logger.error(f"Мониторинг ошибка: {e}")
            stats["pending"] -= 1
            stats["losses"] += 1

# ---------- Telegram ----------
async def send_telegram_message(text: str):
    global admin_chat_id
    if not admin_chat_id:
        logger.warning("❌ Админ ещё не отправил /start, сообщение не отправлено.")
        return
    try:
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        await app.bot.send_message(chat_id=admin_chat_id, text=text)
    except Exception as e:
        logger.error(f"Ошибка отправки сообщения в Telegram: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global admin_chat_id
    admin_chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"🤖 Бот для автотрейдинга на Deriv запущен.\n"
        f"Ваш chat_id: {admin_chat_id}\n"
        f"Все уведомления о сделках будут приходить сюда.\n"
        f"Используйте /stats для просмотра статистики."
    )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = stats["total"]
    wins = stats["wins"]
    losses = stats["losses"]
    pending = stats["pending"]
    winrate = (wins / total * 100) if total > 0 else 0
    msg = (f"📊 Статистика:\n"
           f"Всего сделок: {total}\n"
           f"Выигрышей: {wins}\n"
           f"Проигрышей: {losses}\n"
           f"Активных: {pending}\n"
           f"Винрейт: {winrate:.1f}%")
    await update.message.reply_text(msg)

# ---------- Запуск ----------
async def main():
    trader = DerivTrader()
    await trader.authorize()  # Авторизация без connect

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats_command))

    asyncio.create_task(trader.start_trading())
    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())