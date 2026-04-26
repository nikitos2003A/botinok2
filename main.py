# main.py
import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from deriv_api import DerivAPI
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

# ---------- Переменные окружения ----------
DERIV_TOKEN = os.getenv("pat_591100809b3c11d68bc7593bc571ec38736a157eba47f44dd878ce8d51cca74d")
DERIV_SYMBOL = os.getenv("DERIV_SYMBOL", "frxEURUSD")
DERIV_STAKE = float(os.getenv("DERIV_STAKE", "1.0"))
DERIV_DURATION = int(os.getenv("DERIV_DURATION", "5"))
DERIV_DURATION_UNIT = os.getenv("DERIV_DURATION_UNIT", "m")  # m - минуты, t - тики
TELEGRAM_TOKEN = os.getenv("8257190084:AAGxAo0YFHAfSTk9q8Nx-BBRlTiVaEvb6gI")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
MA_LENGTH = int(os.getenv("MA_LENGTH", "5"))
MA_TYPE = int(os.getenv("MA_TYPE", "1"))  # 1..7

# ---------- Проверки ----------
if not DERIV_TOKEN:
    raise ValueError("❌ Не задан DERIV_API_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("❌ Не задан TELEGRAM_BOT_TOKEN")
if TELEGRAM_CHAT_ID == 0:
    raise ValueError("❌ Не задан TELEGRAM_CHAT_ID")

# ---------- Настройка логирования ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------- Глобальное хранилище статистики ----------
stats = {
    "total": 0,
    "wins": 0,
    "losses": 0,
    "pending": 0,
    "signals": []
}

# ---------- Вспомогательные индикаторы ----------
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
        # RMA (RSI moving average)
        return series.ewm(alpha=1/length, adjust=False).mean()
    elif ma_type == 6:
        # VWMA – если объема нет, считаем как WMA (все веса=1)
        # В Deriv для Forex объем всегда 1, используем WMA
        return series.rolling(window=length).apply(lambda x: np.average(x, weights=range(1, length+1)), raw=True)
    elif ma_type == 7:
        return series.rolling(window=length).mean()
    else:
        return series.rolling(window=length).mean()  # fallback SMA

# ---------- Определение сигнала ----------
def generate_signal(closes: pd.Series, ma_type: int, length: int) -> Optional[str]:
    if len(closes) < length + 2:
        return None
    ma = calc_ma(closes, ma_type, length)
    # Последние два значения
    ma_prev, ma_curr = ma.iloc[-2], ma.iloc[-1]
    close_curr = closes.iloc[-1]
    close_prev = closes.iloc[-2]

    trend_up = ma_curr > ma_prev
    trend_down = ma_curr < ma_prev

    # Текущие условия
    long_cond = trend_up and close_curr > ma_curr
    short_cond = trend_down and close_curr < ma_curr

    # Предыдущее состояние тренда и пересечения (для избежания повтора)
    prev_trend_up = ma_prev > ma.iloc[-3] if len(ma) >= 3 else False
    prev_trend_down = ma_prev < ma.iloc[-3] if len(ma) >= 3 else False
    prev_long = prev_trend_up and close_prev > ma_prev
    prev_short = prev_trend_down and close_prev < ma_prev

    if long_cond and not prev_long:
        return "LONG"
    elif short_cond and not prev_short:
        return "SHORT"
    return None

# ---------- Работа с Deriv API ----------
class DerivTrader:
    def __init__(self):
        self.api = DerivAPI(app_id=11780)  # app_id для тестового окружения Deriv
        self.symbol = DERIV_SYMBOL
        self.running = False
        self.candles = pd.DataFrame(columns=["close"])
        self.long_pos_flag = False
        self.short_pos_flag = False

    async def connect(self):
        self.connection = await self.api.connect()
        logger.info("✅ Подключено к Deriv API")

    async def authorize(self):
        authorize = await self.connection.authorize(DERIV_TOKEN)
        if authorize.get("error"):
            raise Exception(f"Ошибка авторизации: {authorize['error']}")
        logger.info("✅ Авторизовано")

    async def get_history(self, count=50):
        """Загружаем исторические свечи (минутки)"""
        response = await self.connection.ticks_history({
            "ticks_history": self.symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "style": "candles",
            "granularity": 60  # 1 минута
        })
        if response.get("error"):
            raise Exception(f"Ошибка истории: {response['error']}")
        candles = response["candles"]
        closes = [c["close"] for c in candles]
        self.candles = pd.DataFrame({"close": closes})
        logger.info(f"📊 Загружено {len(closes)} свечей")

    async def subscribe_candles(self):
        """Подписка на новые свечи в реальном времени"""
        source = await self.connection.subscribe({
            "ticks": self.symbol
        })
        async for tick in source:
            if not self.running:
                break
            # В потоке ticks мы получаем текущую цену, но для свечей надо агрегировать
            # Deriv даёт тики с полем 'quote', можно формировать минутные свечи вручную
            # Однако для простоты подпишемся сразу на candles через ticks_history подписку
            # Но в deriv-api нет прямой подписки на свечи. Используем ticks и строим свечи.
            await self.handle_tick(tick)

    async def handle_tick(self, tick):
        """Строим минутную свечу из тиков"""
        # Используем упрощённый подход: каждый новый тик обновляет текущую свечу
        # Для точности лучше использовать подписку на свечи через другой метод
        pass

    async def start_trading(self):
        """Основной цикл: загружаем историю, затем каждую минуту получаем новую свечу"""
        await self.get_history(50)
        self.running = True
        last_candle_time = None
        while self.running:
            # Запрашиваем последнюю завершённую свечу (1m)
            response = await self.connection.ticks_history({
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
            # Проверяем, не обработали ли мы уже эту свечу
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
                await self.place_trade(signal)
                await send_telegram_message(f"📢 Сигнал {signal} по {DERIV_SYMBOL} в {datetime.now().strftime('%H:%M:%S')}")
            await asyncio.sleep(2)  # задержка между проверками

    async def place_trade(self, direction: str):
        contract_type = "CALL" if direction == "LONG" else "PUT"
        try:
            proposal = await self.connection.proposal({
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
            buy = await self.connection.buy({
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

            # Ждём завершения контракта
            await self.monitor_contract(contract_id)

        except Exception as e:
            logger.error(f"Исключение в place_trade: {e}")

    async def monitor_contract(self, contract_id):
        """Отслеживаем результат контракта"""
        try:
            # Подписываемся на обновления открытого контракта
            subscription = await self.connection.subscribe({
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

# ---------- Telegram бот ----------
async def send_telegram_message(text: str):
    if TELEGRAM_CHAT_ID:
        try:
            app = Application.builder().token(TELEGRAM_TOKEN).build()
            await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения в Telegram: {e}")

# ---------- Команды Telegram ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Бот для автотрейдинга на Deriv запущен.\n"
                                    "Используйте /stats для просмотра статистики.")

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
    await trader.connect()
    await trader.authorize()

    # Запускаем Telegram бота в фоне
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats_command))

    # Фоновая задача торговли
    asyncio.create_task(trader.start_trading())
    # Запуск поллинга Telegram
    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
