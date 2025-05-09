import logging
import os
from dotenv import load_dotenv
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# Загрузка переменных из .env
load_dotenv()

API_URL = os.getenv("MARK_API_URL", "http://127.0.0.1:8000/greet")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Напиши свой вопрос.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    response = requests.post(API_URL, json={"text": user_text})
    data = response.json()
    results = data.get("results", [])
    if not results:
        await update.message.reply_text("Нет подходящих тем.")
        return
    keyboard = [
        [InlineKeyboardButton(r["topic"], callback_data=f"show_{i}")] for i, r in enumerate(results)
    ]
    context.user_data["results"] = results
    await update.message.reply_text(
        "Выберите тему:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data.startswith("show_"):
        idx = int(data.split("_")[1])
        results = context.user_data.get("results", [])
        if 0 <= idx < len(results):
            full_text = results[idx].get("full_text", "Текст не найден")
            keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]
            await query.edit_message_text(
                text=full_text,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
    elif data == "back":
        results = context.user_data.get("results", [])
        keyboard = [
            [InlineKeyboardButton(r["topic"], callback_data=f"show_{i}")] for i, r in enumerate(results)
        ]
        await query.edit_message_text(
            text="Выберите тему:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Установите переменную окружения TELEGRAM_BOT_TOKEN")
        return
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button))
    print("Бот запущен!")
    app.run_polling()

if __name__ == "__main__":
    main()
