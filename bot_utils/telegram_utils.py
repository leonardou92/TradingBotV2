import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

try:
    from telegram import Bot
except ImportError:
    Bot = None
    print("Advertencia: No se pudo importar 'telegram'. Instala python-telegram-bot para notificaciones.")

def enviar_telegram_mensaje(mensaje):
    if Bot is None or TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
        print("[TELEGRAM] No configurado correctamente. Mensaje:", mensaje)
        print(f"Bot: {Bot}, TOKEN: {TELEGRAM_BOT_TOKEN}, CHAT_ID: {TELEGRAM_CHAT_ID}")
        return
    try:
        print(f"[TELEGRAM] Intentando enviar mensaje a chat_id={TELEGRAM_CHAT_ID} con token={str(TELEGRAM_BOT_TOKEN)[:10]}... (truncado)")
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        resp = bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=mensaje)
        print("[TELEGRAM] Notificación enviada. Respuesta:", resp)
    except Exception as e:
        import traceback
        print(f"[TELEGRAM] Error al enviar mensaje: {e}")
        traceback.print_exc()
