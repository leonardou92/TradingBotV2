import os
from dotenv import load_dotenv, find_dotenv
import sys
import asyncio
# Cargar .env explícitamente desde el path del proyecto
load_dotenv(find_dotenv(), override=True)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

print(f"[DEBUG] TELEGRAM_BOT_TOKEN: {TELEGRAM_BOT_TOKEN}")
print(f"[DEBUG] TELEGRAM_CHAT_ID: {TELEGRAM_CHAT_ID}")

try:
    from telegram import Bot
except ImportError:
    Bot = None
    print("Advertencia: No se pudo importar 'telegram'. Instala python-telegram-bot para notificaciones.")

def enviar_telegram_mensaje(mensaje):
    if Bot is None or TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
        print("[TELEGRAM] No configurado correctamente. Mensaje:", mensaje)
        print(f"Bot: {Bot}, TOKEN: {TELEGRAM_BOT_TOKEN}, CHAT_ID: {TELEGRAM_CHAT_ID}")
        sys.stdout.flush()
        return
    async def _enviar():
        print(f"[TELEGRAM] Intentando enviar mensaje a chat_id={TELEGRAM_CHAT_ID} con token={str(TELEGRAM_BOT_TOKEN)[:10]}... (truncado)")
        sys.stdout.flush()
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        resp = await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=mensaje)
        print("[TELEGRAM] Notificación enviada. Respuesta:", resp)
        sys.stdout.flush()
    try:
        try:
            loop = asyncio.get_running_loop()
            # Si ya hay un loop, crea una tarea
            loop.create_task(_enviar())
        except RuntimeError:
            # Si no hay loop, ejecuta normalmente
            asyncio.run(_enviar())
    except Exception as e:
        import traceback
        print(f"[TELEGRAM] Error al enviar mensaje: {e}")
        traceback.print_exc()
        sys.stdout.flush()
