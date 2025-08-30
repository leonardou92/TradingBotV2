from bot_utils.telegram_utils import enviar_telegram_mensaje

if __name__ == "__main__":
    print("Probando envío de mensaje de Telegram...")
    enviar_telegram_mensaje("🔔 Prueba directa: ¿Recibes este mensaje en Telegram?")
    print("Fin de la prueba de Telegram.")
