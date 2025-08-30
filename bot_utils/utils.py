import time
from datetime import datetime

def fetch_ohlcv_safe(exchange, symbol, timeframe, limit):
    for intento in range(3):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            print(f"[ERROR] Fallo al obtener OHLCV ({intento+1}/3): {e}")
            time.sleep(5)
    raise RuntimeError("No se pudo obtener datos OHLCV tras 3 intentos.")

def save_plot_snapshot(fig, filename_prefix="snapshot"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.png"
    fig.savefig(filename)
    print(f"[SNAPSHOT] Gráfico guardado: {filename}")
