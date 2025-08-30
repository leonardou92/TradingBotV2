#!/usr/bin/env python3
# --- IMPORTS MODULARES ---
from bot_utils.utils import fetch_ohlcv_safe, save_plot_snapshot
from bot_utils.indicadores import calcular_indicadores
from bot_utils.patrones import detectar_onda_elliott, detectar_fibonacci_experto
from bot_utils.estrategias import decision_cierre_ia
from bot_utils.plot_utils import plot_candles
from bot_utils.ml_utils import entrenar_modelo_rf
from bot_utils.telegram_utils import enviar_telegram_mensaje

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
import time
from datetime import datetime
import threading
from dotenv import load_dotenv
import traceback

# Cargar variables de entorno desde .env
load_dotenv()
# Control para mostrar gráficas en ventanas (evitar pop-ups en ejecución de segundo plano)
SHOW_PLOTS = os.getenv('SHOW_PLOTS', '0') == '1'

def descargar_y_entrenar_historico(symbol='ETH/USDT', timeframe='5m', total_limit=2000, chunk=500, log_path='log_operaciones_historico.json'):
    """
    Descarga históricos extensos, detecta patrones, simula entradas y entrena el modelo ML y Gemini AI.
    """
    try:
        enviar_telegram_mensaje("⬇️ Descargando histórico de datos para entrenamiento...")
    except Exception as e:
        print(f"[TELEGRAM] Error al notificar descarga: {e}")
    exchange = ccxt.binance()
    all_ohlcv = []
    since = None
    # usar fetch_ohlcv_safe (importado) que ya implementa reintentos; si no, reintentar localmente
    for _ in range(total_limit // chunk):
        try:
            ohlcv = fetch_ohlcv_safe(exchange, symbol, timeframe, limit=chunk, since=since)
        except Exception as e:
            print(f"[ERROR] fetch_ohlcv en descargar_y_entrenar_historico falló: {e}")
            traceback.print_exc()
            notify_connection_failure_once(str(e))
            # intentar seguir con lo descargado hasta ahora
            break
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        time.sleep(0.2)
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    if df.empty:
        print("[WARN] No se descargaron velas históricas. Abortando entrenamiento histórico.")
        try:
            enviar_telegram_mensaje("[ALERTA] No se pudieron descargar velas históricas para entrenamiento. Reintentando en el próximo ciclo.")
        except Exception:
            pass
        return
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = calcular_indicadores(df)
    log_ops = []
    for i in range(60, len(df)):
        subdf = df.iloc[:i].copy()
        elliott_tipo, _ = detectar_onda_elliott(subdf, lookback=30)
        fibo_tipo, _, _ = detectar_fibonacci_experto(subdf, lookback=50)
        patron = None
        tipo = None
        if elliott_tipo is not None:
            patron = f"elliott_{elliott_tipo}"
            tipo = 'long' if elliott_tipo == 'alcista' else 'short'
        elif fibo_tipo is not None:
            patron = f"fibonacci_{fibo_tipo}"
            tipo = fibo_tipo
        if tipo is not None:
            op = {
                'tipo': tipo,
                'precio_entrada': float(subdf['close'].iloc[-1]),
                'fecha_entrada': str(subdf['datetime'].iloc[-1]),
                'patron_detectado': patron,
                'prediccion': tipo
            }
            log_ops.append(op)
    with open(log_path, 'w') as f:
        json.dump(log_ops, f, indent=2)
    print(f"Histórico de operaciones simulado guardado en {log_path} ({len(log_ops)} operaciones)")
    try:
        enviar_telegram_mensaje("🤖 Entrenando modelo ML con el histórico descargado...")
    except Exception as e:
        print(f"[TELEGRAM] Error al notificar entrenamiento: {e}")
    entrenar_modelo_rf(df)
    entrenar_con_gemini(log_path)
import requests

# Configuración Gemini AI
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Cargar clave desde variable de entorno

def entrenar_con_gemini(log_path='log_operaciones.json'):
    """
    Envía el histórico de operaciones con patrones a Gemini AI y recibe recomendaciones de entrenamiento.
    """
    if not os.path.exists(log_path):
        print('No hay log de operaciones para entrenar.')
        return None
    with open(log_path, 'r') as f:
        data = json.load(f)
    # Construir el prompt para Gemini
    prompt = (
        "Eres un experto en trading algorítmico. Analiza el siguiente histórico de operaciones con patrones (Fibonacci/Elliott) y resultados. "
        "Sugiere cómo mejorar la estrategia y qué patrones son más rentables. Responde en español.\n\n"
        f"Histórico:\n{json.dumps(data, ensure_ascii=False, indent=2)}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': GEMINI_API_KEY
    }
    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            print('Gemini AI entrenamiento exitoso.')
            recomendaciones = response.json()
            print('Recomendaciones Gemini:', recomendaciones)
            return recomendaciones
        else:
            print('Error Gemini AI:', response.status_code, response.text)
            return None
    except Exception as e:
        print('Error al conectar con Gemini AI:', e)
        return None
# --- Detección básica de onda de impulso de Elliott ---
def detectar_onda_elliott(df, lookback=30):
    closes = df['close'].iloc[-lookback:]
    pivotes = []
    for i in range(2, len(closes)-2):
        if closes.iloc[i] > closes.iloc[i-2] and closes.iloc[i] > closes.iloc[i-1] and closes.iloc[i] > closes.iloc[i+1] and closes.iloc[i] > closes.iloc[i+2]:
            pivotes.append((i, 'max'))
        if closes.iloc[i] < closes.iloc[i-2] and closes.iloc[i] < closes.iloc[i-1] and closes.iloc[i] < closes.iloc[i+1] and closes.iloc[i] < closes.iloc[i+2]:
            pivotes.append((i, 'min'))
    if len(pivotes) < 5:
        return None, None
        pivotes = pivotes[-5:]
    def distancia_minima(pivotes, min_dist=2):
        return all(abs(pivotes[i][0] - pivotes[i-1][0]) >= min_dist for i in range(1, len(pivotes)))
    # Alcista: min, max, min, max, min (impulso up)
    if len(pivotes) >= 5 and [p[1] for p in pivotes[-5:]] == ['min','max','min','max','min'] and distancia_minima(pivotes[-5:]):
        base_idx = df.index[-lookback:][0]
        return 'alcista', [base_idx + p[0] for p in pivotes[-5:]]
    # Bajista: max, min, max, min, max (impulso down)
    if len(pivotes) >= 5 and [p[1] for p in pivotes[-5:]] == ['max','min','max','min','max'] and distancia_minima(pivotes[-5:]):
        base_idx = df.index[-lookback:][0]
        return 'bajista', [base_idx + p[0] for p in pivotes[-5:]]
    return None, None

# --- Detección experta de niveles de Fibonacci ---
def detectar_fibonacci_experto(df, lookback=50, min_retracement=0.382, max_retracement=0.618):
    '''
    Busca un swing alto y bajo reciente y verifica si el precio actual está reaccionando en un nivel clave de Fibonacci.
    Retorna ('long' o 'short', niveles_fibo, [idx_low, idx_high]) si hay patrón experto, si no None.
    '''
    closes = df['close'].iloc[-lookback:]
    idx_high = closes.idxmax()
    idx_low = closes.idxmin()
    price_high = closes.loc[idx_high]
    price_low = closes.loc[idx_low]
    if idx_low < idx_high:
        # Movimiento bajista, buscar retroceso para short
        move = price_low - price_high
        levels = [price_high + move * r for r in [0, min_retracement, 0.5, max_retracement, 1, 1.618]]
        current = closes.iloc[-1]
        for lvl, r in zip(levels, [0, min_retracement, 0.5, max_retracement, 1, 1.618]):
            if abs(current - lvl) / abs(move) < 0.01 and r in [min_retracement, max_retracement]:
                return 'short', levels, [idx_high, idx_low]
    elif idx_high < idx_low:
        # Movimiento alcista, buscar retroceso para long
        move = price_high - price_low
        levels = [price_low + move * r for r in [0, min_retracement, 0.5, max_retracement, 1, 1.618]]
        current = closes.iloc[-1]
        for lvl, r in zip(levels, [0, min_retracement, 0.5, max_retracement, 1, 1.618]):
            if abs(current - lvl) / abs(move) < 0.01 and r in [min_retracement, max_retracement]:
                return 'long', levels, [idx_low, idx_high]
    return None, None, None

import json
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import ccxt
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pandas as pd
import matplotlib.dates as mdates
try:
    from ta.trend import PSARIndicator
except ImportError:
    PSARIndicator = None
    print("Advertencia: No se pudo importar PSARIndicator de la librería 'ta'. El indicador PSAR no estará disponible.")

# Configuración
symbol = 'ETH/USDT'
timeframe = '5m'  # 5 minutos
limit = 200  # Últimos 200 velas

# Función para calcular indicadores
def calcular_indicadores(df):
    # Primero calcular todas las EMAs, MACD, RSI, PSAR, etc.
    df['EMA_1'] = df['close']
    df['EMA_3'] = df['close'].ewm(span=3, adjust=False).mean()
    df['EMA_4'] = df['close'].ewm(span=4, adjust=False).mean()
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['EMA_55'] = df['close'].ewm(span=55, adjust=False).mean()
    df['EMA_660'] = df['close'].ewm(span=660, adjust=False).mean()
    delta = df['EMA_5'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14_EMA5'] = 100 - (100 / (1 + rs))
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    # PSAR (Parabolic SAR)
    if PSARIndicator is not None:
        psar = PSARIndicator(df['high'], df['low'], df['close'], step=0.10, max_step=0.10)
        df['PSAR'] = psar.psar()
    else:
        df['PSAR'] = np.nan
    # Squeeze Momentum (LazyBear approximation)
    # Bollinger Bands
    bb_mid = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = bb_mid + 2 * bb_std
    df['BB_lower'] = bb_mid - 2 * bb_std
    # Keltner Channel
    ema_kelt = df['close'].ewm(span=20, adjust=False).mean()
    atr_kelt = df['TR'].rolling(window=20).mean()
    df['KC_upper'] = ema_kelt + 1.5 * atr_kelt
    df['KC_lower'] = ema_kelt - 1.5 * atr_kelt
    # Squeeze On/Off
    df['SQZ_ON'] = (df['BB_lower'] > df['KC_lower']) & (df['BB_upper'] < df['KC_upper'])
    df['SQZ_OFF'] = (df['BB_lower'] < df['KC_lower']) & (df['BB_upper'] > df['KC_upper'])
    # Momentum (diferencia de cierre)
    df['SQZMOM_LB'] = df['close'] - df['close'].rolling(window=20).mean()

    # --- Trend Meter ---
    trend_score = (
        (df['EMA_5'] > df['EMA_55']).astype(int) - (df['EMA_5'] < df['EMA_55']).astype(int) +
        (df['MACD'] > df['MACD_signal']).astype(int) - (df['MACD'] < df['MACD_signal']).astype(int) +
        (df['RSI_14_EMA5'] > 60).astype(int) - (df['RSI_14_EMA5'] < 40).astype(int) +
        (df['PSAR'] < df['close']).astype(int) - (df['PSAR'] > df['close']).astype(int)
    )
    df['TREND_METER'] = trend_score

    # --- Donchian Channel y Ribbon ---
    donchian_high = df['high'].rolling(window=20, min_periods=1).max()
    donchian_low = df['low'].rolling(window=20, min_periods=1).min()
    df['DONCHIAN_HIGH'] = donchian_high
    df['DONCHIAN_LOW'] = donchian_low
    ribbon = (df['close'] > donchian_high).astype(int) - (df['close'] < donchian_low).astype(int)
    df['DONCHIAN_RIBBON'] = ribbon
    # Garantizar que existan aunque sean NaN si hay pocos datos
    if 'DONCHIAN_HIGH' not in df.columns:
        df['DONCHIAN_HIGH'] = np.nan
    if 'DONCHIAN_LOW' not in df.columns:
        df['DONCHIAN_LOW'] = np.nan
    if 'DONCHIAN_RIBBON' not in df.columns:
        df['DONCHIAN_RIBBON'] = np.nan

    # --- EFMUS System ---
    efmus = (df['EMA_8'] > df['EMA_21']).astype(int) - (df['EMA_8'] < df['EMA_21']).astype(int)
    momentum = (df['close'] - df['close'].shift(4)).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df['EFMUS'] = efmus + momentum
    df['EFMUS_SIGNAL'] = df['EFMUS'].apply(lambda x: 2 if x >= 2 else (-2 if x <= -2 else 0))

    # --- EMAs (recalcular para asegurar consistencia) ---
    df['EMA_660'] = df['close'].ewm(span=660, adjust=False).mean()
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_4'] = df['close'].ewm(span=4, adjust=False).mean()
    df['EMA_55'] = df['close'].ewm(span=55, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_3'] = df['close'].ewm(span=3, adjust=False).mean()
    df['EMA_1'] = df['close']  # EMA de periodo 1 es igual al precio de cierre
    return df

# Inicializar exchange y cargar histórico una sola vez
symbol = 'ETH/USDT'
timeframe = '30m'  # Cambia a 30 minutos
limit = 200  # Últimos 200 velas de 30 minutos
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df = calcular_indicadores(df)

# Gráficos
if SHOW_PLOTS:
    plt.ion()
else:
    plt.ioff()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14,10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})


# Función para graficar velas japonesas
def plot_candles(ax, df, entrada_idx=None):
    """Grafica velas japonesas en el eje ax."""
    ax.clear()
    width = mdates.date2num(df['datetime'][1]) - mdates.date2num(df['datetime'][0])
    width2 = width * 0.4
    for idx, row in df.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax.add_patch(plt.Rectangle((mdates.date2num(row['datetime'])-width2/2, min(row['open'], row['close'])),
                                   width2, abs(row['close']-row['open']), color=color, alpha=0.7))
        ax.plot([mdates.date2num(row['datetime']), mdates.date2num(row['datetime'])],
                [row['low'], row['high']], color=color, linewidth=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.set_xlabel('Fecha y hora (30m)')
    ax.set_title('ETH/USDT - Velas de 30 minutos')
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    if entrada_idx is not None and 0 <= entrada_idx < len(df):
        row = df.iloc[entrada_idx]
        ax.add_patch(plt.Rectangle((mdates.date2num(row['datetime'])-width2/2, row['low']),
                                   width2, row['high']-row['low'], color='yellow', alpha=0.3, zorder=0))
    ax.xaxis_date()
    ax.set_ylabel('Precio (USDT)')
    ax.grid()



import threading



def actualizar_velas():
    global df
    while True:
        try:
            ohlcv = fetch_ohlcv_safe(exchange, symbol, timeframe, limit)
            df_local = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_local['datetime'] = pd.to_datetime(df_local['timestamp'], unit='ms')
            df_local = calcular_indicadores(df_local)
            globals()['df'] = df_local
        except Exception as e:
            print(f"[ERROR] No se pudo actualizar velas: {e}")
        time.sleep(300)  # 5 minutos

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df = calcular_indicadores(df)


# --- Entrenamiento del modelo ML encapsulado ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

features = [
    'EMA_660', 'EMA_5', 'EMA_4', 'EMA_55', 'EMA_10', 'EMA_3', 'EMA_1',
    'RSI_14_EMA5', 'ATR_14', 'MACD', 'MACD_signal',
    'PSAR', 'SQZMOM_LB', 'BB_upper', 'BB_lower', 'KC_upper', 'KC_lower', 'close', 'volume'
]

def entrenar_modelo_rf(df):
    """
    Entrena un modelo RandomForestClassifier y retorna el modelo y el score.
    """
    df_train = df.dropna()
    X = df_train[features].values
    y = df_train['target'].values
    best_score = 0
    clf = None
    if len(np.unique(y)) > 1:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 8, None],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
        grid.fit(X, y)
        clf = grid.best_estimator_
        best_score = grid.best_score_
        print(f"[ML] Modelo entrenado. Mejor score: {best_score:.4f}")
    else:
        print("[ML] No hay suficientes clases para entrenar el modelo.")
    return clf, best_score


threading.Thread(target=actualizar_velas, daemon=True).start()



# --- Simulación de operaciones ---
posicion_abierta = None
log_operaciones = []
log_path = 'log_operaciones.json'
capital = 100.0  # Capital inicial en USDT
saldo = capital
# Flags para notificaciones de conexión (evitar spam)
connection_failed = False
last_failure_notify = None
FAILURE_NOTIFY_INTERVAL = 1800  # 30 minutos

def notify_connection_failure_once(reason: str):
    global connection_failed, last_failure_notify
    now_ts = time.time()
    if last_failure_notify is None or (now_ts - last_failure_notify) >= FAILURE_NOTIFY_INTERVAL:
        try:
            enviar_telegram_mensaje(f"[ALERTA] Problema de conexión: {reason}. El bot intentará reconectar automáticamente.")
        except Exception as e:
            print(f"[TELEGRAM] Error al notificar fallo de conexión: {e}")
        last_failure_notify = now_ts
        connection_failed = True

def notify_connection_recovered_once():
    global connection_failed, last_failure_notify
    if connection_failed:
        try:
            enviar_telegram_mensaje("✅ Conexión restablecida. El bot ha vuelto a operar normalmente.")
        except Exception as e:
            print(f"[TELEGRAM] Error al notificar recuperación de conexión: {e}")
        connection_failed = False
        last_failure_notify = None




# --- FLUJO PRINCIPAL DEL BOT ---
def main():
    global df, posicion_abierta, log_operaciones, saldo
    print("Entrenando modelo ML...")
    clf, best_score = entrenar_modelo_rf(df)
    print("Modelo ML listo. Iniciando ciclo de trading...")
    features = [
        'EMA_660', 'EMA_5', 'EMA_4', 'EMA_55', 'EMA_10', 'EMA_3', 'EMA_1',
        'RSI_14_EMA5', 'ATR_14', 'MACD', 'MACD_signal',
        'PSAR', 'SQZMOM_LB', 'BB_upper', 'BB_lower', 'KC_upper', 'KC_lower', 'close', 'volume'
    ]
    abrir_operacion_snapshot = False
    cerrar_operacion_snapshot = False
    notificado_espera = False
    while True:
        # Manejo robusto de errores: reintentos con backoff para evitar caída por problemas de red
        try:
            # Recalcular indicadores para asegurar que todas las columnas existen
            df = calcular_indicadores(df)
            # reintentar fetch_ticker en caso de fallo
            retry_attempts = 0
            while True:
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    # Si antes había fallo, notificar recuperación una sola vez
                    notify_connection_recovered_once()
                    break
                except Exception as e:
                    retry_attempts += 1
                    # Notificar fallo la primera vez y luego cada FAILURE_NOTIFY_INTERVAL segundos
                    notify_connection_failure_once(str(e))
                    wait = min(60, 2 ** retry_attempts)
                    print(f"[WARN] fetch_ticker fallo (intento {retry_attempts}): {e}. Reintentando en {wait}s...")
                    traceback.print_exc()
                    time.sleep(wait)
            last_price = ticker['last']

            # Detectar señal 4EMA: EMA_1 > EMA_3 > EMA_10 > EMA_55 para long, o al revés para short
            entrada_idx = None
            ema1 = df['EMA_1'].iloc[-1]
            ema3 = df['EMA_3'].iloc[-1]
            ema10 = df['EMA_10'].iloc[-1]
            ema55 = df['EMA_55'].iloc[-1]
            print(f"EMA_1: {ema1:.2f}, EMA_3: {ema3:.2f}, EMA_10: {ema10:.2f}, EMA_55: {ema55:.2f}")
            tol = 0.01
            long_patron = (ema1 >= ema3 - tol) and (ema3 >= ema10 - tol) and (ema10 >= ema55 - tol) and (ema1 > ema3 and ema3 > ema10 and ema10 > ema55)
            short_patron = (ema1 <= ema3 + tol) and (ema3 <= ema10 + tol) and (ema10 <= ema55 + tol) and (ema1 < ema3 and ema3 < ema10 and ema10 < ema55)
            if long_patron:
                print("Señal LONG detectada por 4EMA (patrón flexible)")
                entrada_idx = len(df) - 1
            elif short_patron:
                print("Señal SHORT detectada por 4EMA (patrón flexible)")
                entrada_idx = len(df) - 1
            elif (abs(ema1-ema3)<tol or abs(ema3-ema10)<tol or abs(ema10-ema55)<tol):
                print("EMAs muy cerca, pero no cumplen el patrón exacto.")

            plot_candles(ax1, df, entrada_idx=entrada_idx)
            ax1.plot(df['datetime'], df['EMA_660'], label='EMA 660', color='blue', linewidth=1)
            ax1.plot(df['datetime'], df['EMA_5'], label='EMA 5', color='orange', linewidth=1)
            ax1.plot(df['datetime'], df['EMA_4'], label='EMA 4', color='magenta', linewidth=1)
            ax1.plot(df['datetime'], df['EMA_55'], label='EMA 55', color='red', linewidth=1, linestyle='--')
            ax1.plot(df['datetime'], df['EMA_10'], label='EMA 10', color='green', linewidth=1, linestyle='--')
            ax1.plot(df['datetime'], df['EMA_3'], label='EMA 3', color='cyan', linewidth=1, linestyle='--')
            ax1.plot(df['datetime'], df['EMA_1'], label='EMA 1', color='black', linewidth=1, linestyle='--')
            df = calcular_indicadores(df)
            for col in ['DONCHIAN_HIGH', 'DONCHIAN_LOW', 'DONCHIAN_RIBBON']:
                if col not in df.columns:
                    df[col] = np.nan
            ax1.plot(df['datetime'], df['DONCHIAN_HIGH'], color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Donchian High')
            ax1.plot(df['datetime'], df['DONCHIAN_LOW'], color='blue', linestyle='--', linewidth=1, alpha=0.7, label='Donchian Low')
            for i in range(-20, 0):
                val = df['DONCHIAN_RIBBON'].iloc[i] if not pd.isnull(df['DONCHIAN_RIBBON'].iloc[i]) else 0
                color = 'lime' if val == 1 else ('red' if val == -1 else 'gray')
                ax1.axvspan(df['datetime'].iloc[i-1], df['datetime'].iloc[i], color=color, alpha=0.08)
            ax1.legend(loc='upper left', fontsize=8)

            ax3.clear()
            ax3.plot(df['datetime'], df['TREND_METER'], color='purple', label='Trend Meter', linewidth=1.5)
            ax3.plot(df['datetime'], df['EFMUS'], color='magenta', label='EFMUS', linewidth=1)
            ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax3.set_ylabel('Trend/EFMUS')
            ax3.legend(loc='upper left', fontsize=8)
            ax3.set_title('Trend Meter y EFMUS')
            ax3.grid(True, linestyle='--', alpha=0.3)
            for i in range(-20, 0):
                if df['EFMUS_SIGNAL'].iloc[i] == 2:
                    ax3.axvline(df['datetime'].iloc[i], color='lime', linestyle=':', alpha=0.2)
                elif df['EFMUS_SIGNAL'].iloc[i] == -2:
                    ax3.axvline(df['datetime'].iloc[i], color='red', linestyle=':', alpha=0.2)
            ax1.text(df['datetime'].iloc[-1], last_price, f'Precio actual: {last_price:.2f}',
                     color='black', fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='yellow', alpha=0.5))
            ax1.text(df['datetime'].iloc[-1], last_price*1.03, f'Saldo: {saldo:.2f} USDT',
                     color='blue', fontsize=12, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

            # Mostrar información de posición si existe
            if posicion_abierta is not None:
                y_pos = posicion_abierta['precio_entrada']
                color_pos = 'green' if posicion_abierta['tipo'] == 'long' else 'red'
                ax1.axhline(y=y_pos, color=color_pos, linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.text(df['datetime'].iloc[-1], y_pos, f"{posicion_abierta['tipo'].upper()} @ {y_pos:.2f}",
                         color=color_pos, fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
                # Usar SL/TP almacenados en la posición (si existen), si no calcular por defecto
                sl = posicion_abierta.get('sl', y_pos * (0.99 if posicion_abierta['tipo'] == 'long' else 1.01))
                tp = posicion_abierta.get('tp', y_pos * (1.01 if posicion_abierta['tipo'] == 'long' else 0.99))
                margin = posicion_abierta.get('margin', None)
                apal = posicion_abierta.get('apalancamiento', None)
                ax1.axhline(y=sl, color='red', linestyle=':', linewidth=1.2, alpha=0.8, label='SL')
                ax1.axhline(y=tp, color='green', linestyle=':', linewidth=1.2, alpha=0.8, label='TP')
                ax1.text(df['datetime'].iloc[-1], sl, f'SL: {sl:.2f}', color='red', fontsize=9, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.3))
                ax1.text(df['datetime'].iloc[-1], tp, f'TP: {tp:.2f}', color='green', fontsize=9, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.3))
                # Mostrar margin y apalancamiento cerca del borde derecho de la gráfica
                extra_info_y = df['close'].iloc[-1] * 1.03
                margin_text = f"Margin: {margin:.2f} USDT" if margin is not None else "Margin: N/A"
                apal_text = f"Apalancamiento: x{apal}" if apal is not None else "Apalancamiento: N/A"
                ax1.text(df['datetime'].iloc[-1], extra_info_y, f"{margin_text} | {apal_text}", color='black', fontsize=9, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
                ax1.text(df['datetime'].iloc[-1], df['close'].iloc[-1], f"TrendMeter: {df['TREND_METER'].iloc[-1]}", color='purple', fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))

            if 'SQZ_ON' not in df.columns:
                print('Error: La columna SQZ_ON no está presente en el DataFrame. Recalculando indicadores...')
                df = calcular_indicadores(df)
                if 'SQZ_ON' not in df.columns:
                    raise KeyError('No se pudo calcular la columna SQZ_ON. Revisa el cálculo de indicadores.')

            if 'sl' in locals() and 'tp' in locals():
                print(f"SL: {sl:.2f} | TP: {tp:.2f}")

            ml_pred = None
            ml_text = ''
            if clf is not None:
                ult = df[features].iloc[[-1]].values
                ml_pred = clf.predict(ult)[0]
                ml_text = 'ML: COMPRAR' if ml_pred == 1 else 'ML: VENDER'
                ax1.text(df['datetime'].iloc[-1], last_price*1.01, ml_text,
                         color='green' if ml_pred == 1 else 'red', fontsize=12, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

            now = df['datetime'].iloc[-1]
            volatilidad = df['close'].rolling(window=10).std().iloc[-1]
            max_leverage = 20
            min_leverage = 10
            leverage = int(max(min_leverage, min(max_leverage, round(10 - min(volatilidad, 5)))))

            # Helper: calcular saldo efectivo incluyendo PnL no realizado para usar como margin disponible
            def saldo_con_pnl_actual():
                """Retorna el saldo efectivo que incluye PnL no realizado de la posición abierta (si existe)."""
                if posicion_abierta is None:
                    return saldo
                entrada = posicion_abierta.get('precio_entrada', 0)
                tipo_pos = posicion_abierta.get('tipo')
                apal = posicion_abierta.get('apalancamiento', 1)
                # Calcular PnL no realizado en USDT según tipo
                if tipo_pos == 'long':
                    pnl = (last_price - entrada) / entrada * apal * saldo / 100
                else:
                    pnl = (entrada - last_price) / entrada * apal * saldo / 100
                return max(0.0, saldo + pnl)

            ema_cross = (df['EMA_5'].iloc[-2] < df['EMA_660'].iloc[-2] and df['EMA_5'].iloc[-1] > df['EMA_660'].iloc[-1]) or \
                        (df['EMA_5'].iloc[-2] > df['EMA_660'].iloc[-2] and df['EMA_5'].iloc[-1] < df['EMA_660'].iloc[-1])
            macd_bull = df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]
            macd_bear = df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1]
            rsi_bull = df['RSI_14_EMA5'].iloc[-1] < 30
            rsi_bear = df['RSI_14_EMA5'].iloc[-1] > 70

            onda_tipo, onda_idxs = detectar_onda_elliott(df)
            elliott_long = onda_tipo == 'alcista'
            elliott_short = onda_tipo == 'bajista'

            abrir_long = ema_cross and macd_bull and rsi_bull and ml_pred == 1 and elliott_long
            abrir_short = ema_cross and macd_bear and rsi_bear and ml_pred == 0 and elliott_short

            if posicion_abierta is None:
                tipo = None
                patron_detectado = None
                elliott_tipo, elliott_idxs = detectar_onda_elliott(df, lookback=30)
                fibo_tipo, fibo_levels, fibo_idxs = detectar_fibonacci_experto(df, lookback=50)
                if elliott_tipo is not None:
                    tipo = 'long' if elliott_tipo == 'alcista' else 'short'
                    patron_detectado = f"elliott_{elliott_tipo}"
                elif fibo_tipo is not None:
                    tipo = fibo_tipo
                    patron_detectado = f"fibonacci_{fibo_tipo}"
                if tipo is not None:
                    # Calcular SL, TP y margin
                    if tipo == 'long':
                        sl = last_price * 0.99
                        tp = last_price * 1.01
                    else:
                        sl = last_price * 1.01
                        tp = last_price * 0.99
                    margin = saldo_con_pnl_actual() if leverage > 0 else 0
                    posicion_abierta = {
                        'tipo': tipo,
                        'precio_entrada': last_price,
                        'fecha_entrada': str(now),
                        'apalancamiento': leverage,
                        'indicadores': {k: float(df[k].iloc[-1]) for k in features},
                        'patron_detectado': patron_detectado,
                        'prediccion': tipo,
                        'sl': sl,
                        'tp': tp,
                        'margin': margin
                    }
                if onda_idxs is not None:
                    elliott_color = 'green' if onda_tipo == 'alcista' else 'red'
                    elliott_label = f"Onda Elliott: {onda_tipo.upper()}"
                    ax1.plot(df['datetime'].iloc[onda_idxs], df['close'].iloc[onda_idxs], marker='o', color=elliott_color, linestyle='-', linewidth=2, label=elliott_label)
                    ax1.text(df['datetime'].iloc[onda_idxs[-1]], df['close'].iloc[onda_idxs[-1]]*1.01, elliott_label, color=elliott_color, fontsize=9, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.2))
                if abrir_long:
                    tipo = 'long'
                elif abrir_short:
                    tipo = 'short'
                if tipo is not None:
                    # Calcular SL, TP y margin
                    if tipo == 'long':
                        sl = last_price * 0.99
                        tp = last_price * 1.01
                    else:
                        sl = last_price * 1.01
                        tp = last_price * 0.99
                    margin = saldo_con_pnl_actual() if leverage > 0 else 0
                    posicion_abierta = {
                        'tipo': tipo,
                        'precio_entrada': last_price,
                        'fecha_entrada': str(now),
                        'apalancamiento': leverage,
                        'indicadores': {k: float(df[k].iloc[-1]) for k in features},
                        'patron_detectado': patron_detectado,
                        'prediccion': tipo,
                        'sl': sl,
                        'tp': tp,
                        'margin': margin
                    }
                    # Notificación Telegram de entrada detallada
                    try:
                        mensaje = (
                            f"🚦 ENTRADA: {patron_detectado}\n"
                            f"Tipo: {tipo.upper()}\n"
                            f"Precio: {last_price:.2f} USDT\n"
                            f"Fecha: {now}\n"
                            f"SL: {sl:.2f} USDT\n"
                            f"TP: {tp:.2f} USDT\n"
                            f"Margin: {margin:.2f} USDT\n"
                            f"Apalancamiento: x{leverage}\n"
                            f"Saldo: {saldo:.2f} USDT"
                        )
                        enviar_telegram_mensaje(mensaje)
                    except Exception as e:
                        print(f"[TELEGRAM] Error al notificar entrada: {e}")
                    save_plot_snapshot(fig, filename_prefix="backtest_entrada")
                    abrir_operacion_snapshot = True
                    cerrar_operacion_snapshot = False
            else:
                mantener = decision_cierre_ia(df, posicion_abierta, ml_pred)
                cerrar_ia = not mantener
                if posicion_abierta['tipo'] == 'long':
                    sl = posicion_abierta['precio_entrada'] * 0.99
                    tp = posicion_abierta['precio_entrada'] * 1.01
                    sl_hit = last_price <= sl
                    tp_hit = last_price >= tp
                else:
                    sl = posicion_abierta['precio_entrada'] * 1.01
                    tp = posicion_abierta['precio_entrada'] * 0.99
                    sl_hit = last_price >= sl
                    tp_hit = last_price <= tp
                motivo_cierre = ''
                if cerrar_ia:
                    motivo_cierre = 'ML+INDICADORES'
                elif sl_hit:
                    motivo_cierre = 'SL'
                elif tp_hit:
                    motivo_cierre = 'TP'
                if cerrar_ia or sl_hit or tp_hit:
                    saldo_entrada = saldo
                    resultado = (last_price - posicion_abierta['precio_entrada']) if posicion_abierta['tipo'] == 'long' else (posicion_abierta['precio_entrada'] - last_price)
                    resultado_pct = resultado / posicion_abierta['precio_entrada'] * posicion_abierta['apalancamiento'] * 100
                    ganancia = saldo * (resultado_pct / 100)
                    comision_pct = 0.0004 * 2
                    comision = saldo_entrada * comision_pct
                    saldo += ganancia - comision
                    operacion = {
                        'tipo': posicion_abierta['tipo'],
                        'precio_entrada': posicion_abierta['precio_entrada'],
                        'fecha_entrada': posicion_abierta['fecha_entrada'],
                        'precio_salida': last_price,
                        'fecha_salida': str(now),
                        'apalancamiento': posicion_abierta['apalancamiento'],
                        'resultado_pct': resultado_pct,
                        'ganancia_usdt': ganancia,
                        'comision_usdt': comision,
                        'saldo_entrada': saldo_entrada,
                        'saldo_final': saldo,
                        'motivo_cierre': motivo_cierre,
                        'indicadores_entrada': posicion_abierta['indicadores'],
                        'indicadores_salida': {k: float(df[k].iloc[-1]) for k in features}
                    }
                    log_operaciones.append(operacion)
                    with open(log_path, 'w') as f:
                        json.dump(log_operaciones, f, indent=2)
                    posicion_abierta = None
                    # Notificación Telegram de salida
                    try:
                        resultado_pct = operacion.get('resultado_pct', 0)
                        despedida = "me partieron 😭" if resultado_pct < 0 else "vamos pa dubai bro 😎 jajajaj"
                        mensaje = (
                            f"🏁 SALIDA: {operacion.get('tipo','')}\n"
                            f"Entrada: {operacion.get('precio_entrada',0):.2f} USDT\n"
                            f"Salida: {operacion.get('precio_salida',0):.2f} USDT\n"
                            f"Resultado: {resultado_pct:.2f}%\n"
                            f"Saldo entrada: {operacion.get('saldo_entrada',0):.2f} USDT\n"
                            f"Saldo final: {operacion.get('saldo_final',0):.2f} USDT\n"
                            f"Comisión: {operacion.get('comision_usdt',0):.4f} USDT\n"
                            f"Motivo: {operacion.get('motivo_cierre','')}\n"
                            f"Patrón: {operacion.get('patron_detectado','')}\n"
                            f"Fecha entrada: {operacion.get('fecha_entrada','')}\n"
                            f"Fecha salida: {operacion.get('fecha_salida','') if 'fecha_salida' in operacion else ''}\n\n"
                            f"{despedida}"
                        )
                        enviar_telegram_mensaje(mensaje)
                    except Exception as e:
                        print(f"[TELEGRAM] Error al notificar salida: {e}")
                    save_plot_snapshot(fig, filename_prefix="backtest_salida")
                    cerrar_operacion_snapshot = True
                    abrir_operacion_snapshot = False

            tipo_elliott, pivotes_elliott = detectar_onda_elliott(df, lookback=30)
            if tipo_elliott is not None:
                patron_experto = f"Elliott-{tipo_elliott}"
                print(f"ESTRATEGIA: Señal experta Elliott detectada: {tipo_elliott.upper()}")
                color_elliott = 'green' if tipo_elliott == 'alcista' else 'red'
                for idx in pivotes_elliott:
                    if idx in df.index:
                        ax1.scatter(df['datetime'].iloc[idx], df['close'].iloc[idx], color=color_elliott, s=80, marker='o', label='Elliott Pivot')
                entrada_idx = len(df) - 1
            else:
                fibo_tipo, fibo_levels, fibo_idxs = detectar_fibonacci_experto(df, lookback=50)
                if fibo_tipo is not None:
                    patron_experto = f"Fibonacci-{fibo_tipo}"
                    print(f"ESTRATEGIA: Señal experta Fibonacci detectada: {fibo_tipo.upper()} en nivel clave")
                    color_fibo = 'blue' if fibo_tipo == 'long' else 'red'
                    min_price = df['low'].iloc[-60:].min()
                    max_price = df['high'].iloc[-60:].max()
                    for lvl in fibo_levels:
                        if min_price <= lvl <= max_price:
                            ax1.axhline(y=lvl, color=color_fibo, linestyle=':', linewidth=1.2, alpha=0.5)
                    if fibo_idxs:
                        for idx in fibo_idxs:
                            if idx in df.index:
                                ax1.scatter(df['datetime'].iloc[idx], df['close'].iloc[idx], color=color_fibo, s=80, marker='o', label='Fibo Pivot')
                    entrada_idx = len(df) - 1

            if best_score > 0:
                acc_text = f"ML accuracy (val): {best_score*100:.2f}%"
                print(acc_text)
                ax1.text(df['datetime'].iloc[-1], last_price*1.07, acc_text, color='blue', fontsize=9, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.3))

            if ml_pred is not None:
                if (abrir_long or abrir_short):
                    ax1.scatter(df['datetime'].iloc[-1], last_price, color='lime', s=120, marker='*', label='ML+Estrategia')

            plt.subplots_adjust(top=0.88, bottom=0.13)

            if posicion_abierta is None:
                estado = "Esperando señal de entrada..."
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {estado}")
                ax1.text(df['datetime'].iloc[-1], last_price*1.05, estado, color='gray', fontsize=11, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))
                if not notificado_espera:
                    try:
                        enviar_telegram_mensaje("⏳ El bot está esperando una señal de entrada...")
                        notificado_espera = True
                    except Exception as e:
                        print(f"[TELEGRAM] Error al notificar espera de entrada: {e}")
            else:
                tipo = 'COMPRA (LONG)' if posicion_abierta['tipo'] == 'long' else 'VENTA (SHORT)'
                indicadores = posicion_abierta['indicadores']
                patron = f"EMA660: {indicadores['EMA_660']:.2f} | EMA5: {indicadores['EMA_5']:.2f} | EMA4: {indicadores['EMA_4']:.2f} | RSI14_EMA5: {indicadores['RSI_14_EMA5']:.2f}"
                if posicion_abierta['tipo'] == 'long':
                    pnl = (last_price - posicion_abierta['precio_entrada']) / posicion_abierta['precio_entrada'] * posicion_abierta['apalancamiento'] * 100
                else:
                    notificado_espera = False  # Se resetea al abrir una posición
                    pnl = (posicion_abierta['precio_entrada'] - last_price) / posicion_abierta['precio_entrada'] * posicion_abierta['apalancamiento'] * 100
                saldo_pos = f"Saldo actual: {saldo:.2f} USDT | PnL: {pnl:.2f}%"
                estado = f"En operación: {tipo} | Entrada: {posicion_abierta['precio_entrada']:.2f} USDT | Apalancamiento: x{posicion_abierta['apalancamiento']}\nPatrón: {patron}\n{saldo_pos}"
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {estado}")
                ax1.text(df['datetime'].iloc[-1], last_price*1.05, estado, color='black', fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='yellow', alpha=0.3, edgecolor='none'))

            min_y = df['low'].iloc[-60:].min()
            max_y = df['high'].iloc[-60:].max()
            ax1.set_ylim(min_y * 0.995, max_y * 1.005)
            ax1.legend()
            ax2.clear()
            ax2.plot(df['datetime'], df['RSI_14_EMA5'], label='RSI 14 (EMA 5)', color='purple')
            ax2.axhline(70, color='red', linestyle='--', linewidth=0.8)
            ax2.axhline(30, color='green', linestyle='--', linewidth=0.8)
            ax2.set_ylabel('RSI')
            ax2.set_xlabel('Fecha')
            ax2.legend()
            ax2.grid()

            if SHOW_PLOTS:
                plt.pause(0.01)
            time.sleep(1)
        except KeyboardInterrupt:
            print("Interrumpido por el usuario.")
            break
        except Exception as e:
            print(f"[ERROR] Excepción en ciclo principal: {e}")
            traceback.print_exc()
            # esperar un poco antes de reintentar el ciclo principal
            time.sleep(5)
            continue



# Mostrar log de operaciones al finalizar y entrenar con Gemini
def mostrar_resumen_y_entrenar():
    if log_operaciones:
        print("\nResumen de operaciones:")
        for op in log_operaciones:
            print(f"{op['fecha_entrada']} {op['tipo'].upper()} entrada: {op['precio_entrada']:.2f} salida: {op['precio_salida']:.2f} resultado: {op['resultado_pct']:.2f}% saldo entrada: {op['saldo_entrada']:.2f} saldo final: {op['saldo_final']:.2f} USDT comisión: {op['comision_usdt']:.4f} motivo: {op.get('motivo_cierre','')} patrón: {op.get('patron_detectado','')}")
        # Entrenamiento con Gemini AI
        entrenar_con_gemini(log_path)
    if SHOW_PLOTS:
        plt.show()

# --- Entrenamiento y predicción automática cada 30 minutos ---
def ciclo_entrenamiento_30min():
    while True:
        print("[CICLO] Descargando histórico, entrenando IA y aplicando estrategia...")
        descargar_y_entrenar_historico(symbol='ETH/USDT', timeframe='5m', total_limit=2000, chunk=500, log_path='log_operaciones_historico.json')
        print("[CICLO] Esperando 30 minutos para el próximo entrenamiento...")
        time.sleep(1800)

threading.Thread(target=ciclo_entrenamiento_30min, daemon=True).start()

# Enviar mensaje de bienvenida a Telegram al iniciar el bot
if __name__ == "__main__":
    print("Iniciando bot de trading...")
    try:
        enviar_telegram_mensaje("🚀🤖 ¡El bot de trading se ha iniciado correctamente! Listo para operar y aprender. ¡Vamos a por esos profits! 💰📈")
    except Exception as e:
        print(f"[TELEGRAM] Error al enviar mensaje de bienvenida: {e}")
    print("Bot de trading inicializado. Ejecutando ciclo principal...")
    main()
    mostrar_resumen_y_entrenar()
#!/usr/bin/env python3
"""
Versión renombrada de historico_btcusdt.py — se ejecuta como `tradingbot.py`.
"""
