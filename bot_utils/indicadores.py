import pandas as pd
import numpy as np
try:
    from ta.trend import PSARIndicator
except ImportError:
    PSARIndicator = None

def calcular_indicadores(df):
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
    if PSARIndicator is not None:
        psar = PSARIndicator(df['high'], df['low'], df['close'], step=0.10, max_step=0.10)
        df['PSAR'] = psar.psar()
    else:
        df['PSAR'] = np.nan
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
    df['SQZ_ON'] = (df['BB_lower'] > df['KC_lower']) & (df['BB_upper'] < df['KC_upper'])
    df['SQZ_OFF'] = (df['BB_lower'] < df['KC_lower']) & (df['BB_upper'] > df['KC_upper'])
    df['SQZMOM_LB'] = df['close'] - df['close'].rolling(window=20).mean()
    trend_score = (
        (df['EMA_5'] > df['EMA_55']).astype(int) - (df['EMA_5'] < df['EMA_55']).astype(int) +
        (df['MACD'] > df['MACD_signal']).astype(int) - (df['MACD'] < df['MACD_signal']).astype(int) +
        (df['RSI_14_EMA5'] > 60).astype(int) - (df['RSI_14_EMA5'] < 40).astype(int) +
        (df['PSAR'] < df['close']).astype(int) - (df['PSAR'] > df['close']).astype(int)
    )
    df['TREND_METER'] = trend_score
    return df
