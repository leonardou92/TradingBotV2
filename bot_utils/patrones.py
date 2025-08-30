import pandas as pd

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
    def distancia_minima(pivotes, min_dist=2):
        return all(abs(pivotes[i][0] - pivotes[i-1][0]) >= min_dist for i in range(1, len(pivotes)))
    if len(pivotes) >= 5 and [p[1] for p in pivotes[-5:]] == ['min','max','min','max','min'] and distancia_minima(pivotes[-5:]):
        base_idx = df.index[-lookback:][0]
        return 'alcista', [base_idx + p[0] for p in pivotes[-5:]]
    if len(pivotes) >= 5 and [p[1] for p in pivotes[-5:]] == ['max','min','max','min','max'] and distancia_minima(pivotes[-5:]):
        base_idx = df.index[-lookback:][0]
        return 'bajista', [base_idx + p[0] for p in pivotes[-5:]]
    return None, None

def detectar_fibonacci_experto(df, lookback=50, min_retracement=0.382, max_retracement=0.618):
    closes = df['close'].iloc[-lookback:]
    idx_high = closes.idxmax()
    idx_low = closes.idxmin()
    price_high = closes.loc[idx_high]
    price_low = closes.loc[idx_low]
    if idx_low < idx_high:
        move = price_low - price_high
        levels = [price_high + move * r for r in [0, min_retracement, 0.5, max_retracement, 1, 1.618]]
        current = closes.iloc[-1]
        for lvl, r in zip(levels, [0, min_retracement, 0.5, max_retracement, 1, 1.618]):
            if abs(current - lvl) / abs(move) < 0.01 and r in [min_retracement, max_retracement]:
                return 'short', levels, [idx_high, idx_low]
    elif idx_high < idx_low:
        move = price_high - price_low
        levels = [price_low + move * r for r in [0, min_retracement, 0.5, max_retracement, 1, 1.618]]
        current = closes.iloc[-1]
        for lvl, r in zip(levels, [0, min_retracement, 0.5, max_retracement, 1, 1.618]):
            if abs(current - lvl) / abs(move) < 0.01 and r in [min_retracement, max_retracement]:
                return 'long', levels, [idx_low, idx_high]
    return None, None, None
