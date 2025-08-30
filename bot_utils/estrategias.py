from .patrones import detectar_onda_elliott, detectar_fibonacci_experto

def decision_cierre_ia(df, posicion_abierta, ml_pred):
    if posicion_abierta['tipo'] == 'long':
        mantener = not (ml_pred == 0 and (detectar_onda_elliott(df, 30)[0] == 'bajista' or detectar_fibonacci_experto(df, 50)[0] == 'short'))
    elif posicion_abierta['tipo'] == 'short':
        mantener = not (ml_pred == 1 and (detectar_onda_elliott(df, 30)[0] == 'alcista' or detectar_fibonacci_experto(df, 50)[0] == 'long'))
    else:
        mantener = False
    return mantener
