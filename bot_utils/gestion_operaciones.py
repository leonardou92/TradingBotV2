from datetime import datetime
from bot_utils.telegram_utils import enviar_telegram_mensaje
from bot_utils.utils import save_plot_snapshot
import json

def manejar_entrada(df, last_price, ml_pred, leverage, saldo, features):
    # Lógica de entrada multi-indicador y gestión de riesgo
    # ...extraer factores y señales...
    # Devuelve dict posicion_abierta o None
    # Ejemplo simplificado:
    atr = df['ATR_14'].iloc[-1]
    tipo = None
    patron_detectado = None
    factores = {}
    # Aquí deberías extraer todos los factores como en el main
    # ...
    # Si hay señal:
    if ml_pred == 1:
        tipo = 'long'
        patron_detectado = 'long_multi'
    elif ml_pred == 0:
        tipo = 'short'
        patron_detectado = 'short_multi'
    if tipo:
        sl = last_price - atr * 2 if tipo == 'long' else last_price + atr * 2
        tp = last_price + atr * 3 if tipo == 'long' else last_price - atr * 3
        margin = saldo
        posicion_abierta = {
            'tipo': tipo,
            'precio_entrada': last_price,
            'fecha_entrada': str(df['datetime'].iloc[-1]),
            'apalancamiento': leverage,
            'indicadores': {k: float(df[k].iloc[-1]) for k in features},
            'patron_detectado': patron_detectado,
            'prediccion': tipo,
            'sl': sl,
            'tp': tp,
            'margin': margin,
            'factores': factores
        }
        # Notificación Telegram
        mensaje = (
            f"🚦 ENTRADA: {patron_detectado}\n"
            f"Tipo: {tipo.upper()}\n"
            f"Precio: {last_price:.2f} USDT\n"
            f"Fecha: {df['datetime'].iloc[-1]}\n"
            f"SL: {sl:.2f} USDT\n"
            f"TP: {tp:.2f} USDT\n"
            f"Margin: {margin:.2f} USDT\n"
            f"Apalancamiento: x{leverage}\n"
            f"ATR: {atr:.2f}\n"
            f"Saldo: {saldo:.2f} USDT\n"
            f"Factores: {factores}"
        )
        enviar_telegram_mensaje(mensaje)
        # Solo guardar snapshot si fig está disponible
        try:
            from matplotlib import pyplot as plt
            fig = plt.gcf()
            if fig:
                save_plot_snapshot(fig, filename_prefix="backtest_entrada")
        except Exception:
            pass
        return posicion_abierta
    return None

def manejar_salida(df, posicion_abierta, last_price, saldo, features, log_operaciones, log_path):
    # Lógica de cierre y registro
    apal = posicion_abierta.get('apalancamiento', 1)
    margin_amount = posicion_abierta.get('margin', saldo)
    sl = posicion_abierta['sl']
    tp = posicion_abierta['tp']
    if posicion_abierta['tipo'] == 'long':
        sl_hit = last_price <= sl
        tp_hit = last_price >= tp
    else:
        sl_hit = last_price >= sl
        tp_hit = last_price <= tp
    motivo_cierre = ''
    if sl_hit:
        motivo_cierre = 'SL'
    elif tp_hit:
        motivo_cierre = 'TP'
    if sl_hit or tp_hit:
        saldo_entrada = saldo
        entrada_price = posicion_abierta['precio_entrada']
        pnl_pct_on_margin = (last_price - entrada_price) / entrada_price * apal * 100 if posicion_abierta['tipo']=='long' else (entrada_price - last_price) / entrada_price * apal * 100
        pnl_usdt = margin_amount * (pnl_pct_on_margin / 100)
        notional = margin_amount * apal
        comision_pct = 0.0004 * 2
        comision = notional * comision_pct
        saldo = saldo + pnl_usdt - comision
        duracion = (datetime.strptime(str(df['datetime'].iloc[-1]), "%Y-%m-%d %H:%M:%S") - datetime.strptime(posicion_abierta['fecha_entrada'], "%Y-%m-%d %H:%M:%S")).total_seconds() / 60.0
        operacion = {
            'tipo': posicion_abierta['tipo'],
            'precio_entrada': posicion_abierta['precio_entrada'],
            'fecha_entrada': posicion_abierta['fecha_entrada'],
            'precio_salida': last_price,
            'fecha_salida': str(df['datetime'].iloc[-1]),
            'apalancamiento': apal,
            'resultado_pct': pnl_pct_on_margin,
            'ganancia_usdt': pnl_usdt,
            'patron_detectado': posicion_abierta.get('patron_detectado',''),
            'comision_usdt': comision,
            'saldo_entrada': saldo_entrada,
            'saldo_final': saldo,
            'motivo_cierre': motivo_cierre,
            'indicadores_entrada': posicion_abierta['indicadores'],
            'indicadores_salida': {k: float(df[k].iloc[-1]) for k in features},
            'factores': posicion_abierta.get('factores', {}),
            'sl': sl,
            'tp': tp,
            'duracion_min': duracion
        }
        log_operaciones.append(operacion)
        with open(log_path, 'w') as f:
            json.dump(log_operaciones, f, indent=2)
        # Notificación Telegram de salida
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
            f"SL: {operacion.get('sl',0):.2f} USDT\n"
            f"TP: {operacion.get('tp',0):.2f} USDT\n"
            f"Duración: {operacion.get('duracion_min',0):.2f} min\n"
            f"Factores: {operacion.get('factores',{})}\n"
            f"Fecha entrada: {operacion.get('fecha_entrada','')}\n"
            f"Fecha salida: {operacion.get('fecha_salida','') if 'fecha_salida' in operacion else ''}\n\n"
            f"{despedida}"
        )
        enviar_telegram_mensaje(mensaje)
        # Solo guardar snapshot si fig está disponible
        try:
            from matplotlib import pyplot as plt
            fig = plt.gcf()
            if fig:
                save_plot_snapshot(fig, filename_prefix="backtest_salida")
        except Exception:
            pass
        return saldo, None, log_operaciones
    return saldo, posicion_abierta, log_operaciones
