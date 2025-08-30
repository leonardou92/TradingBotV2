# TradingViewBot

Bot de trading algorítmico para criptomonedas con señales avanzadas, Machine Learning y notificaciones por Telegram.

## Características principales
- Descarga y análisis de históricos de precios desde Binance usando ccxt.
- Cálculo de indicadores técnicos avanzados (EMAs, MACD, RSI, PSAR, Donchian, Squeeze Momentum, etc).
- Detección de patrones expertos: Onda de Elliott y niveles de Fibonacci.
- Entrenamiento y predicción con modelo Random Forest (scikit-learn).
- Entrenamiento y sugerencias con Gemini AI (Google Generative Language API).
- Visualización en tiempo real de velas, indicadores y señales.
- Simulación y registro de operaciones (log JSON).
- Notificaciones automáticas por Telegram para:
  - Inicio del bot
  - Espera de señal de entrada (solo una vez por ciclo)
  - Entrada y salida de operaciones
  - Descarga y entrenamiento de históricos
- Modularización del código en utilidades y estrategias.
- Configuración segura de claves y tokens vía `.env`.

## Estructura del proyecto

```
TradingViewBot/
├── historico_btcusdt.py           # Script principal del bot
├── test_telegram.py               # Test de notificación Telegram
├── .env                           # Variables sensibles (no subir)
├── bot_utils/
│   ├── indicadores.py             # Indicadores técnicos
│   ├── patrones.py                # Detección de patrones
│   ├── ml_utils.py                # Utilidades ML
│   ├── plot_utils.py              # Gráficas
│   ├── telegram_utils.py          # Notificaciones Telegram
│   ├── utils.py                   # Utilidades generales
│   └── estrategias.py             # Estrategias de cierre/gestión
└── ...
```

## Instalación

1. Clona el repositorio y entra al directorio:
   ```bash
   git clone <repo_url>
   cd TradingViewBot
   ```
2. Crea y activa un entorno virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Crea un archivo `.env` con tus claves:
   ```env
   TELEGRAM_BOT_TOKEN=xxxxxx
   TELEGRAM_CHAT_ID=xxxxxx
   GEMINI_API_KEY=xxxxxx
   ```

## Uso

Ejecuta el bot principal:
```bash
.venv/bin/python historico_btcusdt.py
```

- El bot enviará notificaciones a tu Telegram y mostrará gráficos en tiempo real.
- El ciclo de entrenamiento y descarga se ejecuta automáticamente cada 30 minutos.
- El bot opera de forma simulada, pero puedes adaptar la lógica para operar real si lo deseas.

## Personalización
- Modifica los parámetros de indicadores, estrategias o el modelo ML en los archivos de `bot_utils/`.
- Puedes agregar más exchanges, marcos temporales o estrategias fácilmente.

## Seguridad
- **Nunca subas tu archivo `.env` ni compartas tus claves.**
- El bot está pensado para backtesting y simulación. Si deseas operar en real, revisa y adapta la gestión de riesgos.

## Créditos
- Basado en Python, ccxt, pandas, numpy, matplotlib, scikit-learn, ta, requests, python-telegram-bot, python-dotenv.
- Desarrollado por leonardou92 y colaboradores.

---

¡Disfruta y mejora tu trading algorítmico!
