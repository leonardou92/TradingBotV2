import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_candles(ax, df, entrada_idx=None):
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
