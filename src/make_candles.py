import numpy as np
from matplotlib.patches import Rectangle


def plot_candlesticks(ax, o, h, l, c, width=0.9, min_body=0.5, x_offset=0):
    """
    Filled candlestick renderer.
    x_offset lets you place candles later on the x-axis (for future candles).
    """
    x = np.arange(len(c)) + int(x_offset)

    for i in range(len(c)):
        open_i = float(o[i])
        high_i = float(h[i])
        low_i = float(l[i])
        close_i = float(c[i])

        # wick
        ax.vlines(x[i], low_i, high_i, linewidth=1)

        # body
        body_low = min(open_i, close_i)
        body_high = max(open_i, close_i)
        height = max(body_high - body_low, float(min_body))

        color = "green" if close_i >= open_i else "red"

        rect = Rectangle(
            (x[i] - width / 2, body_low),
            width,
            height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.6,
            alpha=0.9,
        )
        ax.add_patch(rect)

    return x
