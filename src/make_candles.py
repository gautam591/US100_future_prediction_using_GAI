import numpy as np
from matplotlib.patches import Rectangle

def plot_candlesticks(ax, o, h, l, c, width=0.6):
    """Minimal candlestick renderer using Matplotlib patches."""
    x = np.arange(len(c))
    for i in range(len(c)):
        ax.vlines(x[i], l[i], h[i])
        body_low = float(min(o[i], c[i]))
        body_high = float(max(o[i], c[i]))
        height = max(body_high - body_low, 1e-9)
        rect = Rectangle((x[i] - width / 2, body_low), width, height)
        ax.add_patch(rect)
    ax.set_xlim(-1, len(c))
    return x
