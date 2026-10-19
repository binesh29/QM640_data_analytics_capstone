"""Shared chart style for the QM640 capstone figures (print, light background)."""
import matplotlib as mpl

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def apply():
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'axes.titlesize': 11,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.axisbelow': True, 'xtick.color': INK2, 'ytick.color': INK2,
        'legend.frameon': False, 'figure.dpi': 110, 'savefig.dpi': 200,
        'savefig.bbox': 'tight', 'lines.linewidth': 2,
    })


def wilson(k, n, z=1.96):
    """Wilson 95% confidence interval for a proportion k/n."""
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / d
    return c - h, c + h
