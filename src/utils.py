# src/utils.py
import os


def ensure_directory(path):
    """
    Ensure that a directory exists.

    Parameters:
    - path: str
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")


def save_plot(fig, path):
    """
    Save a matplotlib figure to the specified path.

    Parameters:
    - fig: matplotlib.figure.Figure
    - path: str
    """
    fig.savefig(path)
    print(f"Plot saved to {path}.")
