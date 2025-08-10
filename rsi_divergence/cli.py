"""Command-line interface for the RSI divergence detector."""

import typer
import pandas as pd
from .core import find_divergences

app = typer.Typer()

@app.command()
def main(
    file: str = typer.Option(..., "--file", "-f", help="Path to the OHLC data file (CSV)."),
    rsi_period: int = typer.Option(14, help="RSI lookback period."),
    price_prominence: int = typer.Option(1, help="Price peak prominence."),
    rsi_prominence: int = typer.Option(1, help="RSI peak prominence."),
    price_width: int = typer.Option(1, help="Price peak width."),
    rsi_width: int = typer.Option(1, help="RSI peak width."),
):
    """Detect RSI divergences from an OHLC data file."""
    try:
        ohlc = pd.read_csv(file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: File not found at {file}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"Error reading file: {e}")
        raise typer.Exit(code=1)

    divergences = find_divergences(
        ohlc,
        rsi_period=rsi_period,
        price_prominence=price_prominence,
        rsi_prominence=rsi_prominence,
        price_width=price_width,
        rsi_width=rsi_width,
    )

    if divergences.empty:
        print("No divergences found.")
    else:
        print(divergences)

if __name__ == "__main__":
    app()
