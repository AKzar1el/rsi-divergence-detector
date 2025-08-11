from __future__ import annotations

from typing import Optional
import pandas as pd
import typer

from .core import calculate_rsi, find_divergences

app = typer.Typer(help="Detect RSI divergences from an OHLC CSV with a 'close' column.")

@app.command()
def main(
    file: str = typer.Option(..., "--file", "-f", help="Path to OHLC CSV with a 'close' column."),
    rsi_period: int = typer.Option(14, help="RSI lookback period."),
    # Use Optional[...] for Typer compatibility
    price_prominence: Optional[float] = typer.Option(None, help="Price peak prominence (auto if omitted)."),
    rsi_prominence: Optional[float] = typer.Option(None, help="RSI peak prominence (auto if omitted)."),
    price_width: Optional[int] = typer.Option(None, help="Price peak min width in samples (auto if omitted)."),
    rsi_width: Optional[int] = typer.Option(None, help="RSI peak min width in samples (auto if omitted)."),
    distance: Optional[int] = typer.Option(None, help="Min distance between peaks (samples, auto if omitted)."),
    max_lag: int = typer.Option(3, help="Max bars between paired price/RSI pivots."),
    include_hidden: bool = typer.Option(True, help="Detect hidden divergences."),
):
    try:
        df = pd.read_csv(file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        typer.echo(f"Error: File not found at {file}")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error reading file: {e}")
        raise typer.Exit(code=1)

    if "close" in df.columns:
        price = df["close"]
    else:
        num = df.select_dtypes("number")
        if num.empty:
            typer.echo("CSV must contain a numeric 'close' column.")
            raise typer.Exit(code=2)
        price = num.iloc[:, 0].rename("close")

    if not price.index.is_monotonic_increasing:
        price = price.sort_index()

    rsi = calculate_rsi(price, period=rsi_period)
    divs = find_divergences(
        price, rsi,
        rsi_period=rsi_period,
        price_prominence=price_prominence,
        rsi_prominence=rsi_prominence,
        price_width=price_width,
        rsi_width=rsi_width,
        distance=distance,
        max_lag=max_lag,
        include_hidden=include_hidden,
    )

    typer.echo("No divergences found." if divs.empty else divs.to_string(index=False))

if __name__ == "__main__":
    app()
