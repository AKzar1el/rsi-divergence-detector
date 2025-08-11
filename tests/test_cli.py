import pandas as pd
from typer.testing import CliRunner

from rsi_divergence.cli import app

def test_cli_smoke(tmp_path):
    # Minimal CSV with a 'close' column
    idx = pd.date_range("2024-01-01", periods=60, freq="T")
    s = pd.Series(100 + (idx.view("i8") % 11) * 0.1, index=idx, name="close")
    df = pd.DataFrame({"close": s}, index=idx)
    csv = tmp_path / "ohlc.csv"
    df.to_csv(csv)

    runner = CliRunner()
    result = runner.invoke(app, ["--file", str(csv), "--rsi-period", "9"])
    assert result.exit_code == 0
    # Output is either "No divergences found." or a table; both are fine for smoke.
    assert result.stdout.strip() != ""
