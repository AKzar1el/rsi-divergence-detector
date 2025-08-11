# rsi-divergence-detector

**Lightweight, tested Python library & CLI for detecting RSI divergences (regular & hidden) on time‑series price data.**

It uses **Wilder‑style RSI**, **robust peak finding** via SciPy, and **deterministic, monotonic pivot pairing** with adaptive defaults so you can get useful results with minimal tuning. Clean function contracts, no hidden globals, and a tidy tabular output.

> **Python**: 3.9–3.13 · **Dependencies**: NumPy (>=1.26), pandas (>=2.2.2), SciPy (>=1.13), Typer (for the CLI)

---

## Table of Contents

* [Features](#features)
* [Install](#install)
* [Quickstart (Python API)](#quickstart-python-api)
* [Quickstart (CLI)](#quickstart-cli)
* [Returned Schema](#returned-schema)
* [Concepts & Algorithm](#concepts--algorithm)

  * [Divergences we detect](#divergences-we-detect)
  * [RSI (Wilder‑style)](#rsi-wilderstyle)
  * [Peak detection (SciPy)](#peak-detection-scipy)
  * [Monotonic pivot pairing](#monotonic-pivot-pairing)
* [API Reference](#api-reference)

  * [`calculate_rsi`](#calculate_rsi)
  * [`find_divergences`](#find_divergences)
* [Tuning Tips](#tuning-tips)
* [Examples](#examples)
* [Testing](#testing)
* [Versioning / Compatibility](#versioning--compatibility)
* [Limitations & Notes](#limitations--notes)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [References](#references)
* [SEO Keywords](#seo-keywords)

---

## Features

* ✅ **Regular & hidden RSI divergences**: `regular_bullish`, `regular_bearish`, `hidden_bullish`, `hidden_bearish`.
* ✅ **Deterministic pairing**: nearest‑in‑time RSI pivots for each consecutive price pivot pair; **monotonic in time** (no re‑use/backtracking).
* ✅ **Adaptive defaults** for peak gates (prominence/width/distance) to stay scale‑aware without hand‑tuning.
* ✅ **Clear contracts**: explicit, stateless functions returning a tidy DataFrame.
* ✅ **CLI + Python API** (Typer‑powered CLI with `--help`).
* ✅ **Tests**: unit tests (RSI consistency & divergence scenarios) + property tests (Hypothesis) + CLI smoke tests.

---

## Install

From PyPI:

```bash
pip install rsi-divergence-detector
```

> Uses prebuilt wheels for NumPy/SciPy on common platforms; keep Python/pip recent for smooth installs.

---

## Quickstart (Python API)

```python
import pandas as pd
from rsi_divergence import calculate_rsi, find_divergences

# Price series with DatetimeIndex
close = pd.read_csv("ohlc.csv", index_col=0, parse_dates=True)["close"]

# Wilder-style RSI (period=14)
rsi = calculate_rsi(close, period=14)

# Detect divergences using adaptive defaults
divs = find_divergences(
    prices=close,
    rsi=rsi,
    rsi_period=14,       # used for adaptive defaults
    max_lag=3,           # bars allowed between paired price/RSI pivots
    include_hidden=True, # include hidden (continuation) divergences
)

print(divs.head())
```

---

## Quickstart (CLI)

Read a CSV with a `close` column (index = timestamps):

```bash
rsi_divergence --file ohlc.csv --rsi-period 14
```

Optional tuning (omit to use adaptive defaults):

```bash
rsi_divergence \
  --file ohlc.csv \
  --rsi-period 14 \
  --price-prominence 0.25 \
  --rsi-prominence 5 \
  --price-width 2 \
  --rsi-width 2 \
  --distance 7 \
  --max-lag 3 \
  --include-hidden
```

> The CLI is built with **Typer**; `--help` shows all options and help text.

---

## Returned Schema

The detector returns a `pd.DataFrame` with one row per divergence:

```
['kind','p1_idx','p1_price','p2_idx','p2_price','r1_idx','r1_val','r2_idx','r2_val']
```

* `kind`: one of `regular_bullish`, `regular_bearish`, `hidden_bullish`, `hidden_bearish`
* `p1_idx`, `p2_idx`: timestamps of the price pivots (first → second)
* `p1_price`, `p2_price`: price at those pivots
* `r1_idx`, `r2_idx`: timestamps of the paired RSI pivots
* `r1_val`, `r2_val`: RSI values at those pivots

---

## Concepts & Algorithm

### Divergences we detect

* **Regular bullish**: price makes **lower low (LL)**, RSI makes **higher low (HL)** → potential reversal up.
* **Regular bearish**: price makes **higher high (HH)**, RSI makes **lower high (LH)** → potential reversal down.
* **Hidden bullish**: price **HL**, RSI **LL** → continuation of **uptrend**.
* **Hidden bearish**: price **LH**, RSI **HH** → continuation of **downtrend**.

> Hidden divergences are commonly framed as **trend continuation**; regular divergences as **reversal‑leaning**.

### RSI (Wilder‑style)

We compute RSI exactly as described by Wilder: average gains/losses smoothed with **Wilder’s smoothing** (a form of exponential smoothing with $\alpha = 1/\text{period}$), then
$\text{RSI} = 100 - \frac{100}{1+RS} \quad \text{where} \quad RS = \frac{\text{avg gain}}{\text{avg loss}}.$
The output is aligned to the input index and lives in **\[0, 100]** (initial warm‑up may contain NaNs).

### Peak detection (SciPy)

We locate local extrema on **price** and **RSI** using `scipy.signal.find_peaks`. Minima are obtained by applying `find_peaks` to the **negated series**. The following gates control selection:

* **prominence** — how much a peak stands out relative to its surroundings
* **width** — peak “thickness” in samples
* **distance** — minimum separation between neighboring peaks

**Adaptive defaults** (used when the parameter is `None`):

* `price_prominence` ≈ `0.5 * rolling_std(pct_change, window=rsi_period).iloc[-1] * last_price` (scale‑aware)
* `rsi_prominence` = `5.0` (RSI points)
* `price_width` = `2`, `rsi_width` = `2`
* `distance` = `max(1, rsi_period // 2)`

### Monotonic pivot pairing

For each **consecutive price pivot pair** (minima for bullish paths, maxima for bearish), we pick the **nearest‑in‑time RSI pivots** within ±`max_lag` bars. Pairing is **monotonic**: the second RSI pivot must occur **after** the first RSI pivot used in that loop, and we don’t re‑use RSI pivots out of order. When comparisons below hold, we append a row:

* `regular_bullish`: `price2 < price1` **and** `rsi2 > rsi1`
* `regular_bearish`: `price2 > price1` **and** `rsi2 < rsi1`
* `hidden_bullish`: `price2 > price1` **and** `rsi2 < rsi1`
* `hidden_bearish`: `price2 < price1` **and** `rsi2 > rsi1`

`allow_equal=True` (default) makes comparisons inclusive (tolerant to ties on intrabar data).

---

## API Reference

### `calculate_rsi`

```python
calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series
```

**Contract**

* `prices`: non‑empty `pd.Series` with a **monotonic increasing `DatetimeIndex`**.
* Returns an RSI `pd.Series` aligned to `prices.index` with values in **\[0,100]** (warm‑up may contain NaNs).
* Implements **Wilder‑style** smoothing (equivalent to an EMA with `alpha = 1/period`, `adjust=False`).

### `find_divergences`

```python
find_divergences(
    prices: pd.Series,
    rsi: pd.Series,
    *,
    rsi_period: int = 14,
    price_prominence: float | None = None,
    rsi_prominence: float | None = None,
    price_width: int | None = None,
    rsi_width: int | None = None,
    distance: int | None = None,
    max_lag: int = 3,
    include_hidden: bool = True,
    allow_equal: bool = True,
) -> pd.DataFrame
```

**Inputs**

* `prices`, `rsi`: numeric `pd.Series` with the **same `DatetimeIndex`** (RSI will be reindexed to `prices`).
* `rsi` can have initial NaNs (warm‑up).
* **Adaptive defaults** apply when gates are `None` (see above).

**Pairing & rules**

* For each consecutive price pivot pair, find nearest RSI pivots within ±`max_lag` bars, enforcing **monotonic time**.
* Append a row when the rule for the target divergence holds (see [Monotonic pivot pairing](#monotonic-pivot-pairing)).

**Returns**

* A tidy `pd.DataFrame` with the [columns described here](#returned-schema).

---

## Tuning Tips

* Too **few** pivots? Lower `price_prominence` / `rsi_prominence` and/or `distance`, or increase `max_lag`.
* Too **many/weak** pivots? Increase **prominence** and **width**.
* Intraday noise: increase `width` and `distance` to suppress micro‑wiggles.
* Interested in classical RSI zones (≈30/70)? Filter results post‑hoc by `r1_val`/`r2_val` ranges. (Heuristics, not hard rules.)

---

## Examples

**Minimal flow**

```python
import pandas as pd
from rsi_divergence import calculate_rsi, find_divergences

close = pd.read_csv("ohlc.csv", index_col=0, parse_dates=True)["close"]
rsi = calculate_rsi(close, period=14)
divs = find_divergences(close, rsi, rsi_period=14, include_hidden=True)
print(divs.tail())
```

**Filtering for “strong” bearish signals**

```python
strong = divs[(divs["kind"].str.contains("bearish")) & (divs["r2_val"] > 65.0)]
```

**Plotting (optional)**
Layer `p1/p2` and `r1/r2` pivots over candles (e.g., with `mplfinance`) to visualize each divergence. Plotting is intentionally **out‑of‑scope** here to keep the core lightweight.

---

## Testing

We ship unit tests and property tests:

* **Unit tests**: RSI contract, range, Wilder consistency; deterministic divergence scenarios; CLI smoke test.
* **Property tests** (Hypothesis): generator‑driven random but valid series; **no crashes**, **stable schema**.

Run the suite:

```bash
pip install -r requirements-dev.txt  # or: pip install -e ".[dev]"
pytest -q
```

---

## Versioning / Compatibility

* **Python**: 3.9–3.13
* **NumPy**: ≥ 1.26 (works with NumPy 2.x)
* **pandas**: ≥ 2.2.2
* **SciPy**: ≥ 1.13

We avoid strict upper pins to reduce resolver conflicts across environments.

---

## Limitations & Notes

* Divergences can **persist** without immediate reversal/continuation; confirm with broader structure/flow.
* Peak‑based methods are **parameter‑sensitive**; adaptive defaults help, but **context matters** (market, timeframe).
* This library **does not** predict or place orders; it **annotates structure** to support your own analysis.

---

## Roadmap

A forward‑looking sketch (subject to change):

* Optional **zone‑aware scoring** (down‑weight signals far from RSI 30/70 bands).
* **Multi‑timeframe** aggregation (e.g., 1h pivots confirming 5m signals).
* **Volume/OI/CVD** hooks for richer filters.
* **Numba/JIT** fast‑path for large universes.
* Streaming examples (WebSocket → rolling detection).

PRs welcome!

---

## Contributing

* Keep public contracts stable; add/adjust tests when behavior changes.
* Maintain stateless, explicit‑parameter functions (no hidden globals).
* Run `pytest` locally before pushing.

---

## License

MIT — do what you like, include the license.

---

## References

* **RSI (Wilder)** and smoothing background — StockCharts “RSI” notes; TC2000 help; Wikipedia overview.

  * StockCharts: [https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi)
  * TC2000 Help: [https://help.tc2000.com/m/69404/l/747071-rsi-wilder-s-rsi](https://help.tc2000.com/m/69404/l/747071-rsi-wilder-s-rsi)
  * Wikipedia: [https://en.wikipedia.org/wiki/Relative\_strength\_index](https://en.wikipedia.org/wiki/Relative_strength_index)
* **Divergence** (regular vs hidden) — Babypips primers; Investopedia overviews.

  * Babypips Hidden Divergence: [https://www.babypips.com/learn/forex/hidden-divergence](https://www.babypips.com/learn/forex/hidden-divergence)
  * Babypips Divergence Cheatsheet: [https://www.babypips.com/learn/forex/divergence-cheat-sheet](https://www.babypips.com/learn/forex/divergence-cheat-sheet)
  * Investopedia Divergence: [https://www.investopedia.com/terms/d/divergence.asp](https://www.investopedia.com/terms/d/divergence.asp)
* **SciPy peak selection** — `find_peaks`, `peak_prominences`, `peak_widths`.

  * find\_peaks: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find\_peaks.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)
  * peak\_prominences: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak\_prominences.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html)
  * peak\_widths: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak\_widths.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html)
* **Typer (CLI)** — official docs.

  * Typer: [https://typer.tiangolo.com/](https://typer.tiangolo.com/)

---

## SEO Keywords

RSI divergence detector, RSI hidden divergence, Wilder RSI Python, RSI Wilder smoothing, RSI divergence Python library, SciPy find\_peaks RSI, RSI divergence CLI, regular vs hidden divergence, price oscillator divergence, RSI 30/70 heuristic, pandas RSI, quantitative trading divergence, Python technical analysis.
