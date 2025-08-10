# RSI Divergence Detector

A complete, industry-grade RSI divergence detector.

## Installation

```bash
pip install rsi-divergence
```

## Usage

### CLI

```bash
rsi_divergence --file data.csv
```

### Python API

```python
import pandas as pd
from rsi_divergence.core import find_divergences

df = pd.read_csv("data.csv")
divergences = find_divergences(df)
print(divergences)
```
