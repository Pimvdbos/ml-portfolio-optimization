# Data

The annual files contain aligned daily closing prices for the five ETFs used in the analysis. They are stored as gzip-compressed CSVs (`.csv.gz`) to keep the repository lightweight; `pandas.read_csv` reads them transparently.

| Ticker | Exposure |
|---|---|
| SPY | U.S. equities |
| EFA | International developed equities |
| AGG | U.S. investment-grade bonds |
| GLD | Gold |
| VNQ | U.S. real estate |

The source series were obtained from **Stooq** for the original university project. Only the 2018–2024 research window is retained.
