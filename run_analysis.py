from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_portfolio.backtest import (
    BacktestConfig,
    load_prices,
    log_returns_from_prices,
    performance_summary,
    run_walk_forward,
)

config = BacktestConfig()
prices = load_prices(ROOT / "data")
returns = log_returns_from_prices(prices, config.start, config.end)

gross, net, weights, forecasts = run_walk_forward(returns, config)
summary = performance_summary(net)

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)
summary.to_csv(ROOT / "results" / "performance_summary.csv")
net.to_csv(ROOT / "results" / "daily_log_returns_net.csv")
forecasts.to_csv(ROOT / "results" / "ml_forecasts.csv", index=False)
for name, df in weights.items():
    df.to_csv(ROOT / "results" / f"weights_{name.lower().replace(' ', '_')}.csv")

wealth = np.exp(net.cumsum())
mpl.rcParams["svg.fonttype"] = "none"
plt.figure(figsize=(9, 4.5))
for column in wealth.columns:
    monthly = wealth[column].resample("ME").last()
    plt.plot(monthly.index, monthly, marker="o", markersize=2, label=column)
plt.title("Walk-Forward Portfolio Performance (Net of Transaction Costs)")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(ROOT / "figures" / "cumulative_performance.svg")
plt.close()

avg = pd.DataFrame({name: frame.mean() for name, frame in weights.items()})
avg.to_csv(ROOT / "results" / "average_weights.csv")

print(summary.round(4))
print("\nAverage weights:")
print(avg.round(3))
