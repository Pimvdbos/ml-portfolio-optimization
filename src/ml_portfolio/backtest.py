from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor

TRADING_DAYS = 252
MONTH_DAYS = 21


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2018-01-01"
    end: str = "2024-12-31"
    backtest_start: str = "2023-01-01"
    lookback_days: int = 756  # ~3 years
    horizon_days: int = MONTH_DAYS
    n_lags: int = 5
    n_estimators: int = 30
    max_weight: float = 0.40
    uncertainty_penalty: float = 0.50
    transaction_cost_bps: float = 10.0
    random_state: int = 42


def load_prices(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    compact_parts = [data_dir / "etf_prices_2018_2021.csv", data_dir / "etf_prices_2022_2024.csv"]
    if all(path.exists() for path in compact_parts):
        prices = pd.concat([pd.read_csv(path, parse_dates=["Date"]) for path in compact_parts], ignore_index=True)
        return prices.set_index("Date").sort_index()

    files = {
        "SPY": "spy_us_d.csv",
        "EFA": "efa_us_d.csv",
        "AGG": "agg_us_d.csv",
        "GLD": "gld_us_d.csv",
        "VNQ": "vnq_us_d.csv",
    }
    frames = []
    for ticker, filename in files.items():
        df = pd.read_csv(data_dir / filename, parse_dates=["Date"])
        frame = df[["Date", "Close"]].rename(columns={"Close": ticker})
        frames.append(frame)
    prices = frames[0]
    for frame in frames[1:]:
        prices = prices.merge(frame, on="Date", how="inner")
    return prices.set_index("Date").sort_index()


def log_returns_from_prices(prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    prices = prices.loc[start:end]
    return np.log(prices / prices.shift(1)).dropna()


def make_supervised(series: pd.Series, n_lags: int, horizon: int) -> pd.DataFrame:
    """Features known at t; target is cumulative log return over t+1..t+horizon."""
    out = pd.DataFrame(index=series.index)
    for lag in range(n_lags):
        out[f"ret_lag_{lag}"] = series.shift(lag)
    out["momentum_21"] = series.rolling(21).sum()
    out["vol_21"] = series.rolling(21).std()
    out["target"] = sum(series.shift(-k) for k in range(1, horizon + 1))
    return out.dropna()


def latest_features(series: pd.Series, n_lags: int) -> pd.DataFrame:
    vals = {f"ret_lag_{lag}": series.iloc[-1 - lag] for lag in range(n_lags)}
    vals["momentum_21"] = series.iloc[-21:].sum()
    vals["vol_21"] = series.iloc[-21:].std()
    return pd.DataFrame([vals])


def optimize_max_sharpe(mu: np.ndarray, cov: np.ndarray, max_weight: float) -> np.ndarray:
    n = len(mu)
    init = np.repeat(1 / n, n)
    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w: np.ndarray) -> float:
        vol = float(np.sqrt(w @ cov @ w))
        if vol <= 1e-12:
            return 1e6
        return -float(w @ mu) / vol

    res = minimize(objective, init, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x


def risk_parity_weights(cov: np.ndarray, max_weight: float) -> np.ndarray:
    n = cov.shape[0]
    init = np.repeat(1 / n, n)
    bounds = [(1e-8, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w: np.ndarray) -> float:
        port_var = float(w @ cov @ w)
        if port_var <= 1e-16:
            return 1e6
        marginal = cov @ w
        rc = w * marginal / port_var
        target = np.repeat(1 / n, n)
        return float(np.sum((rc - target) ** 2))

    res = minimize(objective, init, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"Risk-parity optimization failed: {res.message}")
    return res.x


def month_end_trading_dates(returns: pd.DataFrame, start: str) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start)
    prior_month = (start_ts - pd.offsets.MonthBegin(1)).normalize()
    subset = returns.loc[prior_month:]
    return pd.DatetimeIndex(subset.groupby(subset.index.to_period("M")).apply(lambda x: x.index[-1]).values)


def predict_ml_monthly_return(
    asset_returns: pd.Series,
    config: BacktestConfig,
) -> Tuple[float, float]:
    supervised = make_supervised(asset_returns, config.n_lags, config.horizon_days)
    if len(supervised) < 100:
        raise ValueError("Insufficient training observations")
    X = supervised.drop(columns="target")
    y = supervised["target"]
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        min_samples_leaf=5,
        random_state=config.random_state,
        n_jobs=1,
    )
    model.fit(X, y)
    x_latest = latest_features(asset_returns, config.n_lags)
    tree_preds = np.array([tree.predict(x_latest.values)[0] for tree in model.estimators_])
    return float(tree_preds.mean()), float(tree_preds.std(ddof=0))


def run_walk_forward(log_returns: pd.DataFrame, config: BacktestConfig):
    dates = month_end_trading_dates(log_returns, config.backtest_start)
    strategies = ["ML Robust", "Historical MVO", "Equal Weight", "Risk Parity"]
    weights: Dict[str, list[pd.Series]] = {s: [] for s in strategies}
    gross_returns: Dict[str, list[pd.Series]] = {s: [] for s in strategies}
    net_returns: Dict[str, list[pd.Series]] = {s: [] for s in strategies}
    forecast_rows = []
    prev_weights = {s: np.zeros(log_returns.shape[1]) for s in strategies}
    assets = list(log_returns.columns)
    tc_rate = config.transaction_cost_bps / 10000.0

    for i, d in enumerate(dates[:-1]):
        next_d = dates[i + 1]
        train = log_returns.loc[:d].tail(config.lookback_days)
        realized = log_returns.loc[(log_returns.index > d) & (log_returns.index <= next_d)]
        if realized.empty or len(train) < 252:
            continue

        cov_month = train.cov().values * config.horizon_days
        hist_mu = train.mean().values * config.horizon_days

        ml_mu, ml_sigma = [], []
        for asset in assets:
            mean_pred, std_pred = predict_ml_monthly_return(train[asset], config)
            ml_mu.append(mean_pred)
            ml_sigma.append(std_pred)
            forecast_rows.append({
                "rebalance_date": d,
                "asset": asset,
                "predicted_return": mean_pred,
                "tree_dispersion": std_pred,
                "robust_return": mean_pred - config.uncertainty_penalty * std_pred,
            })
        ml_mu = np.asarray(ml_mu)
        ml_sigma = np.asarray(ml_sigma)
        robust_mu = ml_mu - config.uncertainty_penalty * ml_sigma

        w_map = {
            "ML Robust": optimize_max_sharpe(robust_mu, cov_month, config.max_weight),
            "Historical MVO": optimize_max_sharpe(hist_mu, cov_month, config.max_weight),
            "Equal Weight": np.repeat(1 / len(assets), len(assets)),
            "Risk Parity": risk_parity_weights(cov_month, config.max_weight),
        }

        for strategy, w in w_map.items():
            gross = pd.Series(realized.values @ w, index=realized.index, name=strategy)
            turnover = float(np.sum(np.abs(w - prev_weights[strategy])))
            net = gross.copy()
            if len(net):
                net.iloc[0] -= tc_rate * turnover
            gross_returns[strategy].append(gross)
            net_returns[strategy].append(net)
            weights[strategy].append(pd.Series(w, index=assets, name=d))
            prev_weights[strategy] = w

    gross = pd.DataFrame({k: pd.concat(v) for k, v in gross_returns.items()})
    net = pd.DataFrame({k: pd.concat(v) for k, v in net_returns.items()})
    weight_frames = {k: pd.DataFrame(v) for k, v in weights.items()}
    forecasts = pd.DataFrame(forecast_rows)
    return gross, net, weight_frames, forecasts


def performance_summary(log_return_df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for col in log_return_df:
        r = log_return_df[col].dropna()
        simple = np.exp(r) - 1.0
        wealth = np.exp(r.cumsum())
        ann_return = float(np.exp(r.mean() * TRADING_DAYS) - 1.0)
        ann_vol = float(simple.std(ddof=1) * np.sqrt(TRADING_DAYS))
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        downside = simple[simple < 0]
        downside_dev = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else np.nan
        sortino = ann_return / downside_dev if downside_dev and downside_dev > 0 else np.nan
        max_dd = float((wealth / wealth.cummax() - 1.0).min())
        total_return = float(wealth.iloc[-1] - 1.0)
        rows[col] = {
            "Total Return": total_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown": max_dd,
        }
    return pd.DataFrame(rows).T
