import numpy as np
import pandas as pd

from ml_portfolio.backtest import make_supervised, optimize_max_sharpe, risk_parity_weights


def test_forward_target_uses_future_returns_only():
    s = pd.Series(np.arange(1, 50, dtype=float), index=pd.date_range('2020-01-01', periods=49))
    df = make_supervised(s, n_lags=2, horizon=3)
    t = df.index[0]
    i = s.index.get_loc(t)
    expected = s.iloc[i+1:i+4].sum()
    assert np.isclose(df.loc[t, 'target'], expected)


def test_max_sharpe_weights_respect_constraints():
    mu = np.array([0.03, 0.02, 0.01])
    cov = np.diag([0.04, 0.03, 0.02])
    w = optimize_max_sharpe(mu, cov, max_weight=0.5)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-8).all()
    assert (w <= 0.5 + 1e-8).all()


def test_risk_parity_weights_respect_constraints():
    cov = np.diag([0.04, 0.03, 0.02])
    w = risk_parity_weights(cov, max_weight=0.6)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    assert (w <= 0.6 + 1e-8).all()
