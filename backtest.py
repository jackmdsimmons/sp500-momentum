"""
Quintile analysis of momentum metrics vs. forward returns.

For each momentum metric:
  - Sort monthly observations into quintiles (Q1=lowest, Q5=highest)
  - Calculate mean forward return for each quintile
  - Report Q5-Q1 spread

Forward horizons: 1m, 3m, 6m, 12m
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


TRADING_DAYS_PER_MONTH = 21


def monthly_returns(prices: pd.Series) -> pd.Series:
    monthly = prices.resample("ME").last()
    return monthly.pct_change().dropna()


def forward_returns(prices: pd.Series, horizons: list[int] = [1, 3, 6, 12]) -> pd.DataFrame:
    """
    Compute forward returns at multiple horizons from month-end prices.
    horizon=1 means next month's return, horizon=3 means next 3 months, etc.
    """
    monthly = prices.resample("ME").last()
    fwd = pd.DataFrame(index=monthly.index)
    for h in horizons:
        fwd[f"fwd_{h}m"] = monthly.pct_change(h).shift(-h)
    return fwd.dropna()


def quintile_analysis(
    signal: pd.Series,
    fwd: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bin signal into quintiles, compute mean forward return per quintile.
    Returns a DataFrame: rows = quintiles (Q1..Q5), columns = forward horizons.
    """
    sig_monthly = signal.resample("ME").last().shift(1)  # signal known at month-end, applied next month
    combined = pd.concat([sig_monthly.rename("signal"), fwd], axis=1).dropna()

    # Check we have enough unique values to form 5 bins
    if combined["signal"].nunique() < 5:
        return None

    results = {}
    for col in fwd.columns:
        combined["quintile"] = pd.qcut(
            combined["signal"], 5, labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop"
        )
        means = combined.groupby("quintile", observed=True)[col].mean() * 100
        results[col] = means

    return pd.DataFrame(results)


def spread_table(prices: pd.Series, metrics: pd.DataFrame) -> pd.DataFrame:
    """
    For each metric and each forward horizon, compute Q5-Q1 spread.
    Returns a DataFrame: rows = metrics, columns = forward horizons.
    """
    fwd = forward_returns(prices)
    spreads = {}

    for col in metrics.columns:
        qt = quintile_analysis(metrics[col], fwd)
        if qt is None:
            continue  # skip discrete metrics like tsmom
        spreads[col] = (qt.loc["Q5"] - qt.loc["Q1"]).round(2)

    return pd.DataFrame(spreads).T


def plot_quintile_bars(prices: pd.Series, metrics: pd.DataFrame, horizon: str = "fwd_1m"):
    """
    For a given forward horizon, plot mean return by quintile for each metric.
    """
    fwd = forward_returns(prices)
    valid_cols = [c for c in metrics.columns if quintile_analysis(metrics[c], fwd) is not None]
    n_metrics = len(valid_cols)
    ncols = 4
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharey=False)
    axes = axes.flatten()
    colors = ["#d73027", "#fc8d59", "#fee090", "#91cf60", "#1a9850"]

    for i, col in enumerate(valid_cols):
        ax = axes[i]
        qt = quintile_analysis(metrics[col], fwd)
        vals = qt[horizon]
        bars = ax.bar(vals.index, vals.values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(col, fontsize=11)
        ax.set_ylabel("Mean fwd return (%)" if i % ncols == 0 else "")
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Annotate spread
        spread = vals["Q5"] - vals["Q1"]
        ax.text(0.98, 0.02, f"Q5-Q1: {spread:.2f}%",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    h = horizon.replace("fwd_", "").replace("m", "")
    fig.suptitle(f"Mean Forward {h}-Month Return by Momentum Quintile", fontsize=13, y=1.01)
    plt.tight_layout()
    fname = f"data/quintiles_{horizon}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {fname}")


def plot_spread_heatmap(spreads: pd.DataFrame):
    """
    Heatmap of Q5-Q1 spread for all metrics x forward horizons.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        spreads.astype(float),
        annot=True, fmt=".2f", center=0,
        cmap="RdYlGn", ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Q5-Q1 spread (%)"},
    )
    ax.set_title("Q5-Q1 Forward Return Spread by Metric and Horizon (%)")
    ax.set_xlabel("Forward Horizon")
    ax.set_ylabel("Momentum Metric")
    plt.tight_layout()
    plt.savefig("data/spread_heatmap.png", dpi=150)
    plt.close()
    print("Chart saved to data/spread_heatmap.png")


def rolling_spread(
    prices: pd.Series,
    metrics: pd.DataFrame,
    horizon: str = "fwd_1m",
    window: int = 36,
) -> pd.DataFrame:
    """
    Compute Q5-Q1 spread over a rolling window of months.
    Returns a DataFrame: index = month-end dates, columns = metrics.
    """
    fwd = forward_returns(prices)
    valid_cols = [c for c in metrics.columns if quintile_analysis(metrics[c], fwd) is not None]

    results = {col: {} for col in valid_cols}

    # Build monthly signal + forward return table for each metric
    for col in valid_cols:
        sig_monthly = metrics[col].resample("ME").last().shift(1)
        combined = pd.concat([sig_monthly.rename("signal"), fwd[horizon]], axis=1).dropna()

        # Roll through time
        for end_idx in range(window, len(combined) + 1):
            window_data = combined.iloc[end_idx - window : end_idx]
            date = window_data.index[-1]

            if window_data["signal"].nunique() < 5:
                continue

            window_data = window_data.copy()
            window_data["quintile"] = pd.qcut(
                window_data["signal"], 5, labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop"
            )
            means = window_data.groupby("quintile", observed=True)[horizon].mean() * 100

            if "Q5" in means.index and "Q1" in means.index:
                results[col][date] = means["Q5"] - means["Q1"]

    return pd.DataFrame(results)


def plot_rolling_spread(
    prices: pd.Series,
    metrics: pd.DataFrame,
    horizon: str = "fwd_1m",
    window: int = 36,
):
    """
    Plot rolling Q5-Q1 spread over time for each metric.
    """
    spreads = rolling_spread(prices, metrics, horizon, window)

    n = len(spreads.columns)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), sharey=False)
    axes = axes.flatten()

    for i, col in enumerate(spreads.columns):
        ax = axes[i]
        s = spreads[col].dropna()
        ax.plot(s.index, s.values, color="steelblue", linewidth=1.2)
        ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
        ax.fill_between(s.index, s.values, 0,
                        where=s.values > 0, alpha=0.15, color="green")
        ax.fill_between(s.index, s.values, 0,
                        where=s.values < 0, alpha=0.15, color="red")
        ax.set_title(col, fontsize=11)
        ax.set_ylabel("Q5-Q1 spread (%)")
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    h = horizon.replace("fwd_", "").replace("m", "")
    fig.suptitle(
        f"{window}-Month Rolling Q5-Q1 Spread — {h}-Month Forward Return",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    fname = f"data/rolling_spread_{horizon}_{window}m.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {fname}")


def plot_yearly_heatmap(
    prices: pd.Series,
    metrics: pd.DataFrame,
    horizon: str = "fwd_1m",
):
    """
    Heatmap: rows = metrics, columns = years, cells = Q5-Q1 spread for that year.
    Shows clearly which metrics worked in which regimes.
    """
    fwd = forward_returns(prices)
    valid_cols = [c for c in metrics.columns if quintile_analysis(metrics[c], fwd) is not None]

    yearly = {}
    for col in valid_cols:
        sig_monthly = metrics[col].resample("ME").last().shift(1)
        combined = pd.concat([sig_monthly.rename("signal"), fwd[horizon]], axis=1).dropna()
        combined["year"] = combined.index.year

        row = {}
        for year, grp in combined.groupby("year"):
            if grp["signal"].nunique() < 5 or len(grp) < 10:
                continue
            # Use 20th/80th percentile cutoffs instead of fixed quintile labels
            lo = grp["signal"].quantile(0.20)
            hi = grp["signal"].quantile(0.80)
            q1_ret = grp.loc[grp["signal"] <= lo, horizon].mean() * 100
            q5_ret = grp.loc[grp["signal"] >= hi, horizon].mean() * 100
            row[year] = round(q5_ret - q1_ret, 2)
        yearly[col] = row

    df = pd.DataFrame(yearly).T.sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 0.7), 5))
    sns.heatmap(
        df.astype(float),
        annot=True, fmt=".1f", center=0,
        cmap="RdYlGn", ax=ax,
        linewidths=0.4,
        cbar_kws={"label": "Q5-Q1 spread (%)"},
    )
    h = horizon.replace("fwd_", "").replace("m", "")
    ax.set_title(f"Q5-Q1 Spread by Metric and Year — {h}-Month Forward Return (%)")
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    plt.tight_layout()
    fname = f"data/yearly_heatmap_{horizon}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {fname}")
