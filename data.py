"""
Data pipeline: regional Fama-French factors + one ETF per region.
Monthly, USD, excess returns ready for regression.
"""
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

# One entry per region. Check names with:
#   from pandas_datareader.famafrench import get_available_datasets
REGIONS = {
    "US":     {"ff": "F-F_Research_Data_Factors",        "mom": "F-F_Momentum_Factor",              "etf": "SPY"},
    "Europe": {"ff": "Europe_3_Factors",                 "mom": "Europe_Mom_Factor",                "etf": "VGK"},
    "Japan":  {"ff": "Japan_3_Factors",                  "mom": "Japan_Mom_Factor",                 "etf": "EWJ"},
    # Note: French's region includes Australia/NZ; AAXJ doesn't. Mention this in the write-up.
    "Asia":   {"ff": "Asia_Pacific_ex_Japan_3_Factors",  "mom": "Asia_Pacific_ex_Japan_Mom_Factor", "etf": "AAXJ"},
}
FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RF"]


def load_factors(region, start):
    """Monthly factors for one region, in decimals (French data comes in %)."""
    cfg = REGIONS[region]
    ff = web.DataReader(cfg["ff"], "famafrench", start=start)[0]
    mom = web.DataReader(cfg["mom"], "famafrench", start=start)[0]
    df = ff.join(mom, how="inner") / 100
    df.columns = [c.strip() for c in df.columns]
    # Momentum is called "Mom" in the US file and "WML" in regional files
    df = df.rename(columns={c: "MOM" for c in df.columns if c not in FACTOR_COLS})
    return df


def load_etf_returns(ticker, start):
    """Monthly ETF returns. Same cleaning rules as the tape-cleaning task."""
    px = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px = px.sort_index()
    px = px[~px.index.duplicated(keep="first")]   # drop duplicate dates
    px = px.where(px > 0).ffill()                  # bad prints -> NaN, forward-fill only
    monthly = px.groupby(px.index.to_period("M")).last()   # month-end price
    return monthly.pct_change().dropna().rename(ticker)


def build_region_dataset(region, start="2008-01-01"):
    """Factors + ETF excess return, aligned on month."""
    ticker = REGIONS[region]["etf"]
    df = load_factors(region, start).join(load_etf_returns(ticker, start), how="inner")
    df["ETF_excess"] = df[ticker] - df["RF"]
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    for region in REGIONS:
        df = build_region_dataset(region)
        df.to_csv(f"data/{region}.csv")
        print(f"{region}: {len(df)} months, {df.index.min()} to {df.index.max()}")
