"""Shared data loader for FreshRetailNet-50K.

Usage in Colab (after the setup cell):
    from src.load_data import download_raw, build_daily, load_daily, load_hourly
    download_raw()          # once: downloads the raw files to the shared Drive folder
    build_daily()           # once: builds and saves the daily table
    daily = load_daily()    # every time after that
    hourly = load_hourly(series_ids=[...])  # hourly data for chosen series only
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ID = "Dingdong-Inc/FreshRetailNet-50K"
DRIVE_DIR = Path("/content/drive/MyDrive/Stockout Project Data")
RAW_DIR = DRIVE_DIR / "raw"
PROC_DIR = DRIVE_DIR / "processed"
HOURLY_COLS = ["hours_sale", "hours_stock_status"]
SPLITS = ("train", "eval")


def _raw_path(split):
    return RAW_DIR / "data" / f"{split}.parquet"


def _series_id(store_ids, product_ids):
    return pd.Series(store_ids).astype(str).str.cat(pd.Series(product_ids).astype(str), sep="_").to_numpy()


def download_raw():
    """Download train and eval parquet files (about 115 MB) into the shared Drive folder."""
    from huggingface_hub import hf_hub_download
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        hf_hub_download(REPO_ID, f"data/{split}.parquet", repo_type="dataset", local_dir=RAW_DIR)
    print("Raw files saved to", RAW_DIR / "data")


def build_daily(save=True):
    """One row per series per day, without the hourly lists. Saves processed/daily.parquet."""
    frames = []
    for split in SPLITS:
        pf = pq.ParquetFile(_raw_path(split))
        cols = [c for c in pf.schema_arrow.names if c not in HOURLY_COLS]
        df = pf.read(columns=cols).to_pandas()
        df["split"] = split
        frames.append(df)
    daily = pd.concat(frames, ignore_index=True)
    daily["dt"] = pd.to_datetime(daily["dt"])
    daily["series_id"] = _series_id(daily["store_id"], daily["product_id"])
    # Data fix: discount = 0 appears on 0.36% of rows with high sales, so treat it as missing
    daily["discount_missing"] = (daily["discount"] == 0).astype("int8")
    daily.loc[daily["discount"] == 0, "discount"] = np.nan
    daily = daily.sort_values(["series_id", "dt"]).reset_index(drop=True)
    if save:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(PROC_DIR / "daily.parquet", index=False)
        print(f"Saved {len(daily):,} rows to", PROC_DIR / "daily.parquet")
    return daily


def load_daily():
    """Load the saved daily table."""
    return pd.read_parquet(PROC_DIR / "daily.parquet")


def load_hourly(series_ids=None, splits=SPLITS):
    """One row per series per hour: sales and out_of_stock (1 = shelf empty that hour).

    Pass series_ids (for example the 2,000-series sample) to keep memory low.
    Loading all 50,000 series hourly is about 116M rows; avoid it on standard Colab.
    """
    keep = pd.Index(series_ids) if series_ids is not None else None
    parts = []
    for split in splits:
        pf = pq.ParquetFile(_raw_path(split))
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i, columns=["store_id", "product_id", "dt"] + HOURLY_COLS)
            sid = _series_id(t.column("store_id").to_numpy(), t.column("product_id").to_numpy())
            mask = np.ones(len(sid), bool) if keep is None else pd.Index(sid).isin(keep)
            if not mask.any():
                continue
            sales = t.column("hours_sale").combine_chunks().flatten().to_numpy()
            stock = t.column("hours_stock_status").combine_chunks().flatten().to_numpy()
            assert sales.size == 24 * len(sid), "expected 24 hourly values per row"
            sales = sales.reshape(-1, 24)[mask]
            stock = stock.reshape(-1, 24)[mask]
            n = int(mask.sum())
            dts = pd.to_datetime(np.asarray(t.column("dt").to_numpy(zero_copy_only=False))[mask])
            parts.append(pd.DataFrame({
                "series_id": np.repeat(sid[mask], 24),
                "dt": np.repeat(dts, 24),
                "hour": np.tile(np.arange(24, dtype="int8"), n),
                "sales": sales.ravel(),
                "out_of_stock": stock.ravel().astype("int8"),
                "split": split,
            }))
    return pd.concat(parts, ignore_index=True)
