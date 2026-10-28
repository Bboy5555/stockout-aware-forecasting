# Stockout-Aware Demand Forecasting for Grocery Retail
MIS 382N term project. Data: [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (Dingdong-Inc, CC BY 4.0).

## Getting started (each person, once)
1. Create a GitHub fine-grained token for this repo (Contents: read and write).
2. In Colab, add it as a secret named `GITHUB_TOKEN` (key icon) and turn on Notebook access.
3. Add a shortcut to the shared **Final Project** folder directly in your My Drive.
4. Start every notebook with the setup cell (mount Drive, clone or pull this repo), then:
   ```python
   from src.load_data import DRIVE_DIR, load_daily, load_hourly
   ```

## Data files (in `Stockout Project Data/processed/`)
| File | What it is |
|---|---|
| `daily_model.parquet` | One row per product per day (4.85M rows). **Use this for modeling.** |
| `sample_ids.csv` | The 2,000 products used for heavy models |
| `daily.parquet` | Cleaned daily table without the modeling columns |

Hourly data: `load_hourly(series_ids)`. Load only the products you need (all 50,000 is about 116M rows).

## Key columns
- `series_id`: store + product. `sale_amount`: daily sales (scaled units, no prices).
- `stock_hour6_22_cnt`: empty-shelf hours between 6:00 and 22:00. `any_stockout`: at least one empty hour. `zero_sales_day`: zero sales (what a daily system sees).
- `discount`: 1.0 = no discount, 0.9 = 10% off. Missing when `discount_missing` = 1.
- `set`: train / valid / test. `in_sample`: 1 if in the 2,000-product sample.

## Frozen splits
| Set | Dates (2024) | Days |
|---|---|---|
| Train | Mar 28 to Jun 11 | 76 |
| Validation | Jun 12 to Jun 25 | 14 |
| Test | Jun 26 to Jul 2 | 7 |
Never train on validation or test dates.

## Rules and notes
- **Selling hours are 6:00 to 22:00.** Overnight out-of-stock flags are evening stockouts lasting until the morning restock.
- **Score with `src/metrics.py`** (WAPE, MAE, bias, pinball, recovery error) so results are comparable.
- **Only edit your own notebooks.** Ask before changing files in `src/`.
- Sales in the 2 hours before a stockout run **1.79x** normal, so a plain historical average underestimates missed demand (relevant for Stream 2).
- List any pretrained model or borrowed code you use, for the report's disclosure section.
