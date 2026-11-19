"""Creates a synthetic dataset with the SAME schema as the paper's
'Retail Store Inventory' data (73,100 rows = 731 days x 5 stores x 20 products).
If you have the real Kaggle CSV, put it at data/retail_store_inventory.csv and skip this."""
from pathlib import Path
import numpy as np
import pandas as pd

PATH = Path(__file__).parent / "data" / "retail_store_inventory.csv"


def generate(path=PATH, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=731)
    stores = [f"S{i:03d}" for i in range(1, 6)]
    products = [f"P{i:04d}" for i in range(1, 21)]
    cats = ["Groceries", "Toys", "Electronics", "Furniture", "Clothing"]
    regions = ["North", "South", "East", "West", "North"]
    df = pd.MultiIndex.from_product([dates, stores, products],
                                    names=["Date", "Store ID", "Product ID"]).to_frame(index=False)
    n = len(df)
    df["Category"] = df["Product ID"].map({p: cats[i % 5] for i, p in enumerate(products)})
    df["Region"] = df["Store ID"].map(dict(zip(stores, regions)))
    df["Inventory Level"] = rng.integers(50, 501, n)
    df["Units Ordered"] = rng.integers(20, 201, n)
    df["Price"] = rng.uniform(10, 100, n).round(2)
    df["Discount"] = rng.choice([0, 5, 10, 15, 20], n)
    df["Weather Condition"] = rng.choice(["Sunny", "Rainy", "Cloudy", "Snowy"], n)
    df["Holiday/Promotion"] = rng.binomial(1, 0.5, n)
    df["Competitor Pricing"] = (df["Price"] + rng.normal(0, 8, n)).clip(5, 105).round(2)
    m = df["Date"].dt.month
    df["Seasonality"] = np.select([m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])],
                                  ["Winter", "Spring", "Summer"], "Autumn")
    cat_eff = df["Category"].map({"Groceries": 12, "Toys": -6, "Electronics": 3, "Furniture": -12, "Clothing": 5})
    sea_eff = df["Seasonality"].map({"Winter": -10, "Spring": 0, "Summer": 10, "Autumn": 35})
    store_eff = df["Store ID"].map(dict(zip(stores, [8, -5, 0, 10, -8])))
    prod_eff = df["Product ID"].map(dict(zip(products, rng.normal(0, 8, 20))))
    score = (0.13 * (df["Inventory Level"] - 275) + 28 * df["Holiday/Promotion"] + 2.8 * df["Discount"]
             - 0.45 * (df["Price"] - 55) + 0.5 * (df["Competitor Pricing"] - df["Price"])
             + cat_eff + sea_eff + store_eff + prod_eff + rng.normal(0, 15, n))
    mu = np.exp(1.3 * score / 100)
    mu = mu / mu.mean() * 141                      # mean demand ~136 units, right-skewed like real retail data
    units = np.clip(rng.gamma(2.6, mu / 2.6), 0, 499).round()
    units[rng.random(n) < 0.0045] = 0              # rare zero-demand days (~0.5%)
    df["Units Sold"] = units.astype(int)
    # Demand Forecast is an externally generated column (deliberately EXCLUDED from model inputs -> avoids leakage)
    df["Demand Forecast"] = (df["Units Sold"] + rng.normal(0, 15, n)).clip(0).round(2)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    cols = ["Date", "Store ID", "Product ID", "Category", "Region", "Inventory Level", "Units Sold",
            "Units Ordered", "Demand Forecast", "Price", "Discount", "Weather Condition",
            "Holiday/Promotion", "Competitor Pricing", "Seasonality"]
    Path(path).parent.mkdir(exist_ok=True)
    df[cols].to_csv(path, index=False)
    print(f"Generated {n:,} rows -> {path}")


if __name__ == "__main__":
    generate()
