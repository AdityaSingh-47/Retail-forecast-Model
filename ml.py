"""Core ML module: preprocessing, feature engineering, 3 models, metrics,
Wilcoxon test, and inventory-risk layer (Coverage = I_t / D_hat_t)."""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).parent
DATA = BASE / "data" / "retail_store_inventory.csv"
MODELS, OUT = BASE / "models", BASE / "outputs"
TARGET = "Units Sold"
CAT = ["Store ID", "Product ID", "Category", "Region", "Weather Condition", "Seasonality"]
NUM = ["Inventory Level", "Units Ordered", "Price", "Discount", "Competitor Pricing", "Price Gap",
       "Holiday/Promotion", "Month", "DayOfWeek", "DayOfMonth", "WeekOfYear", "IsWeekend",
       "Lag1", "Lag7", "Roll7"]
TEST_SIZE = 9300            # last 9,300 rows = chronological test set (as in the paper)
THRESH = {"stockout": 1.0, "low": 1.5, "excess": 3.0}   # coverage thresholds (days of demand)
TARGET_COVER = 2.0


# ---------- preprocessing & features ----------
def load_clean(path=DATA):
    df = pd.read_csv(path).drop_duplicates()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=[TARGET, "Date"])
    for c in df.columns:                                   # simple missing-value handling
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median() if df[c].dtype != object else df[c].mode()[0])
    return df.sort_values(["Date", "Store ID", "Product ID"]).reset_index(drop=True)


def time_features(df):
    d = df["Date"]
    df["Month"], df["DayOfWeek"], df["DayOfMonth"] = d.dt.month, d.dt.dayofweek, d.dt.day
    df["WeekOfYear"] = d.dt.isocalendar().week.astype(int)
    df["IsWeekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["Price Gap"] = df["Price"] - df["Competitor Pricing"]
    return df


def add_features(raw):
    df = time_features(raw.copy())
    g = df.groupby(["Store ID", "Product ID"])[TARGET]
    df["Lag1"], df["Lag7"] = g.shift(1), g.shift(7)                       # Eq. (2): L_t^(k) = D_{t-k}
    df["Roll7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    fill = df[TARGET].iloc[:max(1, len(df) - TEST_SIZE)].mean()          # training-period mean only (no leakage)
    df["Roll7"] = df["Roll7"].fillna(fill)
    df["Lag1"], df["Lag7"] = df["Lag1"].fillna(df["Roll7"]), df["Lag7"].fillna(df["Roll7"])
    return df.reset_index(drop=True)


def make_pipeline(model, num=None, cat=None):
    num, cat = NUM if num is None else num, CAT if cat is None else cat
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("model", model)])


# ---------- metrics ----------
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = np.abs(y) + np.abs(p)
    smape = 100 * np.mean(np.where(denom == 0, 0, 2 * np.abs(y - p) / np.where(denom == 0, 1, denom)))
    return {"MAE": round(mean_absolute_error(y, p), 4),
            "RMSE": round(math.sqrt(mean_squared_error(y, p)), 4),
            "WAPE": round(100 * np.abs(y - p).sum() / y.sum(), 4),
            "sMAPE": round(smape, 4),
            "R2": round(r2_score(y, p), 4)}


# ---------- inventory-risk layer (Eq. 6) ----------
def coverage(inv, pred):
    return np.asarray(inv, float) / np.maximum(np.asarray(pred, float), 1.0)


def status_of(cov):
    if cov < THRESH["stockout"]:
        return "Stockout risk"
    if cov < THRESH["low"]:
        return "Low coverage"
    if cov > THRESH["excess"]:
        return "Excess inventory"
    return "Adequate"


def reorder_qty(inv, pred):
    return int(max(0, math.ceil(TARGET_COVER * pred - inv)))


# ---------- training ----------
def train():
    if not DATA.exists():
        from generate_data import generate
        generate()
    MODELS.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    raw = load_clean()
    df = add_features(raw)
    X, y = df[CAT + NUM], df[TARGET]
    split = len(df) - TEST_SIZE                       # chronological split (no shuffling)
    Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    print(f"Train rows: {len(Xtr):,} | Test rows: {len(Xte):,}")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=5,
                                               n_jobs=-1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                              random_state=42),
    }
    fitted, preds, results = {}, {}, {}
    for name, m in models.items():
        pipe = make_pipeline(m).fit(Xtr, ytr)
        p = np.clip(pipe.predict(Xte), 0, None)
        fitted[name], preds[name] = pipe, p
        results[name] = metrics(yte, p)
        print(f"{name:22s}", results[name])

    # Wilcoxon signed-rank test on paired absolute errors (Eq. 5)
    names, tests = list(models), []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.abs(yte.values - preds[names[i]]) - np.abs(yte.values - preds[names[j]])
            try:
                p_val = float(wilcoxon(d).pvalue)
            except ValueError:
                p_val = 1.0
            tests.append({"A": names[i], "B": names[j], "p_value": p_val, "significant": p_val < 0.05,
                          "median_diff": float(np.median(d))})

    best = min(results, key=lambda k: results[k]["RMSE"])
    joblib.dump(fitted, MODELS / "models.joblib")

    # permutation importance on raw input columns (best model)
    idx = np.random.default_rng(0).choice(len(Xte), 2000, replace=False)
    imp = permutation_importance(fitted[best], Xte.iloc[idx], yte.iloc[idx], n_repeats=3,
                                 scoring="neg_mean_absolute_error", random_state=0)
    importance = sorted(({"feature": c, "importance": float(v)} for c, v in zip(X.columns, imp.importances_mean)),
                        key=lambda r: -r["importance"])[:12]

    # inventory-risk table on the test set using the best model
    test = df.iloc[split:].copy()
    test["Predicted"] = preds[best].round(1)
    test["Coverage"] = coverage(test["Inventory Level"], test["Predicted"]).round(2)
    test["Status"] = [status_of(c) for c in test["Coverage"]]
    test["Reorder Qty"] = [reorder_qty(i, p) for i, p in zip(test["Inventory Level"], test["Predicted"])]
    cols = ["Date", "Store ID", "Product ID", "Category", "Inventory Level", "Units Sold",
            "Predicted", "Coverage", "Status", "Reorder Qty"]
    test[cols].assign(Date=test["Date"].dt.strftime("%Y-%m-%d")).to_csv(OUT / "test_predictions.csv", index=False)

    daily = test.assign(Pred_best=preds[best]).groupby("Date").agg(actual=(TARGET, "sum"), pred=("Pred_best", "sum"))
    stats = raw[["Units Sold", "Inventory Level", "Units Ordered", "Price", "Discount", "Competitor Pricing"]]
    monthly = raw.groupby(raw["Date"].dt.to_period("M"))[TARGET].mean()
    hist = {}
    for (s, p), g in raw.groupby(["Store ID", "Product ID"]):
        v = g[TARGET].values
        hist[f"{s}|{p}"] = {"Lag1": float(v[-1]), "Lag7": float(v[-7]), "Roll7": float(v[-7:].mean())}
    options = {c: sorted(raw[c].astype(str).unique().tolist()) for c in CAT}
    # ---- uncertainty (probabilistic view, refs [6][11][18]): residuals from a validation slice of the TRAIN period
    vs = len(Xtr) - 6000
    vp = clone(fitted[best]).fit(Xtr.iloc[:vs], ytr.iloc[:vs]).predict(Xtr.iloc[vs:])
    vres = ytr.iloc[vs:].values - vp
    q05, q95 = (float(q) for q in np.quantile(vres, [0.05, 0.95]))
    sigma = float(vres.std())
    inside = float(np.mean((yte.values >= preds[best] + q05) & (yte.values <= preds[best] + q95)) * 100)
    uncertainty = {"q05": round(q05, 1), "q95": round(q95, 1), "sigma": round(sigma, 1),
                   "empirical_coverage": round(inside, 1), "safety_stock": int(math.ceil(1.645 * sigma))}
    res = yte.values - preds[best]
    h, edges = np.histogram(res, bins=30)
    residuals = {"labels": [int((edges[i] + edges[i + 1]) / 2) for i in range(len(h))], "counts": h.tolist()}

    # ---- experiments: ablation, leakage demonstration, heterogeneity, gap vs baseline
    def fit_eval(mk, num, cat, name, group):
        pipe = make_pipeline(mk(), num, cat).fit(df.iloc[:split][num + cat], ytr)
        return {"group": group, "setting": name, **metrics(yte, np.clip(pipe.predict(df.iloc[split:][num + cat]), 0, None))}
    lagc, pricec = ["Lag1", "Lag7", "Roll7"], ["Price", "Discount", "Competitor Pricing", "Price Gap", "Holiday/Promotion"]
    calc = ["Month", "DayOfWeek", "DayOfMonth", "WeekOfYear", "IsWeekend"]
    hgb = lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=42)
    ablation = []
    for mname, mk in [("Linear Regression", LinearRegression), ("HistGradientBoosting", hgb)]:
        for sname, nn, cc in [("Full feature set", NUM, CAT),
                              ("Without lag / rolling features", [c for c in NUM if c not in lagc], CAT),
                              ("Without price & promotion features", [c for c in NUM if c not in pricec], CAT),
                              ("Calendar features only", calc, [])]:
            r = fit_eval(mk, nn, cc, sname, mname); ablation.append(r)
    leak = None
    if "Demand Forecast" in df.columns:
        leak = {"clean": results["Linear Regression"],
                "leaky": fit_eval(LinearRegression, NUM + ["Demand Forecast"], CAT, "with Demand Forecast", "Linear Regression")}
    base_m = results["Linear Regression"]
    gap = {n: {"MAE_pct": round(100 * (r["MAE"] - base_m["MAE"]) / base_m["MAE"], 2),
               "RMSE_pct": round(100 * (r["RMSE"] - base_m["RMSE"]) / base_m["RMSE"], 2)}
           for n, r in results.items() if n != "Linear Regression"}
    tt = df.iloc[split:].assign(p=preds[best])
    hetero = {}
    for col in ["Category", "Store ID"]:
        hetero[col] = [{"name": str(k), **{m: v for m, v in metrics(g[TARGET], g["p"]).items() if m in ("MAE", "WAPE")}}
                       for k, g in tt.groupby(col)]

    out = {
        "dataset": {"rows": int(len(raw)), "stores": int(raw["Store ID"].nunique()),
                    "products": int(raw["Product ID"].nunique()), "train_rows": int(len(Xtr)),
                    "test_rows": int(len(Xte)), "zero_demand": int((raw[TARGET] == 0).sum()),
                    "date_min": str(raw["Date"].min().date()), "date_max": str(raw["Date"].max().date())},
        "stats": {c: {"mean": round(stats[c].mean(), 2), "std": round(stats[c].std(), 2),
                      "min": float(stats[c].min()), "max": float(stats[c].max())} for c in stats},
        "monthly": {"labels": [str(k) for k in monthly.index], "values": monthly.round(1).tolist()},
        "by_category": raw.groupby("Category")[TARGET].mean().round(1).to_dict(),
        "by_season": raw.groupby("Seasonality")[TARGET].mean().round(1).to_dict(),
        "by_promo": {("Promotion" if k else "No promotion"): round(v, 1)
                     for k, v in raw.groupby("Holiday/Promotion")[TARGET].mean().items()},
        "results": results, "best": best, "wilcoxon": tests, "importance": importance,
        "daily": {"labels": [d.strftime("%d %b") for d in daily.index],
                  "actual": daily["actual"].round(0).tolist(), "pred": daily["pred"].round(0).tolist()},
        "history": hist, "uncertainty": uncertainty, "residuals": residuals, "ablation": ablation,
        "leakage": leak, "gap": gap, "hetero": hetero, "options": options, "thresholds": THRESH, "target_cover": TARGET_COVER,
    }
    (OUT / "metrics.json").write_text(json.dumps(out))
    print(f"Best model by RMSE: {best}. Artifacts saved.")
    return out


if __name__ == "__main__":
    train()
