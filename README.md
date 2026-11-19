# AI-Based Inventory Demand Forecasting System for Retail Businesses

Working implementation of the research paper: Flask backend + ML pipeline + interactive dashboard.

## Run in VS Code (3 steps)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py                  # open http://127.0.0.1:5000
```
First start auto-trains the models (about a minute). To retrain manually: `python train.py`.

## Using the real dataset
Download "Retail Store Inventory Forecasting Dataset" (Kaggle, 73,100 rows, 15 columns) and save it as
`data/retail_store_inventory.csv`, then delete the `models/` and `outputs/` folders contents and run `python train.py`.
Without it, `generate_data.py` creates a synthetic dataset with the identical schema.

## Mapping to the paper
| Paper section | Where |
|---|---|
| III-C Preprocessing (duplicates, missing, encoding) | `ml.load_clean`, `make_pipeline` |
| III-E Feature engineering (calendar, lag, rolling; Eq. 2) | `ml.time_features`, `ml.add_features` |
| III-F Models (Linear Regression, Random Forest, HistGradientBoosting) | `ml.train` |
| III-G Chronological split (last 9,300 rows = test) | `ml.train` |
| III-H Metrics MAE, RMSE, WAPE, sMAPE, R2 | `ml.metrics` |
| III-I Wilcoxon signed-rank test (Eq. 5) | `ml.train` |
| III-J Inventory layer, Coverage = I/D (Eq. 6) | `ml.coverage`, `ml.status_of` |
| "Demand Forecast" column excluded to prevent leakage | `ml.CAT`, `ml.NUM` |

## Extra experiments beyond the paper (Experiments tab)
- Feature ablation (lag, price/promotion, calendar-only) for Linear Regression and HistGradientBoosting
- Leakage demonstration: why `Demand Forecast` is excluded (R2 jumps to ~0.98 if included)
- Error by category and by store (product heterogeneity, paper refs [9][16])
- 90% prediction interval, safety stock and reorder point (paper future scope: probabilistic forecasting)
- Architecture tab reproduces paper Fig. 1, 2 and 3; inventory report downloads as CSV; upload your own CSV to retrain

## Tests
`pip install pytest` then `python -m pytest tests`

## API
- `GET /api/summary` EDA, metrics, Wilcoxon, importance
- `POST /api/predict` forecast + coverage status + reorder quantity
- `GET /api/risk?store=S001&status=Low coverage` inventory-risk table

## Structure
```
app.py  ml.py  train.py  generate_data.py  requirements.txt
static/index.html    data/    models/    outputs/
```
Split is exactly 63,800 train / 9,300 test, as in the paper. Early lag values are filled using past data only.

The bundled dataset is SYNTHETIC, calibrated to the paper's Table I (mean ~136 units, SD ~109, ~0.5% zero demand).
Replace it with the real Kaggle CSV for final results and state this clearly in your report.
