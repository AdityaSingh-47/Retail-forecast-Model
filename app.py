"""Flask backend:  python app.py  ->  http://127.0.0.1:5000"""
import json
import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory
import ml

app = Flask(__name__, static_folder="static")
MODELS = META = RISK = None


def load_artifacts():
    global MODELS, META, RISK
    MODELS = joblib.load(ml.MODELS / "models.joblib")
    META = json.loads((ml.OUT / "metrics.json").read_text())
    RISK = pd.read_csv(ml.OUT / "test_predictions.csv")


if not (ml.MODELS / "models.joblib").exists():
    print("No trained models found - training now (about 1-2 minutes)...")
    ml.train()
load_artifacts()


@app.get("/")
def home():
    return send_from_directory("static", "index.html")


@app.get("/api/summary")
def summary():
    return jsonify({k: v for k, v in META.items() if k != "history"})


def filtered_risk():
    df = RISK
    store, status = request.args.get("store"), request.args.get("status")
    if store:
        df = df[df["Store ID"] == store]
    return df, status


@app.get("/api/report.csv")
def report():
    df, status = filtered_risk()
    if status:
        df = df[df["Status"] == status]
    return Response(df.sort_values("Coverage").to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=inventory_risk_report.csv"})


@app.post("/api/retrain")
def retrain():
    """Upload a CSV with the paper's 15-column schema and retrain everything."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "Choose a CSV file first."}), 400
    try:
        df = pd.read_csv(f)
        need = {"Date", "Store ID", "Product ID", "Category", "Region", "Inventory Level", "Units Sold", "Units Ordered",
                "Price", "Discount", "Weather Condition", "Holiday/Promotion", "Competitor Pricing", "Seasonality"}
        miss = need - set(df.columns)
        if miss:
            return jsonify({"error": "Missing columns: " + ", ".join(sorted(miss))}), 400
        if len(df) < 20000:
            return jsonify({"error": "Need at least 20,000 rows for the 9,300-row chronological test split."}), 400
        df.to_csv(ml.DATA, index=False)
        ml.train()
        load_artifacts()
    except Exception as e:
        return jsonify({"error": f"Retraining failed: {e}"}), 400
    return jsonify({"ok": True, "rows": len(df), "best": META["best"]})


@app.get("/api/risk")
def risk():
    df, status = filtered_risk()
    counts = df["Status"].value_counts().to_dict()
    if status:
        df = df[df["Status"] == status]
    rows = df.sort_values("Coverage").head(int(request.args.get("limit", 100)))
    return jsonify({"counts": counts, "total": int(len(df)), "rows": rows.to_dict("records")})


@app.post("/api/predict")
def predict():
    f = request.get_json(force=True)
    try:
        h = META["history"].get(f"{f['store']}|{f['product']}") or {"Lag1": 136, "Lag7": 136, "Roll7": 136}
        row = pd.DataFrame([{
            "Store ID": f["store"], "Product ID": f["product"], "Category": f["category"],
            "Region": f["region"], "Weather Condition": f["weather"], "Seasonality": f["seasonality"],
            "Date": pd.to_datetime(f["date"]), "Inventory Level": float(f["inventory"]),
            "Units Ordered": float(f["ordered"]), "Price": float(f["price"]), "Discount": float(f["discount"]),
            "Competitor Pricing": float(f["competitor"]), "Holiday/Promotion": int(f["promo"]), **h}])
        row = ml.time_features(row)
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    X = row[ml.CAT + ml.NUM]
    preds = {n: max(0.0, float(m.predict(X)[0])) for n, m in MODELS.items()}
    best = preds[META["best"]]
    cov = float(ml.coverage(row["Inventory Level"], best)[0])
    u = META["uncertainty"]
    return jsonify({"predictions": {k: round(v, 1) for k, v in preds.items()}, "best_model": META["best"],
                    "interval": [round(max(0, best + u["q05"]), 1), round(best + u["q95"], 1)],
                    "safety_stock": u["safety_stock"], "reorder_point": int(best + u["safety_stock"]),
                    "coverage": round(cov, 2), "status": ml.status_of(cov),
                    "reorder_qty": ml.reorder_qty(float(f["inventory"]), best)})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
