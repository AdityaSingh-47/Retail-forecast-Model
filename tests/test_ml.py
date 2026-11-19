import numpy as np, ml

def test_metrics_perfect():
    m = ml.metrics([10, 20, 30], [10, 20, 30])
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["R2"] == 1

def test_wape():
    assert abs(ml.metrics([100, 100], [90, 110])["WAPE"] - 10) < 1e-6

def test_coverage_status():
    assert ml.status_of(ml.coverage(50, 100)) == "Stockout risk"
    assert ml.status_of(1.2) == "Low coverage"
    assert ml.status_of(2.0) == "Adequate"
    assert ml.status_of(4.0) == "Excess inventory"

def test_reorder():
    assert ml.reorder_qty(100, 100) == 100 and ml.reorder_qty(500, 100) == 0
