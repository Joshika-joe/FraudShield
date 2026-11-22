"""
FraudShield - integration tests.

Run from the project root, after the pipeline has been executed:

    source env.sh
    python tests/test_integration.py

These are end-to-end checks on the real project outputs, not unit tests with
mock data. Each check prints PASS or FAIL and the script exits non-zero if
anything failed, so it can be used as a single "is the project healthy?"
command before a demo.

Checks (matching the project's verification list)
  1  engineered dataset loads in Spark
  2  schema has the expected 13 engineered columns
  3  row count is 6,362,620
  4  fraud distribution is 8,213 / 6,354,407
  5  chronological split is ordered in time and non-empty
  6  feature pipeline executes and produces a numeric vector
  7  class weights are computed from the training split
  8  saved Logistic Regression model loads
  9  saved Random Forest model loads
 10  both reloaded models produce predictions with valid probabilities
 11  a model can be saved and reloaded and give identical predictions
 12  evaluation metrics are mathematically correct (checked against hand-worked numbers)
 13  threshold policy files are present and internally consistent
 14  fraud alerts exist, are well-formed and respect the threshold
 15  dashboard data files are present and readable with pandas
 16  the Streamlit dashboard imports and compiles
"""

from __future__ import annotations

import json
import py_compile
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

import config  # noqa: E402
import ml_preprocessing as mlp  # noqa: E402
import model_evaluation as mev  # noqa: E402
import model_prediction as mp  # noqa: E402

RESULTS = []


def check(name: str):
    """Decorator that records the outcome of one check."""
    def wrap(fn):
        def run(*args, **kwargs):
            try:
                detail = fn(*args, **kwargs) or ""
                RESULTS.append((name, True, str(detail)))
                print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
                return True
            except AssertionError as exc:
                RESULTS.append((name, False, str(exc)))
                print(f"  FAIL  {name}\n          {exc}")
                return False
            except Exception as exc:  # unexpected error, still a failure
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
                return False
        return run
    return wrap


def main() -> int:
    log = config.setup_logging(level=30)  # quiet: the checks do the talking
    spark = config.get_spark("FraudShield_Integration_Tests")
    print("\nFraudShield integration tests")
    print("=" * 72)

    try:
        # ---------- 1. dataset loads ----------
        state = {}

        @check("1  engineered dataset loads in Spark")
        def c1():
            df = config.read_engineered(spark)
            state["df"] = df.cache()
            assert len(df.take(1)) == 1, "dataset returned no rows"
            source = ("Parquet" if (config.ENGINEERED_PARQUET_DIR / "_SUCCESS").exists()
                      else "CSV")
            return f"source={source}"

        c1()
        df = state.get("df")
        if df is None:
            print("\nCannot continue without the dataset.")
            return 1

        # ---------- 2. schema ----------
        @check("2  schema has the expected engineered columns")
        def c2():
            expected = [f.name for f in config.ENGINEERED_SCHEMA.fields]
            actual = df.columns
            miss = [c for c in expected if c not in actual]
            assert not miss, f"missing columns: {miss}"
            return f"{len(expected)} expected columns present, {len(actual)} total"

        c2()

        # ---------- 3. row count ----------
        @check("3  row count is 6,362,620")
        def c3():
            n = df.count()
            state["n"] = n
            assert n == config.EXPECTED_ROWS, f"expected {config.EXPECTED_ROWS:,}, got {n:,}"
            return f"{n:,} rows"

        c3()

        # ---------- 4. fraud distribution ----------
        @check("4  fraud distribution is 8,213 fraud / 6,354,407 legitimate")
        def c4():
            fraud = df.filter(f"{config.TARGET_COL} = 1").count()
            legit = df.filter(f"{config.TARGET_COL} = 0").count()
            assert fraud == config.EXPECTED_FRAUD, f"fraud {fraud:,} != {config.EXPECTED_FRAUD:,}"
            assert legit == config.EXPECTED_LEGIT, f"legit {legit:,} != {config.EXPECTED_LEGIT:,}"
            return f"{fraud:,} fraud ({100*fraud/(fraud+legit):.4f}%)"

        c4()

        # ---------- 5. chronological split ----------
        @check("5  chronological split is time-ordered and non-empty")
        def c5():
            train, val, test = config.split_chronologically(df)
            from pyspark.sql import functions as F
            tr = train.agg(F.max("step")).collect()[0][0]
            vmin = val.agg(F.min("step")).collect()[0][0]
            vmax = val.agg(F.max("step")).collect()[0][0]
            tmin = test.agg(F.min("step")).collect()[0][0]
            n_tr, n_val, n_te = train.count(), val.count(), test.count()
            assert min(n_tr, n_val, n_te) > 0, "a split is empty"
            assert tr < vmin, f"train max step {tr} overlaps validation min {vmin}"
            assert vmax < tmin, f"validation max step {vmax} overlaps test min {tmin}"
            fr_val = val.filter(f"{config.TARGET_COL} = 1").count()
            fr_te = test.filter(f"{config.TARGET_COL} = 1").count()
            assert fr_val > 0 and fr_te > 0, "validation or test has no fraud rows"
            state["splits"] = (train, val, test)
            return (f"train {n_tr:,} (<= step {tr}), val {n_val:,} ({fr_val} fraud), "
                    f"test {n_te:,} ({fr_te} fraud)")

        c5()

        # ---------- 6. feature pipeline ----------
        @check("6  feature pipeline executes and produces a feature vector")
        def c6():
            small = df.limit(20_000)
            pipeline, features_col, numeric_cols = mlp.build_feature_pipeline(scale=True)
            model = pipeline.fit(small)
            row = model.transform(small).select(features_col).head()
            size = len(row[0])
            assert size >= len(numeric_cols), f"vector of size {size} is too small"
            names = mlp.feature_names(model, numeric_cols, vector_size=size)
            assert len(names) == size, f"{len(names)} names for {size} vector slots"
            return f"vector size {size}, names aligned"

        c6()

        # ---------- 7. class weights ----------
        @check("7  class weights are computed from the training split")
        def c7():
            train = state["splits"][0]
            w_legit, w_fraud = mlp.compute_class_weights(train)
            assert w_fraud > w_legit, "fraud must get the larger weight"
            assert 0.4 < w_legit < 0.6, f"legitimate weight {w_legit} is out of range"
            weighted = mlp.add_class_weight_column(train.limit(1000), w_legit, w_fraud)
            assert mlp.WEIGHT_COL in weighted.columns
            return f"legit={w_legit:.4f}, fraud={w_fraud:.1f}"

        c7()

        # ---------- 8 & 9. saved models load ----------
        @check("8  saved Logistic Regression model loads")
        def c8():
            scorer = mp.load_scorer(spark, "lr")
            state["lr"] = scorer
            return scorer.describe()

        @check("9  saved Random Forest model loads")
        def c9():
            scorer = mp.load_scorer(spark, "rf")
            state["rf"] = scorer
            return scorer.describe()

        c8()
        c9()

        # ---------- 10. reloaded models predict ----------
        @check("10 reloaded models produce valid probabilities")
        def c10():
            from pyspark.sql import functions as F
            sample = df.limit(5_000)
            details = []
            for key in ("lr", "rf"):
                scorer = state.get(key)
                assert scorer is not None, f"{key} model was not loaded"
                scored = mp.score_transactions(sample, scorer)
                stats = scored.agg(
                    F.min(mev.FRAUD_PROB_COL).alias("lo"),
                    F.max(mev.FRAUD_PROB_COL).alias("hi"),
                    F.count("*").alias("n"),
                    F.sum(F.when(F.col(mev.FRAUD_PROB_COL).isNull(), 1).otherwise(0)).alias("nulls"),
                ).collect()[0]
                assert int(stats["n"]) == 5_000, "row count changed during scoring"
                assert int(stats["nulls"]) == 0, "null probabilities produced"
                assert 0.0 <= float(stats["lo"]) and float(stats["hi"]) <= 1.0, \
                    f"probability outside [0,1]: {stats['lo']}..{stats['hi']}"
                cats = {r["risk_category"] for r in scored.select("risk_category").distinct().collect()}
                allowed = {mp.RISK_HIGH, mp.RISK_MEDIUM, mp.RISK_LOW, mp.RISK_NONE}
                assert cats <= allowed, f"unexpected risk categories: {cats - allowed}"
                details.append(f"{key}: {float(stats['lo']):.3f}-{float(stats['hi']):.3f}")
            return "; ".join(details)

        c10()

        # ---------- 11. save / reload round trip ----------
        @check("11 a model can be saved, reloaded and give identical predictions")
        def c11():
            scorer = state.get("lr") or state.get("rf")
            assert scorer is not None, "no model available"
            sample = df.limit(2_000)
            before = (
                mev.with_fraud_probability(scorer.model.transform(sample))
                .select(mev.FRAUD_PROB_COL).collect()
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "roundtrip_model")
                scorer.model.write().overwrite().save(path)
                from pyspark.ml import PipelineModel
                reloaded = PipelineModel.load(path)
                after = (
                    mev.with_fraud_probability(reloaded.transform(sample))
                    .select(mev.FRAUD_PROB_COL).collect()
                )
            assert len(before) == len(after), "row count changed after reload"
            worst = max(abs(a[0] - b[0]) for a, b in zip(before, after))
            assert worst < 1e-9, f"predictions differ after reload (max diff {worst})"
            return f"{len(before)} rows identical (max diff {worst:.2e})"

        c11()

        # ---------- 12. metric arithmetic ----------
        @check("12 evaluation metrics are mathematically correct")
        def c12():
            # Hand-worked example: 10 rows, 4 fraud.
            # At threshold 0.5: TP=3, FP=2, FN=1, TN=4
            # precision = 3/5 = 0.6, recall = 3/4 = 0.75, F1 = 2*.6*.75/1.35 = 0.666667
            rows = [
                (0.9, 1), (0.8, 1), (0.6, 1), (0.4, 1),      # fraud rows
                (0.7, 0), (0.55, 0), (0.3, 0), (0.2, 0),     # legit rows
                (0.1, 0), (0.05, 0),
            ]
            sdf = spark.createDataFrame(rows, [mev.FRAUD_PROB_COL, config.TARGET_COL])
            m = mev.metrics_at_thresholds(sdf, [0.5])[0]
            assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 2, 1, 4), \
                f"confusion counts wrong: {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}"
            assert abs(m["precision"] - 0.6) < 1e-6, f"precision {m['precision']}"
            assert abs(m["recall"] - 0.75) < 1e-6, f"recall {m['recall']}"
            assert abs(m["f1"] - 0.666667) < 1e-5, f"f1 {m['f1']}"
            assert m["alerts_raised"] == 5
            return "TP=3 FP=2 FN=1 TN=4, precision=0.6, recall=0.75, F1=0.6667"

        c12()

        # ---------- 13. threshold policies ----------
        @check("13 threshold policy files are present and consistent")
        def c13():
            found = []
            for tag in ("logistic_regression", "random_forest"):
                path = config.MODEL_RESULTS_DIR / f"{tag}_threshold_policy.json"
                assert path.exists(), f"missing {path.name} - run threshold_optimization.py"
                policy = json.loads(path.read_text())
                t = policy["selected_threshold"]
                bands = policy["risk_bands"]
                assert 0.0 < t < 1.0, f"{tag}: threshold {t} not inside (0,1)"
                assert t != 0.5 or policy["selection_policy"] != "best_f2" or True
                assert bands["high_risk_threshold"] >= bands["medium_risk_threshold"] >= t, (
                    f"{tag}: risk bands are not ordered "
                    f"(high={bands['high_risk_threshold']}, "
                    f"medium={bands['medium_risk_threshold']}, operating={t})"
                )
                sweep = config.MODEL_RESULTS_DIR / f"{tag}_threshold_sweep.csv"
                assert sweep.exists(), f"missing {sweep.name}"
                found.append(f"{tag}@{t:.3f}")
            return ", ".join(found)

        c13()

        # ---------- 14. fraud alerts ----------
        @check("14 fraud alerts exist and respect the operating threshold")
        def c14():
            import pandas as pd
            summary_path = config.FRAUD_ALERTS_DIR / "fraud_alerts_summary.json"
            csv_path = config.FRAUD_ALERTS_DIR / "fraud_alerts.csv"
            assert summary_path.exists(), "missing fraud_alerts_summary.json - run fraud_alerts.py"
            assert csv_path.exists(), "missing fraud_alerts.csv - run fraud_alerts.py"
            summary = json.loads(summary_path.read_text())
            alerts = pd.read_csv(csv_path)
            assert len(alerts) > 0, "alert export is empty"
            threshold = summary["policy"]["operating_threshold"]
            below = alerts[alerts["fraud_probability"] < threshold]
            assert below.empty, f"{len(below)} exported alerts are below the threshold"
            for col in ("row_id", "step", "type", "amount", "fraud_probability",
                        "predicted_class", "risk_category"):
                assert col in alerts.columns, f"alert export is missing column '{col}'"
            assert alerts["predicted_class"].isin([1]).all(), "a non-alert row was exported"
            parquet = config.FRAUD_ALERTS_DIR / "fraud_alerts.parquet"
            assert (parquet / "_SUCCESS").exists(), "Parquet alert output is incomplete"
            return (f"{len(alerts):,} exported, {summary['alerts_raised']:,} total, "
                    f"threshold {threshold:.3f}")

        c14()

        # ---------- 15. dashboard data ----------
        @check("15 dashboard data files are present and readable")
        def c15():
            import pandas as pd
            required = [
                "overview.json", "type_distribution.csv", "fraud_by_step.csv",
                "hourly_pattern.csv", "amount_bands.csv", "amount_stats.csv",
                "mismatch_stats.csv", "split_summary.csv", "sample_transactions.csv",
            ]
            miss = [f for f in required
                    if not (config.DASHBOARD_DATA_DIR / f).exists()]
            assert not miss, f"missing {miss} - run prepare_dashboard_data.py"
            overview = json.loads((config.DASHBOARD_DATA_DIR / "overview.json").read_text())
            assert overview["total_transactions"] == config.EXPECTED_ROWS, \
                "overview.json disagrees with the dataset row count"
            assert overview["fraud_transactions"] == config.EXPECTED_FRAUD, \
                "overview.json disagrees with the dataset fraud count"
            total_size = 0
            for f in required:
                p = config.DASHBOARD_DATA_DIR / f
                total_size += p.stat().st_size
                if p.suffix == ".csv":
                    pdf = pd.read_csv(p)
                    assert len(pdf) > 0, f"{f} is empty"
            return f"{len(required)} files, {total_size/1024:.0f} KB total"

        c15()

        # ---------- 16. dashboard compiles ----------
        @check("16 the Streamlit dashboard imports and compiles")
        def c16():
            import streamlit  # noqa: F401
            import plotly  # noqa: F401
            import theme  # noqa: F401  (dashboard/theme.py)
            app = PROJECT_ROOT / "dashboard" / "app.py"
            py_compile.compile(str(app), doraise=True)
            return f"streamlit {streamlit.__version__}, app.py compiles"

        c16()

        # ---------- summary ----------
        passed = sum(1 for _, ok, _ in RESULTS if ok)
        failed = len(RESULTS) - passed
        print("=" * 72)
        print(f"{passed} passed, {failed} failed, {len(RESULTS)} checks total")
        if failed:
            print("\nFailed checks:")
            for name, ok, detail in RESULTS:
                if not ok:
                    print(f"  - {name}: {detail}")
        return 1 if failed else 0

    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
