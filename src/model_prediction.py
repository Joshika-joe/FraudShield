"""
FraudShield - Module 8a: load a saved model and score transactions.

This module is the single place where a trained pipeline is loaded and applied.
It is imported by src/fraud_alerts.py, by the integration tests, and can be
used from any other script.

    from model_prediction import load_scorer, score_transactions

    scorer = load_scorer(spark, "final")          # or "lr" / "rf"
    alerts = score_transactions(df, scorer)

What gets loaded
----------------
A single Spark MLlib `PipelineModel` that contains the whole chain:
StringIndexer -> OneHotEncoder -> VectorAssembler -> (StandardScaler) ->
classifier. Because everything is inside one artefact, scoring new
transactions needs no feature code at all - the saved pipeline performs the
exact same transformations that were used during training, which removes any
chance of a train/serve mismatch.

Alongside the model, the threshold policy produced by
src/threshold_optimization.py is loaded, so the operating threshold and the
risk bands come from the measured validation results rather than being
hard-coded here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

import config
import model_evaluation as mev

LOG = logging.getLogger("fraudshield.prediction")

RISK_HIGH = "High Risk"
RISK_MEDIUM = "Medium Risk"
RISK_LOW = "Low Risk"
RISK_NONE = "No Alert"

_MODEL_PATHS = {
    "lr": (config.LR_MODEL_DIR, "logistic_regression", "LogisticRegression"),
    "rf": (config.RF_MODEL_DIR, "random_forest", "RandomForestClassifier"),
}


@dataclass
class Scorer:
    """A loaded model plus the decision policy that belongs with it."""

    key: str
    name: str
    model: PipelineModel
    model_path: Path
    threshold: float
    high_risk_threshold: float
    medium_risk_threshold: float
    policy: dict

    def describe(self) -> str:
        return (
            f"{self.name} (threshold={self.threshold:.3f}, "
            f"High>={self.high_risk_threshold:.3f}, "
            f"Medium>={self.medium_risk_threshold:.3f})"
        )


def _resolve_key(key: str) -> str:
    """Turn 'final' into the model key chosen on the validation split."""
    if key != "final":
        return key
    selection_path = config.MODEL_RESULTS_DIR / "final_model_selection.json"
    if selection_path.exists():
        chosen = json.loads(selection_path.read_text()).get("selected_model_key")
        if chosen in _MODEL_PATHS:
            LOG.info("Final model from validation-based selection: %s", chosen)
            return chosen
    # Fall back to whichever model is on disk.
    for candidate in ("rf", "lr"):
        if (_MODEL_PATHS[candidate][0] / "metadata").exists():
            LOG.warning(
                "final_model_selection.json not found; falling back to '%s'", candidate
            )
            return candidate
    raise FileNotFoundError(
        "No trained model found. Run src/train_logistic_regression.py and/or "
        "src/train_random_forest.py first."
    )


def load_scorer(spark: SparkSession, key: str = "final") -> Scorer:
    """Load a saved pipeline and its threshold policy."""
    key = _resolve_key(key)
    model_dir, tag, name = _MODEL_PATHS[key]

    if not (model_dir / "metadata").exists():
        raise FileNotFoundError(
            f"No saved model at {model_dir}. Train it before generating alerts."
        )

    LOG.info("Loading %s pipeline from %s", name, model_dir)
    model = PipelineModel.load(str(model_dir))

    policy_path = config.MODEL_RESULTS_DIR / f"{tag}_threshold_policy.json"
    if policy_path.exists():
        policy = json.loads(policy_path.read_text())
        threshold = float(policy["selected_threshold"])
        bands = policy["risk_bands"]
        high = float(bands["high_risk_threshold"])
        medium = float(bands["medium_risk_threshold"])
    else:
        # Without a tuned policy we must not silently invent one.
        LOG.warning(
            "No threshold policy at %s. Falling back to 0.5 for all bands, which "
            "is NOT the project's chosen operating point - run "
            "src/threshold_optimization.py.",
            policy_path,
        )
        policy, threshold, high, medium = {}, 0.5, 0.9, 0.7

    return Scorer(
        key=key, name=name, model=model, model_path=model_dir,
        threshold=threshold, high_risk_threshold=high,
        medium_risk_threshold=medium, policy=policy,
    )


def score_transactions(
    df: DataFrame,
    scorer: Scorer,
    keep_label: bool = True,
) -> DataFrame:
    """
    Score transactions and attach the fraud probability, the predicted class
    and the risk category.

    The returned DataFrame keeps only display columns and the scoring results.
    `row_id` is carried through purely so an analyst can point at a specific
    transaction; it is never a model input.
    """
    if "row_id" not in df.columns:
        LOG.info("Input has no row_id column; generating one for display only")
        df = df.withColumn("row_id", F.monotonically_increasing_id())

    scored = mev.with_fraud_probability(scorer.model.transform(df))

    prob = F.col(mev.FRAUD_PROB_COL)
    scored = scored.withColumn(
        "predicted_class",
        F.when(prob >= F.lit(scorer.threshold), F.lit(1)).otherwise(F.lit(0)),
    ).withColumn(
        "risk_category",
        F.when(prob >= F.lit(scorer.high_risk_threshold), F.lit(RISK_HIGH))
        .when(prob >= F.lit(scorer.medium_risk_threshold), F.lit(RISK_MEDIUM))
        .when(prob >= F.lit(scorer.threshold), F.lit(RISK_LOW))
        .otherwise(F.lit(RISK_NONE)),
    )

    display_cols = ["row_id", "step", "type", "amount"]
    extra = [c for c in ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
                         "newbalanceDest") if c in scored.columns]
    result_cols = [mev.FRAUD_PROB_COL, "predicted_class", "risk_category"]
    label = [config.TARGET_COL] if keep_label and config.TARGET_COL in scored.columns else []

    return scored.select(*(display_cols + extra + result_cols + label))


def alerts_only(scored: DataFrame) -> DataFrame:
    """Keep just the transactions that crossed the operating threshold."""
    return scored.filter(F.col("predicted_class") == 1)
