"""
FraudShield - Module 2: data preparation for machine learning.

This module is imported by the training, evaluation and prediction scripts.
It is not meant to be run on its own (though running it prints a short
self-test on a sample, which is useful while debugging).

What it builds
--------------
A Spark MLlib `Pipeline` that turns one transaction row into one numeric
feature vector:

    type (string)  --StringIndexer-->  type_index  --OneHotEncoder-->  type_ohe
    8 numeric columns  ----------------------------------------------\
    2 mismatch flags (optional) -------------------------------------->  VectorAssembler  -->  features
                                                                      /
    type_ohe  -------------------------------------------------------/

    features --StandardScaler(optional, for Logistic Regression only)--> scaled_features

Feature decisions and the reasoning behind them
-----------------------------------------------
* Target: `isFraud`. It is never placed in the feature vector.
* `isFlaggedFraud` was already removed by Member 1's preprocessing. It is a
  label produced by the bank's own rule system, so using it would be target
  leakage, not prediction.
* `nameOrig` / `nameDest` were also removed. Account identifiers are unique
  strings; a model can memorise them but cannot generalise to new accounts.
* `step` is EXCLUDED from the features even though it is in the dataset. It is
  simulation time and is used only to build the chronological split. In PaySim
  the proportion of fraud rises in later steps because the simulator generates
  fewer legitimate transactions late in the run, so a model given `step` would
  learn that artefact instead of learning fraud behaviour.
* `row_id` (added during Parquet conversion) is a display identifier only.
* The two balance-mismatch flags are optional (`include_mismatch`) so we can
  run the with/without comparison Member 1's handover notes asked for. They
  are flagged in 79.8 % and 65.8 % of all rows, and for CASH_IN transactions
  the sender-mismatch formula is true by construction, so part of what they
  encode is simply the transaction type.
* Scaling is applied for Logistic Regression only. Raw amounts reach ten
  million while `log_amount` stays near 16, and a linear model with
  regularisation is sensitive to that difference in scale. Random Forest
  splits each feature independently, so scaling would add cost and change
  nothing.

Class imbalance (Module 4)
--------------------------
Only 0.1291 % of transactions are fraud. Rather than resampling the data,
each row is given a weight that makes the two classes count equally during
training (`balanced` class weighting). Spark MLlib's LogisticRegression and
RandomForestClassifier both accept a `weightCol`, so no data is duplicated or
thrown away and nothing is resampled before the split.
"""

from __future__ import annotations

import logging

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import config

LOG = logging.getLogger("fraudshield.ml_preprocessing")

TYPE_INDEX_COL = "type_index"
TYPE_OHE_COL = "type_ohe"
ASSEMBLED_COL = "features_raw"
SCALED_COL = "features"
WEIGHT_COL = "class_weight"


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------


def build_feature_pipeline(
    include_mismatch: bool = True,
    scale: bool = False,
) -> tuple[Pipeline, str, list[str]]:
    """
    Build the MLlib feature pipeline.

    Parameters
    ----------
    include_mismatch:
        Include `sender_balance_mismatch` and `destination_balance_mismatch`.
    scale:
        Apply StandardScaler. Use True for Logistic Regression, False for
        Random Forest.

    Returns
    -------
    (pipeline, features_col_name, numeric_input_columns)
    """
    numeric_cols = config.feature_columns(include_mismatch=include_mismatch)

    # handleInvalid="keep" puts any transaction type not seen during training
    # into an extra bucket instead of crashing at prediction time.
    indexer = StringIndexer(
        inputCol="type",
        outputCol=TYPE_INDEX_COL,
        handleInvalid="keep",
    )

    # dropLast=False keeps one column per transaction type. With five types the
    # vector stays tiny, and tree models handle the redundant column fine.
    encoder = OneHotEncoder(
        inputCols=[TYPE_INDEX_COL],
        outputCols=[TYPE_OHE_COL],
        dropLast=False,
        handleInvalid="keep",
    )

    assembler = VectorAssembler(
        inputCols=numeric_cols + [TYPE_OHE_COL],
        outputCol=ASSEMBLED_COL,
        handleInvalid="error",
    )

    stages: list = [indexer, encoder, assembler]
    features_col = ASSEMBLED_COL

    if scale:
        # withMean=False keeps the vector sparse (the one-hot part is sparse);
        # centring would densify it and use far more memory on 6.36 M rows.
        stages.append(
            StandardScaler(
                inputCol=ASSEMBLED_COL,
                outputCol=SCALED_COL,
                withMean=False,
                withStd=True,
            )
        )
        features_col = SCALED_COL

    LOG.info(
        "Feature pipeline: %d numeric columns + one-hot type (scale=%s) -> '%s'",
        len(numeric_cols),
        scale,
        features_col,
    )
    return Pipeline(stages=stages), features_col, numeric_cols


def feature_names(
    pipeline_model: PipelineModel,
    numeric_cols: list[str],
    vector_size: int | None = None,
) -> list[str]:
    """
    Human-readable name for every position in the feature vector, so model
    coefficients and feature importances can be reported per feature.

    `vector_size` should be the real width of the feature vector (for example
    `len(model.coefficients)`). It is used to keep the names aligned with the
    vector: because both StringIndexer and OneHotEncoder are configured with
    handleInvalid="keep", each of them reserves an extra slot for values not
    seen during training, so the one-hot block can be a little wider than the
    number of transaction types. Padding here guarantees that a coefficient is
    never reported under the wrong feature name.
    """
    indexer_model = pipeline_model.stages[0]
    type_labels = list(indexer_model.labels)
    names = list(numeric_cols) + [f"type={label}" for label in type_labels]
    names.append("type=__unseen__")

    if vector_size is not None:
        while len(names) < vector_size:
            names.append(f"type=__reserved_{len(names)}__")
        names = names[:vector_size]
    return names


# ---------------------------------------------------------------------------
# Class imbalance handling (Module 4)
# ---------------------------------------------------------------------------


def compute_class_weights(train_df: DataFrame) -> tuple[float, float]:
    """
    'Balanced' class weights computed from the TRAINING split only.

        weight(class) = total_rows / (2 * rows_in_that_class)

    The rare fraud class therefore gets a large weight and the common
    legitimate class a weight close to 0.5, so the two classes contribute
    equally to the loss. Using only the training split keeps validation and
    test information out of the model.
    """
    counts = (
        train_df.groupBy(config.TARGET_COL).count().collect()
    )
    by_label = {int(r[config.TARGET_COL]): int(r["count"]) for r in counts}
    n_legit = by_label.get(0, 0)
    n_fraud = by_label.get(1, 0)
    if n_fraud == 0 or n_legit == 0:
        raise ValueError(
            f"Training split must contain both classes (got legit={n_legit}, fraud={n_fraud})"
        )
    total = n_legit + n_fraud
    w_legit = total / (2.0 * n_legit)
    w_fraud = total / (2.0 * n_fraud)
    LOG.info(
        "Class weights from training split: legitimate=%.6f (n=%s), fraud=%.4f (n=%s)",
        w_legit, f"{n_legit:,}", w_fraud, f"{n_fraud:,}",
    )
    return w_legit, w_fraud


def add_class_weight_column(
    df: DataFrame, w_legit: float, w_fraud: float, col_name: str = WEIGHT_COL
) -> DataFrame:
    """Attach the per-row training weight."""
    return df.withColumn(
        col_name,
        F.when(F.col(config.TARGET_COL) == 1, F.lit(w_fraud)).otherwise(F.lit(w_legit)),
    )


# ---------------------------------------------------------------------------
# Convenience loader used by the training scripts
# ---------------------------------------------------------------------------


def load_splits(spark, sample_fraction: float | None = None):
    """
    Read the engineered dataset and return (train, validation, test).

    `sample_fraction` takes a reproducible random sample of the WHOLE dataset
    before splitting. It exists only for fast debugging on a laptop; the split
    boundaries and all logic stay identical, so a sample run and a full run
    differ only in how many rows they see.
    """
    df = config.read_engineered(spark)
    if sample_fraction:
        LOG.warning(
            "Using a %.1f%% SAMPLE of the dataset (debug mode, not full-scale results)",
            100 * sample_fraction,
        )
        df = df.sample(withReplacement=False, fraction=sample_fraction, seed=config.RANDOM_SEED)
    return config.split_chronologically(df)


if __name__ == "__main__":
    # Small self-test: build the pipeline on a 0.2 % sample and show the vector.
    log = config.setup_logging()
    spark = config.get_spark("FraudShield_MLPreprocessing_SelfTest")
    try:
        train, val, test = load_splits(spark, sample_fraction=0.002)
        pipeline, features_col, numeric_cols = build_feature_pipeline(scale=True)
        model = pipeline.fit(train)
        out = model.transform(train)
        log.info("Feature column: %s", features_col)
        vec_size = len(out.select(features_col).head()[0])
        log.info("Feature vector size: %d", vec_size)
        log.info("Feature names : %s", feature_names(model, numeric_cols, vec_size))
        out.select("type", "amount", features_col, config.TARGET_COL).show(5, truncate=False)
        w_legit, w_fraud = compute_class_weights(train)
        add_class_weight_column(train, w_legit, w_fraud).select(
            config.TARGET_COL, WEIGHT_COL
        ).show(5)
        log.info("Self-test finished: train=%s val=%s test=%s",
                 f"{train.count():,}", f"{val.count():,}", f"{test.count():,}")
    finally:
        spark.stop()
