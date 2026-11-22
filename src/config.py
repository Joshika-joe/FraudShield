"""
FraudShield - central configuration (Member 2: ML, alerts, dashboard).

Every Member 2 script imports this module instead of hard-coding paths,
column names, Spark settings or split boundaries. Member 1's original
scripts are untouched and keep using their own relative paths.

Why this file exists
--------------------
1. The engineered dataset lives outside the git repository (it is 557 MB).
   `outputs/engineered_data` is a symlink to the real folder, and the path
   can also be overridden with the FRAUDSHIELD_ENGINEERED_DATA env var, so
   the same code runs on another machine without editing any script.
2. An explicit Spark schema is faster and safer than `inferSchema=True`,
   which has to read all 6.36 M rows an extra time just to guess types.
3. The chronological split boundaries are defined in ONE place, so the
   training, evaluation, alert and dashboard scripts can never disagree
   about which rows are train, validation or test.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_RESULTS_DIR = OUTPUTS_DIR / "model_results"
FRAUD_ALERTS_DIR = OUTPUTS_DIR / "fraud_alerts"
DASHBOARD_DATA_DIR = OUTPUTS_DIR / "dashboard_data"

# Member 1's Spark CSV output directory (16 part files + _SUCCESS).
ENGINEERED_CSV_DIR = Path(
    os.environ.get("FRAUDSHIELD_ENGINEERED_DATA", OUTPUTS_DIR / "engineered_data")
).expanduser()

# Our Parquet copy of the same data (columnar, typed, much faster to re-read).
ENGINEERED_PARQUET_DIR = Path(
    os.environ.get(
        "FRAUDSHIELD_ENGINEERED_PARQUET", OUTPUTS_DIR / "engineered_parquet"
    )
).expanduser()

LR_MODEL_DIR = MODELS_DIR / "logistic_regression"
RF_MODEL_DIR = MODELS_DIR / "random_forest"


def ensure_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for path in (
        MODELS_DIR,
        MODEL_RESULTS_DIR,
        FRAUD_ALERTS_DIR,
        DASHBOARD_DATA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset schema (13 columns produced by Member 1's feature_engineering.py)
# ---------------------------------------------------------------------------

ENGINEERED_SCHEMA = StructType(
    [
        StructField("step", IntegerType(), True),
        StructField("type", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("oldbalanceOrg", DoubleType(), True),
        StructField("newbalanceOrig", DoubleType(), True),
        StructField("oldbalanceDest", DoubleType(), True),
        StructField("newbalanceDest", DoubleType(), True),
        StructField("isFraud", IntegerType(), True),
        StructField("sender_balance_change", DoubleType(), True),
        StructField("destination_balance_change", DoubleType(), True),
        StructField("log_amount", DoubleType(), True),
        StructField("sender_balance_mismatch", IntegerType(), True),
        StructField("destination_balance_mismatch", IntegerType(), True),
    ]
)

TARGET_COL = "isFraud"

# `type` is the only categorical column; it is one-hot encoded in the pipeline.
CATEGORICAL_COLS = ["type"]

# Numeric model inputs. `step` is deliberately EXCLUDED: it is simulation
# time, used only to build the chronological split. Keeping it as a feature
# would let the model learn "fraud is more common late in the simulation",
# which is an artefact of PaySim and would not generalise to real data.
NUMERIC_COLS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "sender_balance_change",
    "destination_balance_change",
    "log_amount",
]

# The two balance-mismatch flags are kept separate so we can run an ablation
# (train with and without them) as Member 1's handover notes requested.
MISMATCH_COLS = [
    "sender_balance_mismatch",
    "destination_balance_mismatch",
]

# Columns carried through for display in fraud alerts but never fed to a model.
ALERT_DISPLAY_COLS = ["row_id", "step", "type", "amount"]

# ---------------------------------------------------------------------------
# Chronological split (Module 3)
# ---------------------------------------------------------------------------
# Boundaries were chosen by measuring the cumulative row count over `step`
# so that the three windows hold ~70 % / ~15 % / ~15 % of the 6,362,620 rows:
#
#   train      step <=  323   4,463,587 rows   3,643 fraud   (0.0816 %)
#   validation step 324-378     980,416 rows     564 fraud   (0.0575 %)
#   test       step >=  379     918,617 rows   4,006 fraud   (0.4361 %)
#
# Training rows are strictly earlier in time than validation rows, which are
# strictly earlier than test rows, so no future information leaks backwards.
TRAIN_MAX_STEP = 323
VAL_MAX_STEP = 378
MAX_STEP = 743

# ---------------------------------------------------------------------------
# Expected dataset facts (verified against the real files, Phase 1)
# ---------------------------------------------------------------------------
EXPECTED_ROWS = 6_362_620
EXPECTED_FRAUD = 8_213
EXPECTED_LEGIT = 6_354_407
EXPECTED_COLUMNS = 13

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------
# This laptop has 8 GB RAM and 8 cores, so the driver gets 5 GB and the
# shuffle partition count is lowered from Spark's default 200 (which would
# create hundreds of tiny tasks and waste time on scheduling) to 16.


def get_spark(
    app_name: str,
    driver_memory: str | None = None,
    shuffle_partitions: int | None = None,
) -> SparkSession:
    """Create (or reuse) a local SparkSession tuned for this machine."""
    driver_memory = driver_memory or os.environ.get("FRAUDSHIELD_DRIVER_MEMORY", "5g")
    shuffle_partitions = int(
        shuffle_partitions
        or os.environ.get("FRAUDSHIELD_SHUFFLE_PARTITIONS", "16")
    )

    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("FRAUDSHIELD_SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "true")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(os.environ.get("FRAUDSHIELD_SPARK_LOGLEVEL", "WARN"))
    return spark


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def read_engineered(spark: SparkSession, prefer_parquet: bool = True) -> DataFrame:
    """
    Load Member 1's engineered dataset.

    Reads the Parquet copy when it exists (fast, typed) and falls back to the
    original Spark CSV directory otherwise. Both paths give identical data.
    """
    log = logging.getLogger("fraudshield")

    if prefer_parquet and (ENGINEERED_PARQUET_DIR / "_SUCCESS").exists():
        log.info("Reading engineered dataset (Parquet): %s", ENGINEERED_PARQUET_DIR)
        return spark.read.parquet(str(ENGINEERED_PARQUET_DIR))

    if not ENGINEERED_CSV_DIR.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at {ENGINEERED_CSV_DIR}. "
            "Set FRAUDSHIELD_ENGINEERED_DATA to the folder that contains the "
            "part-*.csv files produced by src/feature_engineering.py."
        )

    log.info("Reading engineered dataset (CSV): %s", ENGINEERED_CSV_DIR)
    return spark.read.csv(
        str(ENGINEERED_CSV_DIR),
        header=True,
        schema=ENGINEERED_SCHEMA,
        mode="FAILFAST",
    )


def add_split_column(df: DataFrame, split_col: str = "split") -> DataFrame:
    """Tag every row as train / validation / test using the `step` boundaries."""
    return df.withColumn(
        split_col,
        F.when(F.col("step") <= TRAIN_MAX_STEP, F.lit("train"))
        .when(F.col("step") <= VAL_MAX_STEP, F.lit("validation"))
        .otherwise(F.lit("test")),
    )


def split_chronologically(df: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Return (train, validation, test) DataFrames split by simulation time."""
    train = df.filter(F.col("step") <= TRAIN_MAX_STEP)
    validation = df.filter(
        (F.col("step") > TRAIN_MAX_STEP) & (F.col("step") <= VAL_MAX_STEP)
    )
    test = df.filter(F.col("step") > VAL_MAX_STEP)
    return train, validation, test


def feature_columns(include_mismatch: bool = True) -> list[str]:
    """Numeric feature column names used by the ML pipeline."""
    return list(NUMERIC_COLS) + (list(MISMATCH_COLS) if include_mismatch else [])


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(name: str = "fraudshield", level: int = logging.INFO) -> logging.Logger:
    """Consistent, timestamped console logging for every script."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Silence py4j's very chatty debug logging.
    logging.getLogger("py4j").setLevel(logging.WARNING)
    return logging.getLogger(name)
