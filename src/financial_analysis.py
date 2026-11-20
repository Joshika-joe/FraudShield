from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, max, sum, when, round
)

spark = SparkSession.builder \
    .appName("FraudShield_Financial_Analysis") \
    .master("local[*]") \
    .getOrCreate()

# Load dataset
df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

# Calculate financial statistics
analysis = df.agg(
    round(
        avg(when(col("isFraud") == 0, col("amount"))), 2
    ).alias("avg_legitimate_amount"),

    round(
        avg(when(col("isFraud") == 1, col("amount"))), 2
    ).alias("avg_fraud_amount"),

    round(
        max(when(col("isFraud") == 1, col("amount"))), 2
    ).alias("max_fraud_amount"),

    round(
        sum(when(col("isFraud") == 1, col("amount"))), 2
    ).alias("total_fraud_amount")
)

analysis.show(truncate=False)

spark.stop()