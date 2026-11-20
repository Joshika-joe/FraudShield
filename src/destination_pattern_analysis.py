from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, round, abs, avg

spark = SparkSession.builder \
    .appName("FraudShield_Destination_Patterns") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

# Calculate destination balance difference
df = df.withColumn(
    "destination_difference",
    col("oldbalanceDest") + col("amount") - col("newbalanceDest")
)

df = df.withColumn(
    "destination_mismatch",
    when(abs(col("destination_difference")) > 0.01, 1).otherwise(0)
)

# Focus only on TRANSFER and CASH_OUT
df = df.filter(col("type").isin("TRANSFER", "CASH_OUT"))

# Analyze mismatch patterns
result = df.groupBy("type", "isFraud").agg(
    count("*").alias("total_transactions"),
    count(when(col("destination_mismatch") == 1, True))
        .alias("mismatched_transactions"),
    round(avg("amount"), 2).alias("avg_transaction_amount")
)

result.show(truncate=False)

spark.stop()