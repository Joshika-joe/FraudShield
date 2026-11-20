from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, round, abs

spark = SparkSession.builder \
    .appName("FraudShield_Mismatch_Analysis") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

# Calculate sender balance difference
df = df.withColumn(
    "balance_difference",
    col("oldbalanceOrg") - col("amount") - col("newbalanceOrig")
)

# Identify balance mismatches
df = df.withColumn(
    "balance_mismatch",
    when(abs(col("balance_difference")) > 0.01, 1).otherwise(0)
)

# Analyze mismatches by transaction type and fraud status
result = df.groupBy("type", "isFraud").agg(
    count("*").alias("total_transactions"),
    count(when(col("balance_mismatch") == 1, True))
        .alias("mismatched_transactions")
)

result = result.withColumn(
    "mismatch_percentage",
    round(
        col("mismatched_transactions") /
        col("total_transactions") * 100, 2
    )
)

result.orderBy("type", "isFraud").show(truncate=False)

spark.stop()