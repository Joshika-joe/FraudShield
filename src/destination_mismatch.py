from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, round, abs

spark = SparkSession.builder \
    .appName("FraudShield_Destination_Mismatch") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

df = df.withColumn(
    "destination_difference",
    col("oldbalanceDest") + col("amount") - col("newbalanceDest")
)

df = df.withColumn(
    "destination_mismatch",
    when(abs(col("destination_difference")) > 0.01, 1).otherwise(0)
)

result = df.groupBy("type", "isFraud").agg(
    count("*").alias("total_transactions"),
    count(when(col("destination_mismatch") == 1, True))
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