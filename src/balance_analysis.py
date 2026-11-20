from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, round
)

spark = SparkSession.builder \
    .appName("FraudShield_Balance_Analysis") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

# Calculate sender balance change
df = df.withColumn(
    "sender_balance_change",
    col("oldbalanceOrg") - col("newbalanceOrig")
)

# Compare legitimate and fraudulent transactions
result = df.groupBy("isFraud").agg(

    count("*").alias("total_transactions"),

    round(
        avg("oldbalanceOrg"), 2
    ).alias("avg_old_balance"),

    round(
        avg("newbalanceOrig"), 2
    ).alias("avg_new_balance"),

    round(
        avg("sender_balance_change"), 2
    ).alias("avg_balance_change")
)

result.show(truncate=False)

spark.stop()