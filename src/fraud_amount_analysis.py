from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, max, sum, round
)

spark = SparkSession.builder \
    .appName("FraudShield_Amount_Analysis") \
    .master("local[*]") \
    .getOrCreate()

# Load dataset
df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

# Filter fraudulent transactions
fraud_df = df.filter(col("isFraud") == 1)

# Analyze by transaction type
result = fraud_df.groupBy("type").agg(

    count("*").alias("fraud_count"),

    round(avg("amount"), 2).alias("average_amount"),

    round(max("amount"), 2).alias("maximum_amount"),

    round(sum("amount"), 2).alias("total_amount")
)

result.orderBy(
    col("total_amount").desc()
).show(truncate=False)

spark.stop()