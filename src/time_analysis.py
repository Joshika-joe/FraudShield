from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum, round
)

spark = SparkSession.builder \
    .appName("FraudShield_Time_Analysis") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

# Group by time step
time_analysis = df.groupBy("step").agg(

    count("*").alias("total_transactions"),

    sum("isFraud").alias("fraud_transactions")
)

# Calculate fraud rate
time_analysis = time_analysis.withColumn(
    "fraud_rate",
    round(
        col("fraud_transactions") /
        col("total_transactions") * 100,
        4
    )
)

# Filter steps with at least 100 transactions
filtered_analysis = time_analysis.filter(
    col("total_transactions") >= 100
)

# Display highest fraud rates
filtered_analysis.orderBy(
    col("fraud_rate").desc()
).show(15, truncate=False)

spark.stop()