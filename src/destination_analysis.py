from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg, round

spark = SparkSession.builder \
    .appName("FraudShield_Destination_Analysis") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

df = df.withColumn(
    "destination_balance_change",
    col("newbalanceDest") - col("oldbalanceDest")
)

df = df.groupBy("isFraud").agg(
    count("*").alias("total_transactions"),
    round(avg("oldbalanceDest"), 2).alias("avg_old_dest_balance"),
    round(avg("newbalanceDest"), 2).alias("avg_new_dest_balance"),
    round(avg("destination_balance_change"), 2)
        .alias("avg_destination_balance_change")
)

df.show(truncate=False)

spark.stop()