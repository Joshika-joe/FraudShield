from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max, avg, expr

spark = SparkSession.builder \
    .appName("FraudShield_Amount_Distribution") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

df.groupBy("isFraud").agg(
    min("amount").alias("minimum_amount"),
    expr("percentile_approx(amount, 0.25)").alias("Q1"),
    expr("percentile_approx(amount, 0.50)").alias("median_amount"),
    expr("percentile_approx(amount, 0.75)").alias("Q3"),
    max("amount").alias("maximum_amount"),
    avg("amount").alias("average_amount")
).show(truncate=False)

spark.stop()