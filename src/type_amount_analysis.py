from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, expr, max, min, round

spark = SparkSession.builder \
    .appName("FraudShield_Type_Amount") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

result = df.groupBy("type", "isFraud").agg(
    min("amount").alias("minimum_amount"),
    expr("percentile_approx(amount, 0.50)").alias("median_amount"),
    round(avg("amount"), 2).alias("average_amount"),
    max("amount").alias("maximum_amount")
)

result.orderBy("type", "isFraud").show(truncate=False)

spark.stop()