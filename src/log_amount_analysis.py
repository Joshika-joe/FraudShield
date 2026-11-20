from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p, avg, expr, round

spark = SparkSession.builder \
    .appName("FraudShield_Log_Amount") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

df = df.withColumn(
    "log_amount",
    log1p(col("amount"))
)

result = df.groupBy("isFraud").agg(
    round(avg("amount"), 2).alias("average_amount"),
    round(avg("log_amount"), 4).alias("average_log_amount"),
    expr("percentile_approx(log_amount, 0.50)")
        .alias("median_log_amount")
)

result.show(truncate=False)

spark.stop()