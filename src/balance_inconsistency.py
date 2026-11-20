from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, sum

spark = SparkSession.builder \
    .appName("FraudShield_Balance_Check") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

df = df.withColumn(
    "balance_difference",
    col("oldbalanceOrg") - col("amount") - col("newbalanceOrig")
)

df = df.withColumn(
    "balance_mismatch",
    when(col("balance_difference") != 0, 1).otherwise(0)
)

df.groupBy("isFraud").agg(
    count("*").alias("total_transactions"),
    sum("balance_mismatch").alias("mismatched_transactions")
).show()

spark.stop()