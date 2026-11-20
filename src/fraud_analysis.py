from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, round

spark = SparkSession.builder \
    .appName("FraudShield_Fraud_Analysis") \
    .master("local[*]") \
    .getOrCreate()

# Load dataset
df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

# Analyze fraud by transaction type
fraud_analysis = df.groupBy("type").agg(
    count("*").alias("total_transactions"),
    sum("isFraud").alias("fraud_transactions")
)

# Calculate fraud percentage
fraud_analysis = fraud_analysis.withColumn(
    "fraud_rate",
    round(
        col("fraud_transactions") /
        col("total_transactions") * 100,
        4
    )
)

# Display results
fraud_analysis.orderBy(
    col("fraud_rate").desc()
).show(truncate=False)

spark.stop()