from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum

import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .appName("FraudShield_Visualization") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

result = df.groupBy("type").agg(
    count("*").alias("total"),
    sum("isFraud").alias("fraud")
)

result = result.withColumn(
    "fraud_rate",
    col("fraud") / col("total") * 100
)

# Convert small aggregated result to Pandas
pdf = result.toPandas()

plt.figure(figsize=(9, 5))

plt.bar(pdf["type"], pdf["fraud_rate"])

plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Rate (%)")

plt.tight_layout()
plt.show()

spark.stop()