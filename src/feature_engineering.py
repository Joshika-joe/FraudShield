from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p, round, when, abs

spark = SparkSession.builder \
    .appName("FraudShield_Feature_Engineering") \
    .master("local[*]") \
    .getOrCreate()

# Read cleaned dataset
df = spark.read.csv(
    "outputs/cleaned_data",
    header=True,
    inferSchema=True
)

# Feature 1: Sender balance change
df = df.withColumn(
    "sender_balance_change",
    round(col("oldbalanceOrg") - col("newbalanceOrig"), 2)
)

# Feature 2: Destination balance change
df = df.withColumn(
    "destination_balance_change",
    round(col("newbalanceDest") - col("oldbalanceDest"), 2)
)

# Feature 3: Log transaction amount
df = df.withColumn(
    "log_amount",
    log1p(col("amount"))
)

# Display engineered features

# Feature 4: Sender balance mismatch
df = df.withColumn(
    "sender_balance_mismatch",
    when(
        abs(
            col("oldbalanceOrg") -
            col("amount") -
            col("newbalanceOrig")
        ) > 0.01, 1
    ).otherwise(0)
)

# Feature 5: Destination balance mismatch
df = df.withColumn(
    "destination_balance_mismatch",
    when(
        abs(
            col("oldbalanceDest") +
            col("amount") -
            col("newbalanceDest")
        ) > 0.01, 1
    ).otherwise(0)
)
df.select(
    "amount",
    "sender_balance_change",
    "destination_balance_change",
    "log_amount",
    "sender_balance_mismatch",
    "destination_balance_mismatch",
    "isFraud"
).show(10, truncate=False)

print("Total records:", df.count())
df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("outputs/engineered_data")

print("Engineered dataset saved successfully!")

spark.stop()