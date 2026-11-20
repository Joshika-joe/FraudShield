from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, count

spark = SparkSession.builder \
    .appName("FraudShield_Data_Quality") \
    .master("local[*]") \
    .getOrCreate()

# Load dataset
df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

# 1. Missing value analysis
print("\n--- MISSING VALUES ---")

missing_values = df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
])

missing_values.show()

# 2. Duplicate analysis
print("\n--- DUPLICATE RECORDS ---")

total_records = df.count()
unique_records = df.dropDuplicates().count()

print("Total Records:", total_records)
print("Unique Records:", unique_records)
print("Duplicate Records:", total_records - unique_records)

# 3. Fraud distribution
print("\n--- FRAUD DISTRIBUTION ---")

df.groupBy("isFraud") \
    .count() \
    .show()

# 4. Transaction type distribution
print("\n--- TRANSACTION TYPE DISTRIBUTION ---")

df.groupBy("type") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

spark.stop()