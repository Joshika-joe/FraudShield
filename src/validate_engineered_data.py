from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

spark = SparkSession.builder \
    .appName("FraudShield_Data_Validation") \
    .master("local[*]") \
    .getOrCreate()

# Load engineered dataset
df = spark.read.csv(
    "outputs/engineered_data",
    header=True,
    inferSchema=True
)

# Display schema
print("\n--- DATASET SCHEMA ---")
df.printSchema()

# Total records
print("\n--- TOTAL RECORDS ---")
print("Total records:", df.count())

# Check missing values
print("\n--- MISSING VALUE CHECK ---")

null_counts = df.select([
    count(
        when(col(c).isNull(), c)
    ).alias(c)
    for c in df.columns
])

null_counts.show()

# Check fraud distribution
print("\n--- FRAUD DISTRIBUTION ---")
df.groupBy("isFraud").count().show()

# Check feature columns
print("\n--- ENGINEERED FEATURES ---")

features = [
    "sender_balance_change",
    "destination_balance_change",
    "log_amount",
    "sender_balance_mismatch",
    "destination_balance_mismatch"
]

for feature in features:
    if feature in df.columns:
        print(f"{feature}: PRESENT")
    else:
        print(f"{feature}: MISSING")

spark.stop()