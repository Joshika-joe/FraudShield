from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("FraudShield_Preprocessing") \
    .master("local[*]") \
    .getOrCreate()

# Load dataset
df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

# Select relevant columns
df = df.select(
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud"
)

# Verify schema
df.printSchema()

# Verify record count
print("Total records:", df.count())

# Save cleaned dataset
df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("outputs/cleaned_data")

print("Cleaned dataset saved successfully!")

spark.stop()