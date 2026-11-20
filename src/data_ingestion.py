from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder \
    .appName("FraudShield_Data_Ingestion") \
    .master("local[*]") \
    .getOrCreate()

# Dataset path
file_path = "dataset/paysim.csv"

# Load CSV into Spark DataFrame
df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

# Display schema
print("DATASET SCHEMA")
df.printSchema()

# Display first 5 records
print("FIRST 5 RECORDS")
df.show(5, truncate=False)

# Display total number of records
print("TOTAL RECORDS:", df.count())

spark.stop()