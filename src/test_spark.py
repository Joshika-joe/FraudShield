from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder \
    .appName("FraudShield") \
    .master("local[*]") \
    .getOrCreate()

print("Spark Version:", spark.version)
print("Spark Session Created Successfully!")

# Stop Spark
spark.stop()