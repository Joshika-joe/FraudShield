from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan

spark = SparkSession.builder \
    .appName("FraudShield_Numerical_Quality") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

numeric_columns = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

for column in numeric_columns:

    invalid_count = df.filter(
        col(column).isNull() |
        isnan(col(column)) |
        (col(column) < 0)
    ).count()

    print(f"{column}: {invalid_count} invalid records")

spark.stop()