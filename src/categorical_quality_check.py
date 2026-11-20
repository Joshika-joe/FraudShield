from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("FraudShield_Categorical_Check") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "dataset/paysim.csv",
    header=True,
    inferSchema=True
)

# Check transaction types
print("Transaction types:")
df.groupBy("type").count().show()

# Check missing categorical values
categorical_columns = [
    "type",
    "nameOrig",
    "nameDest"
]

print("Missing categorical values:")

for column in categorical_columns:
    missing = df.filter(
        col(column).isNull() | (col(column) == "")
    ).count()

    print(f"{column}: {missing}")

spark.stop()