from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Check_Hadoop_Version") \
    .master("local[*]") \
    .getOrCreate()

print("Spark version:", spark.version)

print(
    "Hadoop version:",
    spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion()
)

spark.stop()