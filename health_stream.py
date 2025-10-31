from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Create Spark Session with Kafka support
spark = SparkSession.builder \
    .appName("FamilyHealthStream") \
    .getOrCreate()

# Define schema for incoming health data
schema = StructType([
    StructField("name", StringType()),
    StructField("age", DoubleType()),
    StructField("heart_rate", DoubleType()),
    StructField("bp", StringType()),
    StructField("glucose", DoubleType()),
    StructField("symptoms", StringType())
])

# Read from Kafka topic
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "test") \
    .option("startingOffsets", "earliest") \
    .load()

# Convert value to string and parse JSON
health_df = df.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")) \
    .select("data.*")

# Show real-time stream
query = health_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
