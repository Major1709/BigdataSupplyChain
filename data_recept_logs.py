# stream_to_parquet.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType

# 🧠 Initialisation de Spark
spark = SparkSession.builder \
    .appName("KafkaToParquet") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# 📥 Schéma à inférer ou définir manuellement si besoin
schema = spark.read.csv("hdfs://localhost:9000/projet/raw/logs/tokenized_access_logs.csv", header=True).schema

# 🔁 Lecture du flux Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "logs_topic") \
    .load()

# 🎯 Extraction JSON
json_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# 💾 Écriture dans HDFS en Parquet
query = json_df.writeStream \
    .format("parquet") \
    .option("path", "hdfs://localhost:9000/projet/streamed_logs") \
    .option("checkpointLocation", "hdfs://localhost:9000/projet/checkpoints/logs_topic") \
    .outputMode("append") \
    .start()

query.awaitTermination()
