# kafka_producer.py
from kafka import KafkaProducer
import time
import pandas as pd
from pyspark.sql import SparkSession
import json

spark = SparkSession.builder \
    .appName("KafkaHDFSProducer") \
    .getOrCreate()

# Charger depuis HDFS
# Lire en CSV avec header
df = spark.read.csv("hdfs://localhost:9000/projet/raw/logs/tokenized_access_logs.csv", header=True)
df_pd = df.toPandas()

producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda x: json.dumps(x).encode('utf-8'))

for _, row in df_pd.iterrows():
    data = row.to_dict()
    producer.send("logs_topic", value=data)
    time.sleep(5)  # simule le temps réel
