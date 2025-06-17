# predict_with_pytorch.py

import time
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import from_json, col

# 🧠 Spark Session
spark = SparkSession.builder \
    .appName("PyTorchPrediction") \
    .getOrCreate()

# 📥 Pipeline ML chargé
pipeline_model = PipelineModel.load("/home/toma/Documents/BIGDATA/big_data_pipeline_v3")
pipeline_seg = PipelineModel.load("/home/toma/Documents/BIGDATA/segmentation_model")

# 🔁 Boucle continue pour traitement périodique
while True:
    print("🔄 Lecture des nouvelles données...")
    df = spark.read.parquet("hdfs://localhost:9000/projet/output_stream")
    
    if df.isEmpty():
        print("🟡 Aucune donnée trouvée. Attente...")
        time.sleep(10)
        continue

    # 🧪 Prétraitement
    df = pipeline_model.transform(df)
    df = pipeline_seg.transform(df)
    

    pandas = df.toPandas()
    pandas_df_clean = pandas.applymap(lambda x: str(x) if x is not None else "")

    # 2. Construire un schéma explicite (toutes les colonnes en StringType)
    schema = StructType([StructField(col_name, StringType(), True) for col_name in pandas_df_clean.columns])

    # 3. Convertir en liste de lignes (dictionnaires)
    data_list = pandas_df_clean.to_dict(orient="records")

    # 4. Créer un Spark DataFrame avec schéma explicite
    spark_df = spark.createDataFrame(data_list, schema=schema)
    # 5. Écrire dans HDFS au format Parquet
    spark_df.write.mode("append").parquet("hdfs://localhost:9000/projet/predictions_segment")
    print("✅ Prédictions enregistrées")
    time.sleep(5)  # Attente avant le prochain traitement
