# predict_with_pytorch.py

import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml import PipelineModel
import torch
from torch.utils.data import DataLoader
from pipeline_for_model_day.SupplyChainDataset import SupplyChainDataset
from pipeline_for_model_day.SupplyChainDataset_D import SupplyChainDataset_D
from pipeline_for_model_day.SupplyChainModel_D import SupplyChainModel_D
from pipeline_for_model_day.SupplyChainModel import SupplyChainModel
import pandas as pd

# 🧠 Spark Session
spark = SparkSession.builder \
    .appName("PyTorchPrediction") \
    .getOrCreate()

# 📥 Pipeline ML chargé
pipeline_model = PipelineModel.load("/home/toma/Documents/BIGDATA/big_data_pipeline_v3")
pipeline_day = PipelineModel.load("/home/toma/Documents/BIGDATA/day_pipeline")

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
    df = pipeline_day.transform(df)

    pandas_df = df.toPandas()

    # ✅ Prédiction : late_delivery_risk
    dataset = SupplyChainDataset(pandas_df)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model_late = SupplyChainModel()  # 👈 complète ici*
    model_late.load_state_dict(torch.load("/home/toma/Documents/BIGDATA/late_modal_weights_only.pt"))
    model_late.eval()  # mode évaluation
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            x_cat = batch["cat"]
            x_num = batch["num"]
            y_pred = model_late(x_cat, x_num)
            preds = (torch.sigmoid(y_pred) > 0.5).int()
            predictions.append(preds)

    pandas_df["late_delivery_risk"] = torch.cat(predictions, dim=0).numpy()

    cols_to_drop = [col for col in pandas_df.columns if "_index" in col]
    # Supprimer ces colonnes
    test_df =pandas_df.drop(cols_to_drop, axis=1)


    # ✅ Prédiction : days_for_shipping_(real)
    day_dataset = SupplyChainDataset_D(pandas_df)
    loader_day = DataLoader(day_dataset, batch_size=64, shuffle=False)
    model_day = SupplyChainModel_D()
    model_day.load_state_dict(torch.load("/home/toma/Documents/BIGDATA/day_model_weights_only.pt"))
    model_day.eval()

    predictions_day = []
    with torch.no_grad():
        for batch in loader_day:
            x_cat = batch["cat"]
            x_num = batch["num"]
            y_pred = model_day(x_cat, x_num)
            predictions_day.append(y_pred.squeeze())

    
    predictions_tensor = torch.cat(predictions_day, dim=0).squeeze()
    pandas_df["days_for_shipping_(real)"] = predictions_tensor.round().numpy()

    cols_to_drop = [col for col in pandas_df.columns if "_index" in col]
    # Supprimer ces colonnes
    test_df =pandas_df.drop(cols_to_drop, axis=1)

    # 1. Convertir chaque cellule du DataFrame en chaîne de caractères proprement
    pandas_df_clean = test_df.applymap(lambda x: str(x) if x is not None else "")

    # 2. Construire un schéma explicite (toutes les colonnes en StringType)
    schema = StructType([StructField(col_name, StringType(), True) for col_name in pandas_df_clean.columns])

    # 3. Convertir en liste de lignes (dictionnaires)
    data_list = pandas_df_clean.to_dict(orient="records")

    # 4. Créer un Spark DataFrame avec schéma explicite
    spark_df = spark.createDataFrame(data_list, schema=schema)

    # 5. Écrire dans HDFS au format Parquet
    spark_df.write.mode("append").parquet("hdfs://localhost:9000/projet/predictions")
    print("✅ Prédictions enregistrées")
    time.sleep(15)  # Attente avant le prochain traitement
