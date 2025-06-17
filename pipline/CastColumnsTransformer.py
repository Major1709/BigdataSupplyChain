from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml import Transformer
from pyspark.sql.functions import to_timestamp



class CastColumnsTransformer(Transformer,DefaultParamsWritable, DefaultParamsReadable):

    def __init__(self):
       super(CastColumnsTransformer, self).__init__()

    def _transform(self, dataset):
        # Colonnes à caster en Integer
        int_cols = [
            "Days for shipping (real)",
            "Days for shipment (scheduled)",
            "Customer Zipcode",
            "Department Id",
            "Product Card Id",
            "Order Item Quantity",
            "Product Status",
            "Late_delivery_risk"
        ]
        for c in int_cols:
            dataset = dataset.withColumn(c, col(c).cast(IntegerType()))

        # Colonnes à caster en Double
        double_cols = [
            "Benefit per order",
            "Sales per customer",
            "Order Item Discount",
            "Order Item Discount Rate",
            "Order Item Product Price",
            "Order Item Profit Ratio",
            "Sales",
            "Order Item Total",
            "Order Profit Per Order",
            "Product Price",
            "Latitude",
            "Longitude"
        ]
        for c in double_cols:
            dataset = dataset.withColumn(c, col(c).cast(DoubleType()))

        date_cols = [
            "order date (DateOrders)",
            "shipping date (DateOrders)"
        ]
        
        for c in date_cols:
            dataset = dataset.withColumn(c, to_timestamp(col(c), "M/d/yyyy H:mm"))

            # ✅ Renommage explicite sans utiliser toDF
        for name in dataset.columns:
            new_name = name.lower().replace(" ", "_")
            if name != new_name:
                dataset = dataset.withColumnRenamed(name, new_name)

        return dataset