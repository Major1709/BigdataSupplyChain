from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml import Transformer
from pyspark.sql.functions import radians, cos, sin
from datetime import timedelta



class SegmentationClass(Transformer,DefaultParamsWritable, DefaultParamsReadable):

    def __init__(self):
       super(SegmentationClass, self).__init__()

    def _transform(self, dataset):
        # Colonnes à caster en Integer
        dataset = dataset.withColumn("lat_rad", radians("latitude"))
        dataset = dataset.withColumn("lon_rad", radians("longitude"))

        dataset = dataset.withColumn("x", cos("lat_rad") * cos("lon_rad")) \
                         .withColumn("y", cos("lat_rad") * sin("lon_rad")) \
                         .withColumn("z", sin("lat_rad"))
        dataset = dataset.dropDuplicates(["customer_id"])

        return dataset