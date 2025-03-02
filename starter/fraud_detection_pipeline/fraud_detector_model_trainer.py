from awsglue.context import GlueContext
from pyspark.sql import SparkSession, DataFrame
from awsglue.dynamicframe import DynamicFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler, Bucketizer, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, hour, when
from pyspark.sql.types import DoubleType

glueContext = GlueContext(SparkSession.builder.appName("CreditCardFraudDetection").getOrCreate())
spark = glueContext.spark_session

def MyTransform (glueContext, dfc) -> DynamicFrameCollection:
    try:
        df = dynamic_frame.toDF()

        numeric_columns = [col for col in df.columns if col != 'Class']
        for column in numeric_columns:
            df = df.withColumn(column, col(column).cast(DoubleType()))

        assembler = VectorAssembler(inputCols=numeric_columns, outputCol="numericFeatures")
        scaler = StandardScaler(inputCol="numericFeatures", outputCol="scaledFeatures", withStd=True, withMean=True)

        finalAssembler = VectorAssembler(inputCols=["scaledFeatures"], outputCol="features")

        class_counts = df.groupBy("Class").count().collect()
        total_count = sum([row['count'] for row in class_counts])
        weight_dict = {row['Class']: total_count / row['count'] for row in class_counts}

        df = df.withColumn("weight", when(col("Class") == 0, weight_dict[0]).otherwise(weight_dict[1]))

        indexer = StringIndexer(inputCol="Class", outputCol="label")

        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=10, weightCol="weight")

        pipeline = Pipeline(stages=[assembler, scaler, finalAssembler, indexer, rf])

        model = pipeline.fit(df)

        model_path = 's3://s3://fraud-detector-bucker-123/model/fraud_detection_model_latest'
        model.write().mode("overwrite").save(model_path)

        print(f'Model trained and saved to {model_path}')

        dynamic_frame_output = DynamicFrame.fromDF(df, glueContext, "output_dynamic_frame")
        return dynamic_frame_output
    except Exception as e:
        print(f"Error: {e}")
        return None
