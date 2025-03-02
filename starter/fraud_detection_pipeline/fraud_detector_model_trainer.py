from awsglue.context import GlueContext
from pyspark.sql import SparkSession, DataFrame
from awsglue.dynamicframe import DynamicFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler, Bucketizer, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, hour, when
from pyspark.sql.types import DoubleType
import sys
from awsglue.utils import getResolvedOptions
import logging

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get Glue Job arguments
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'DATA_URL', 'MODEL_SAVE_PATH'])
job_name = args['JOB_NAME']
data_url = args['DATA_URL']
model_save_path = args['MODEL_SAVE_PATH']

glueContext = GlueContext(SparkSession.builder.appName(job_name).getOrCreate())
spark = glueContext.spark_session

def transform_and_train(glueContext, data_url, model_save_path):
    try:
        # Load data from S3
        dynamic_frame = glueContext.create_dynamic_frame.from_options(
            format_options={"quoteChar": '"', "withHeader": True, "separator": ","},
            connection_type="s3",
            connection_options={"paths": [data_url]},
            format="csv",
        )
        df = dynamic_frame.toDF()

        # Data Validation and Type Casting
        numeric_columns = [col_name for col_name in df.columns if col_name != 'Class']
        for column in numeric_columns:
            df = df.withColumn(column, col(column).cast(DoubleType()))

        # Feature Engineering
        assembler = VectorAssembler(inputCols=numeric_columns, outputCol="numericFeatures")
        scaler = StandardScaler(inputCol="numericFeatures", outputCol="scaledFeatures", withStd=True, withMean=True)
        finalAssembler = VectorAssembler(inputCols=["scaledFeatures"], outputCol="features")

        # Handle Class Imbalance
        class_counts = df.groupBy("Class").count().collect()
        total_count = sum([row['count'] for row in class_counts])
        weight_dict = {row['Class']: total_count / row['count'] for row in class_counts}
        df = df.withColumn("weight", when(col("Class") == 0, weight_dict[0]).otherwise(weight_dict[1]))

        # Model Training
        indexer = StringIndexer(inputCol="Class", outputCol="label")
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=10, weightCol="weight")
        pipeline = Pipeline(stages=[assembler, scaler, finalAssembler, indexer, rf])
        model = pipeline.fit(df)

        # Save Model to S3 with Overwrite and logging.
        model.write().mode("overwrite").save(model_save_path)
        logger.info(f'Model trained and saved to {model_save_path}')

        # Create output dynamic frame.
        dynamic_frame_output = DynamicFrame.fromDF(df, glueContext, "output_dynamic_frame")
        return dynamic_frame_output

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True) # Log the full exception
        return None

# Execute the transformation and training
output_dynamic_frame = transform_and_train(glueContext, data_url, model_save_path)

if output_dynamic_frame:
    logger.info("Glue job completed successfully.")
else:
    logger.error("Glue job failed.")
