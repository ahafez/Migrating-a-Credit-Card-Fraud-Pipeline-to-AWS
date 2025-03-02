import boto3
import datetime
import time
from pyspark.sql import SparkSession
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
import sys
import logging
import os

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
bucket_name = 'fraud-detector-bucker-123'
data_directory = 'glue-data/'  # Relative path within the S3 bucket
archive_directory = 'archive/' # Relative path within the S3 bucket
model_save_path = 'output/' # Relative path within the S3 bucket
glue_job_name = 'FraudDetectionTrainingJob'

# Initialize GlueContext
glueContext = GlueContext(SparkSession.builder.appName("CreditCardFraudDetection").getOrCreate())
spark = glueContext.spark_session

def watch_directory_and_retrain():
    while True:
        try:
            # List objects in the data directory
            objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=data_directory)
            if 'Contents' in objects:
                for obj in objects['Contents']:
                    file_key = obj['Key']
                    if file_key.endswith('.csv') and 'retrain' not in file_key and file_key != data_directory:
                        logger.info(f"New data detected: s3://{bucket_name}/{file_key}")

                        # Trigger Glue Job
                        glue_client = boto3.client('glue')
                        response = glue_client.start_job_run(
                            JobName=glue_job_name,
                            Arguments={
                                '--DATA_URL': f's3://{bucket_name}/{file_key}',
                                '--MODEL_SAVE_PATH': f's3://{bucket_name}/{model_save_path}'
                            }
                        )
                        logger.info(f"Glue job {glue_job_name} started with run ID: {response['JobRunId']}")

                        # Move the processed file to the archive directory
                        current_date = datetime.datetime.now().strftime("%Y%m%d")
                        archive_file_key = f"{archive_directory}{os.path.splitext(os.path.basename(file_key))[0]}_retrain_{current_date}.csv"
                        s3.copy_object(
                            Bucket=bucket_name,
                            CopySource={'Bucket': bucket_name, 'Key': file_key},
                            Key=archive_file_key
                        )
                        s3.delete_object(Bucket=bucket_name, Key=file_key)
                        logger.info(f"Moved processed file to: s3://{bucket_name}/{archive_file_key}")

            time.sleep(60)  # Check every 60 seconds

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    watch_directory_and_retrain()
