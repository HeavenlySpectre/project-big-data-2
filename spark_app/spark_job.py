# Coba untuk melatih model SparkML dari data Parquet
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, StructType, StructField
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SparkJob - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Coba tentukan path sesuai dengan volume mounts di docker-compose
PARQUET_INPUT_PATH = "/input_parquet_data" # pipeline_data volume
USERS_CSV_PATH = "/input_datasets/users.csv" # raw_datasets volume (local ./data)
GAMES_CSV_PATH = "/input_datasets/games.csv" # raw_datasets volume (local ./data)
MODEL_OUTPUT_PATH = "/models_output" # model_storage volume

def create_spark_session():
    try:
        spark = SparkSession.builder \
            .appName("SparkRecommendationModelTraining") \
            .config("spark.sql.files.ignoreCorruptFiles", "true") \
            .config("spark.sql.files.ignoreMissingFiles", "true") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .getOrCreate()
        logger.info("✅ Spark session created successfully")
        # Coba set log level Spark lebih rendah untuk mengurangi verbosity
        spark.sparkContext.setLogLevel("WARN")
        return spark
    except Exception as e:
        logger.error(f"Error creating SparkSession: {e}")
        raise

def load_and_prepare_recs_data(spark, path):
    logger.info(f"Loading recommendations data from Parquet path: {path}")
    try:
        # Coba definisikan skema dasar untuk data rekomendasi
        # app_id, user_id, is_recommended, hours, date
        schema = StructType([
            StructField("app_id", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("is_recommended", StringType(), True), # Akan dikonversi ke numerik
            StructField("hours", StringType(), True),         # Akan dikonversi ke numerik
            StructField("date", StringType(), True)
        ])

        df = spark.read.schema(schema).parquet(f"{path}/*.parquet")
        
        df = df.withColumn("app_id_int", col("app_id").cast(IntegerType())) \
               .withColumn("user_id_int", col("user_id").cast(IntegerType())) \
               .withColumn("is_recommended_label", col("is_recommended").cast(DoubleType())) \
               .withColumn("hours_numeric", col("hours").cast(DoubleType()))

        # Coba filter baris dengan nilai null penting setelah konversi
        df = df.na.drop(subset=["app_id_int", "user_id_int", "is_recommended_label", "hours_numeric"])
        
        # Coba tambahkan ID baris unik untuk pemfilteran "first_X_rows" yang konsisten
        df = df.withColumn("row_id", monotonically_increasing_id())

        logger.info(f"Loaded {df.count()} rows from recommendations data.")
        df.printSchema()
        return df
    except Exception as e:
        logger.error(f"Error loading or preparing recommendations data: {e}")
        return None

def train_m1_lr(spark, data_df, model_path):
    logger.info("Starting M1 Logistic Regression training...")
    # Coba ambil "first_500k_rows"
    training_data = data_df.orderBy("row_id").limit(500000)
    
    if training_data.count() < 100: # Coba pastikan ada cukup data
        logger.warning("Not enough data for M1_LR training. Skipping.")
        return

    logger.info(f"M1_LR training with {training_data.count()} rows.")
    
    try:
        # Coba gunakan 'hours_numeric' sebagai fitur tunggal untuk kesederhanaan
        assembler = VectorAssembler(inputCols=["hours_numeric"], outputCol="features", handleInvalid="skip")
        lr = LogisticRegression(featuresCol="features", labelCol="is_recommended_label")
        pipeline = Pipeline(stages=[assembler, lr])
        
        model = pipeline.fit(training_data)
        model_save_path = f"{model_path}/M1_LR"
        model.write().overwrite().save(model_save_path)
        logger.info(f"M1_LR model trained and saved to {model_save_path}")

        # Coba evaluasi sederhana
        predictions = model.transform(training_data)
        evaluator = BinaryClassificationEvaluator(labelCol="is_recommended_label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        logger.info(f"M1_LR Training AUC: {auc}")

    except Exception as e:
        logger.error(f"Error during M1_LR training: {e}")

def train_m2_gbt(spark, data_df, model_path):
    logger.info("Starting M2 GBTClassifier training...")
    # Coba ambil "first_1M_rows"
    training_data = data_df.orderBy("row_id").limit(1000000)

    if training_data.count() < 100: # Coba pastikan ada cukup data
        logger.warning("Not enough data for M2_GBT training. Skipping.")
        return

    logger.info(f"M2_GBT training with {training_data.count()} rows.")

    try:
        # Coba gunakan 'hours_numeric' sebagai fitur tunggal
        assembler = VectorAssembler(inputCols=["hours_numeric"], outputCol="features", handleInvalid="skip")
        gbt = GBTClassifier(featuresCol="features", labelCol="is_recommended_label", maxIter=10) # Coba iterasi rendah untuk kecepatan
        pipeline = Pipeline(stages=[assembler, gbt])

        model = pipeline.fit(training_data)
        model_save_path = f"{model_path}/M2_GBT"
        model.write().overwrite().save(model_save_path)
        logger.info(f"M2_GBT model trained and saved to {model_save_path}")

        # Coba evaluasi sederhana
        predictions = model.transform(training_data)
        evaluator = BinaryClassificationEvaluator(labelCol="is_recommended_label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        logger.info(f"M2_GBT Training AUC: {auc}")
        # Coba cek baseline metrik dari plan.md
        if auc > 0.70:
            logger.info(f"M2_GBT AUC ({auc:.4f}) meets success criteria (> 0.70).")
        else:
            logger.warning(f"M2_GBT AUC ({auc:.4f}) does NOT meet success criteria (> 0.70).")

    except Exception as e:
        logger.error(f"Error during M2_GBT training: {e}")

def train_m3_als(spark, data_df, model_path):
    logger.info("Starting M3 ALS training...")
    # Coba gunakan dataset penuh untuk ALS
    # ALS membutuhkan kolom user, item, rating. Kita gunakan user_id_int, app_id_int, hours_numeric
    training_data = data_df.select(
        col("user_id_int").alias("user"),
        col("app_id_int").alias("item"),
        col("hours_numeric").alias("rating")
    ).na.drop() # Coba hapus baris dengan null di kolom ALS

    if training_data.count() < 100: # Coba pastikan ada cukup data
        logger.warning("Not enough data for M3_ALS training. Skipping.")
        return

    logger.info(f"M3_ALS training with {training_data.count()} user-item-rating interactions.")

    try:
        als = ALS(
            maxIter=5, # Coba iterasi rendah untuk kecepatan
            regParam=0.01,
            userCol="user",
            itemCol="item",
            ratingCol="rating",
            coldStartStrategy="drop", # Coba drop user/item baru saat prediksi
            nonnegative=True # Jika rating (hours) selalu positif
        )
        
        model = als.fit(training_data)
        model_save_path = f"{model_path}/M3_ALS"
        model.write().overwrite().save(model_save_path)
        logger.info(f"M3_ALS model trained and saved to {model_save_path}")

        # Coba evaluasi sederhana (opsional, karena data uji tidak dipisahkan di sini)
        # predictions = model.transform(training_data)
        # evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        # rmse = evaluator.evaluate(predictions.na.drop()) # Drop NaNs in prediction
        # logging.info(f"M3_ALS Training RMSE: {rmse}")

    except Exception as e:
        logger.error(f"Error during M3_ALS training: {e}")

def main():
    logger.info("🚀 Starting Spark ML training job...")
    spark = None
    try:
        spark = create_spark_session()
        
        recs_df = load_and_prepare_recs_data(spark, PARQUET_INPUT_PATH)

        if recs_df and recs_df.count() > 0:
            # Coba pastikan direktori model ada, meskipun Spark akan membuatnya
            # os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True) # Tidak diperlukan jika Spark menyimpan langsung ke HDFS atau volume

            train_m1_lr(spark, recs_df, MODEL_OUTPUT_PATH)
            train_m2_gbt(spark, recs_df, MODEL_OUTPUT_PATH)
            train_m3_als(spark, recs_df, MODEL_OUTPUT_PATH)
            
            logger.info("Spark job finished all training tasks.")
        else:
            logger.warning("No recommendations data found or loaded. Skipping training.")

    except Exception as e:
        logger.critical(f"Spark job failed: {e}")
    finally:
        if spark:
            spark.stop()
            logger.info("SparkSession stopped.")

if __name__ == "__main__":
    main()