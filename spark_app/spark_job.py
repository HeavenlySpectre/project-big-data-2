# Coba untuk melatih model SparkML dari data Parquet
import logging
import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, when
from pyspark.sql.types import StringType, IntegerType, DoubleType, StructType, StructField
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SparkJob - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Coba tentukan path sesuai dengan volume mounts di docker-compose
# Prefix with file:/// to indicate local filesystem paths for Spark
# when fs.defaultFS might be HDFS.
PARQUET_INPUT_PATH = "file:///input_parquet_data" # pipeline_data volume
USERS_CSV_PATH = "file:///input_datasets/users.csv" # raw_datasets volume (local ./data)
GAMES_CSV_PATH = "file:///input_datasets/games.csv" # raw_datasets volume (local ./data)
RECOMMENDATIONS_CSV_PATH = "file:///input_datasets/realrecommendations.csv" # raw_datasets volume (local ./data)
MODEL_OUTPUT_PATH = "file:///models_output" # model_storage volume

def check_model_exists(spark, model_path, model_name):
    """Check if model exists and return True if it should be skipped"""
    try:
        from pyspark.ml.pipeline import PipelineModel
        from pyspark.ml.recommendation import ALSModel
        
        full_path = f"{model_path}/{model_name}"
        
        # Try to load the model to verify it exists and is valid
        if model_name == "M3_ALS":
            ALSModel.load(full_path)
        else:
            PipelineModel.load(full_path)
            
        logger.info(f"✅ Model {model_name} exists and is valid, skipping training")
        return True
        
    except Exception:
        logger.info(f"🆕 Model {model_name} not found or invalid, will train new model")
        return False

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

def load_and_prepare_recs_data(spark, use_csv=True):
    if use_csv:
        logger.info("Loading and joining data from CSV files...")
        try:
            # Define schemas for CSV files based on actual headers
            users_schema = StructType([
                StructField("user_id", StringType(), True),
                StructField("products", StringType(), True),
                StructField("reviews", StringType(), True)
            ])

            games_schema = StructType([
                StructField("app_id", StringType(), True),
                StructField("title", StringType(), True),
                StructField("date_release", StringType(), True),
                StructField("win", StringType(), True),
                StructField("mac", StringType(), True),
                StructField("linux", StringType(), True),
                StructField("rating", StringType(), True),
                StructField("positive_ratio", StringType(), True),
                StructField("user_reviews", StringType(), True),
                StructField("price_final", StringType(), True),
                StructField("price_original", StringType(), True),
                StructField("discount", StringType(), True),
                StructField("steam_deck", StringType(), True)
            ])

            recommendations_schema = StructType([
                StructField("app_id", StringType(), True),
                StructField("helpful", StringType(), True),
                StructField("funny", StringType(), True),
                StructField("date", StringType(), True),
                StructField("is_recommended", StringType(), True),
                StructField("hours", StringType(), True),
                StructField("user_id", StringType(), True),
                StructField("review_id", StringType(), True)
            ])

            # Read CSV files
            logger.info(f"Reading users data from: {USERS_CSV_PATH}")
            users_df = spark.read.csv(USERS_CSV_PATH, header=True, schema=users_schema)
            
            logger.info(f"Reading games data from: {GAMES_CSV_PATH}")
            games_df = spark.read.csv(GAMES_CSV_PATH, header=True, schema=games_schema)
            
            logger.info(f"Reading recommendations data from: {RECOMMENDATIONS_CSV_PATH}")
            recommendations_df = spark.read.csv(RECOMMENDATIONS_CSV_PATH, header=True, schema=recommendations_schema)

            logger.info(f"Loaded {users_df.count()} users, {games_df.count()} games, {recommendations_df.count()} recommendations")

            # Join recommendations with users first
            logger.info("Joining recommendations with users...")
            joined_df = recommendations_df.join(users_df, "user_id", "inner")
            
            # Then join with games
            logger.info("Joining with games...")
            joined_df = joined_df.join(games_df, "app_id", "inner")
            
            logger.info(f"After joins: {joined_df.count()} rows")            # Transform data into expected schema with enhanced features (Phase 1)
            logger.info("Transforming data to enhanced model schema...")
            df = joined_df.select(
                col("app_id"),
                col("user_id"),
                col("is_recommended"),
                col("hours"),
                col("date"),
                # Enhanced features from games data
                col("price_original"),
                col("positive_ratio"),
                col("user_reviews"),
                col("rating")
            )

            # Cast to required types with enhanced features
            df = df.withColumn("app_id_int", col("app_id").cast(IntegerType())) \
                   .withColumn("user_id_int", col("user_id").cast(IntegerType())) \
                   .withColumn("is_recommended_label", 
                              col("is_recommended").cast("boolean").cast(DoubleType())) \
                   .withColumn("hours_numeric", col("hours").cast(DoubleType())) \
                   .withColumn("price_numeric", col("price_original").cast(DoubleType())) \
                   .withColumn("positive_ratio_numeric", col("positive_ratio").cast(DoubleType())) \
                   .withColumn("user_reviews_numeric", col("user_reviews").cast(DoubleType()))

            # Create price categories (budget, mid-range, premium)
            df = df.withColumn("price_category", 
                              when(col("price_numeric") <= 9.99, 1.0)
                              .when(col("price_numeric") <= 29.99, 2.0)
                              .otherwise(3.0))

            # Create quality score from positive ratio and review count
            df = df.withColumn("quality_score", 
                              col("positive_ratio_numeric") * 
                              (col("user_reviews_numeric") / 1000.0).cast(DoubleType()))            # Filter out rows with null values in critical columns
            df = df.na.drop(subset=["app_id_int", "user_id_int", "is_recommended_label", "hours_numeric"])
            
            # Filter out unrealistic data (Phase 1 data cleaning)
            logger.info("Applying data quality filters...")
            df = df.filter(
                (col("hours_numeric") >= 0.1) &  # At least 6 minutes played
                (col("hours_numeric") <= 10000) &  # Max 10,000 hours (reasonable upper bound)
                (col("price_numeric") >= 0) &  # Non-negative price
                (col("price_numeric") <= 200) &  # Reasonable max price
                (col("positive_ratio_numeric") >= 0) &  # Valid positive ratio
                (col("positive_ratio_numeric") <= 100)  # Valid positive ratio
            )
            
            # Add unique row ID for consistent ordering
            df = df.withColumn("row_id", monotonically_increasing_id())

            logger.info(f"Final processed data: {df.count()} rows")
            df.printSchema()
            
            # Optional: Save processed data to Parquet for faster loading in future runs
            try:
                parquet_path = f"{PARQUET_INPUT_PATH}/processed_recs_data.parquet"
                logger.info(f"Saving processed data to Parquet: {parquet_path}")
                df.write.mode("overwrite").parquet(parquet_path)
                logger.info("Successfully saved processed data to Parquet")
            except Exception as parquet_error:
                logger.warning(f"Could not save to Parquet (continuing with CSV data): {parquet_error}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading or preparing data from CSVs: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    else:
        # Fallback to original Parquet loading logic
        logger.info(f"Loading recommendations data from Parquet path: {PARQUET_INPUT_PATH}")
        try:
            schema = StructType([
                StructField("app_id", StringType(), True),
                StructField("user_id", StringType(), True),
                StructField("is_recommended", StringType(), True),
                StructField("hours", StringType(), True),
                StructField("date", StringType(), True)
            ])

            df = spark.read.schema(schema).parquet(f"{PARQUET_INPUT_PATH}/*.parquet")
            
            df = df.withColumn("app_id_int", col("app_id").cast(IntegerType())) \
                   .withColumn("user_id_int", col("user_id").cast(IntegerType())) \
                   .withColumn("is_recommended_label", col("is_recommended").cast(DoubleType())) \
                   .withColumn("hours_numeric", col("hours").cast(DoubleType()))

            df = df.na.drop(subset=["app_id_int", "user_id_int", "is_recommended_label", "hours_numeric"])
            df = df.withColumn("row_id", monotonically_increasing_id())

            logger.info(f"Loaded {df.count()} rows from recommendations data.")
            df.printSchema()
            return df
        except Exception as e:
            logger.error(f"Error loading or preparing recommendations data: {e}")
            return None

def train_m1_lr(spark, data_df, model_path):
    logger.info("Starting M1 Logistic Regression training with enhanced features...")
    # Use first 500k rows for faster training
    training_data = data_df.orderBy("row_id").limit(500000)
    
    if training_data.count() < 100:
        logger.warning("Not enough data for M1_LR training. Skipping.")
        return

    logger.info(f"M1_LR training with {training_data.count()} rows.")
    
    try:
        # Enhanced features: hours + price + quality + price category
        feature_cols = ["hours_numeric", "price_numeric", "positive_ratio_numeric", 
                       "quality_score", "price_category"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        lr = LogisticRegression(featuresCol="features", labelCol="is_recommended_label", maxIter=20)
        pipeline = Pipeline(stages=[assembler, lr])
        
        model = pipeline.fit(training_data)
        model_save_path = f"{model_path}/M1_LR"
        model.write().overwrite().save(model_save_path)
        logger.info(f"M1_LR model trained and saved to {model_save_path}")

        # Evaluate model
        predictions = model.transform(training_data)
        evaluator = BinaryClassificationEvaluator(labelCol="is_recommended_label", 
                                                rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        logger.info(f"M1_LR Training AUC: {auc:.4f}")
        
        if auc > 0.70:
            logger.info(f"✅ M1_LR AUC ({auc:.4f}) meets success criteria (> 0.70)")
        else:
            logger.warning(f"⚠️ M1_LR AUC ({auc:.4f}) below target (0.70)")

    except Exception as e:
        logger.error(f"Error during M1_LR training: {e}")

def train_m2_gbt(spark, data_df, model_path):
    logger.info("Starting M2 GBTClassifier training with enhanced features...")
    # Use first 750k rows (more than LR but less than full for speed)
    training_data = data_df.orderBy("row_id").limit(750000)

    if training_data.count() < 100:
        logger.warning("Not enough data for M2_GBT training. Skipping.")
        return

    logger.info(f"M2_GBT training with {training_data.count()} rows.")

    try:
        # Enhanced features: same as M1_LR for comparison
        feature_cols = ["hours_numeric", "price_numeric", "positive_ratio_numeric", 
                       "quality_score", "price_category"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        gbt = GBTClassifier(featuresCol="features", labelCol="is_recommended_label", 
                           maxIter=15, maxDepth=5)  # Balanced speed vs accuracy
        pipeline = Pipeline(stages=[assembler, gbt])

        model = pipeline.fit(training_data)
        model_save_path = f"{model_path}/M2_GBT"
        model.write().overwrite().save(model_save_path)
        logger.info(f"M2_GBT model trained and saved to {model_save_path}")

        # Evaluate model
        predictions = model.transform(training_data)
        evaluator = BinaryClassificationEvaluator(labelCol="is_recommended_label", 
                                                rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        logger.info(f"M2_GBT Training AUC: {auc:.4f}")
        
        if auc > 0.70:
            logger.info(f"✅ M2_GBT AUC ({auc:.4f}) meets success criteria (> 0.70)")
        else:
            logger.warning(f"⚠️ M2_GBT AUC ({auc:.4f}) below target (0.70)")

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
          # Try to load from CSV files first, fallback to Parquet if needed
        recs_df = load_and_prepare_recs_data(spark, use_csv=True)

        if recs_df and recs_df.count() > 0:
            logger.info("Successfully loaded and prepared data. Starting smart model training...")

            # Smart model checking - only train if models don't exist or are invalid
            models_to_train = []
            
            if not check_model_exists(spark, MODEL_OUTPUT_PATH, "M1_LR"):
                models_to_train.append("M1_LR")
            else:
                logger.info("M1_LR model already exists and is valid, skipping training")
            
            if not check_model_exists(spark, MODEL_OUTPUT_PATH, "M2_GBT"):
                models_to_train.append("M2_GBT")
            else:
                logger.info("M2_GBT model already exists and is valid, skipping training")
            
            if not check_model_exists(spark, MODEL_OUTPUT_PATH, "M3_ALS"):
                models_to_train.append("M3_ALS")
            else:
                logger.info("M3_ALS model already exists and is valid, skipping training")

            if not models_to_train:
                logger.info("🎉 All models already exist and are valid! No training needed.")
                logger.info("⚡ Quick startup completed in seconds instead of minutes!")
            else:
                logger.info(f"🔧 Training {len(models_to_train)} models: {models_to_train}")
                
                # Train only the models that need training
                if "M1_LR" in models_to_train:
                    train_m1_lr(spark, recs_df, MODEL_OUTPUT_PATH)
                
                if "M2_GBT" in models_to_train:
                    train_m2_gbt(spark, recs_df, MODEL_OUTPUT_PATH)
                
                if "M3_ALS" in models_to_train:
                    train_m3_als(spark, recs_df, MODEL_OUTPUT_PATH)
            
            logger.info("✅ Spark job finished all training tasks successfully.")
        else:
            logger.warning("No recommendations data found or loaded. Skipping training.")

    except Exception as e:
        logger.critical(f"Spark job failed: {e}")
        import traceback
        logger.critical(traceback.format_exc())
    finally:
        if spark:
            spark.stop()
            logger.info("SparkSession stopped.")

if __name__ == "__main__":
    main()