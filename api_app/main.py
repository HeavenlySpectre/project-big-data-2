# Coba untuk FastAPI app yang melayani model SparkML
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import VectorAssembler # Coba jika perlu membuat vektor fitur di API
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
# Coba impor col jika belum diimpor di atas
from pyspark.sql.functions import col


logging.basicConfig(level=logging.INFO, format='%(asctime)s - API - %(levelname)s - %(message)s')

# Coba path model sesuai dengan volume mount
MODEL_INPUT_PATH = "/models_input"
M1_LR_PATH = f"{MODEL_INPUT_PATH}/M1_LR"
M2_GBT_PATH = f"{MODEL_INPUT_PATH}/M2_GBT"
M3_ALS_PATH = f"{MODEL_INPUT_PATH}/M3_ALS"

app = FastAPI(title="Recommendation API")

# Coba untuk state aplikasi global (SparkSession dan model)
app_state = {
    "spark": None,
    "m1_lr_model": None,
    "m2_gbt_model": None,
    "m3_als_model": None,
}

@app.on_event("startup")
def load_models_on_startup():
    logging.info("API starting up. Loading SparkSession and models...")
    try:
        app_state["spark"] = (
            SparkSession.builder
            .appName("FastAPI_Spark_Serving")
            .config("spark.driver.memory", "512m")
            .config("spark.executor.memory", "512m")
            .master("local[1]")
            .getOrCreate()
        )
        app_state["spark"].sparkContext.setLogLevel("WARN")
        logging.info("SparkSession created for API.")

        # Coba muat model. Perlu path yang benar dan model ada.
        try:
            app_state["m1_lr_model"] = PipelineModel.load(M1_LR_PATH)
            logging.info(f"M1_LR model loaded successfully from {M1_LR_PATH}")
        except Exception as e:
            logging.warning(f"Could not load M1_LR model: {e}. Endpoint will not work.")

        try:
            app_state["m2_gbt_model"] = PipelineModel.load(M2_GBT_PATH)
            logging.info(f"M2_GBT model loaded successfully from {M2_GBT_PATH}")
        except Exception as e:
            logging.warning(f"Could not load M2_GBT model: {e}. Endpoint will not work.")

        try:
            app_state["m3_als_model"] = ALSModel.load(M3_ALS_PATH)
            logging.info(f"M3_ALS model loaded successfully from {M3_ALS_PATH}")
        except Exception as e:
            logging.warning(f"Could not load M3_ALS model: {e}. Endpoint will not work.")
            
    except Exception as e:
        logging.error(f"Fatal error during API startup (Spark or model loading): {e}")
        # Coba hentikan aplikasi jika Spark gagal dimulai
        # Ini akan menyebabkan kontainer restart jika dikonfigurasi demikian
        raise RuntimeError(f"API startup failed: {e}")


@app.on_event("shutdown")
def shutdown_event():
    if app_state["spark"]:
        app_state["spark"].stop()
        logging.info("SparkSession stopped on API shutdown.")

# ----- Pydantic Models for Request/Response -----
class FeaturesRequest(BaseModel):
    # Enhanced features for Phase 1 models
    features: Dict[str, float]  # Example: {"hours": 5.5, "price": 19.99, "positive_ratio": 85.0}

class ProbabilityResponse(BaseModel):
    probability: float # Probability of recommendation (0.0 to 1.0)

class RecommendationItem(BaseModel):
    app_id: int
    score: float

# ----- Helper untuk LR/GBT Prediksi -----
def predict_classification(model, features_dict: Dict[str, float], spark_session):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    # Check required features
    required_features = ["hours"]
    for feature in required_features:
        if feature not in features_dict:
            raise HTTPException(status_code=400, detail=f"Missing '{feature}' in features.")

    try:
        # Create enhanced features with defaults for missing optional features
        hours = features_dict["hours"]
        price = features_dict.get("price", 15.0)  # Default to median price
        positive_ratio = features_dict.get("positive_ratio", 75.0)  # Default to reasonable rating
        user_reviews = features_dict.get("user_reviews", 100.0)  # Default review count
        
        # Calculate derived features (same as training)
        price_category = 1.0 if price <= 9.99 else (2.0 if price <= 29.99 else 3.0)
        quality_score = positive_ratio * (user_reviews / 1000.0)
        
        # Create DataFrame with enhanced feature schema
        schema = StructType([
            StructField("hours_numeric", FloatType(), True),
            StructField("price_numeric", FloatType(), True),
            StructField("positive_ratio_numeric", FloatType(), True),
            StructField("quality_score", FloatType(), True),
            StructField("price_category", FloatType(), True)
        ])
        
        data_for_prediction = [(hours, price, positive_ratio, quality_score, price_category)]
        
        df_to_predict = spark_session.createDataFrame(data_for_prediction, schema=schema)

        prediction = model.transform(df_to_predict)
        
        # Coba ambil probabilitas kelas 1 ([1]) dari kolom 'probability' (Vector)
        prob_class_1 = prediction.select("probability").first()[0][1]
        return float(prob_class_1)
        
    except Exception as e:
        logging.error(f"Error during classification prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ----- API Endpoints -----
@app.post("/predict/like", response_model=ProbabilityResponse)
async def predict_like_lr(request: FeaturesRequest):
    """Prediksi 'like' menggunakan model M1_LR (Logistic Regression)."""
    logging.info(f"Received /predict/like request: {request.features}")
    if not app_state["m1_lr_model"]:
        raise HTTPException(status_code=503, detail="M1_LR model is not available.")
    
    prob = predict_classification(app_state["m1_lr_model"], request.features, app_state["spark"])
    return ProbabilityResponse(probability=prob)

@app.post("/predict/like_gbt", response_model=ProbabilityResponse)
async def predict_like_gbt(request: FeaturesRequest):
    """Prediksi 'like' menggunakan model M2_GBT (GBTClassifier)."""
    logging.info(f"Received /predict/like_gbt request: {request.features}")
    if not app_state["m2_gbt_model"]:
        raise HTTPException(status_code=503, detail="M2_GBT model is not available.")

    prob = predict_classification(app_state["m2_gbt_model"], request.features, app_state["spark"])
    return ProbabilityResponse(probability=prob)

@app.get("/recommend/{user_id}", response_model=List[RecommendationItem])
async def recommend_for_user(user_id: int):
    """Dapatkan rekomendasi game untuk user_id menggunakan model M3_ALS."""
    logging.info(f"Received /recommend/{user_id} request")
    if not app_state["m3_als_model"]:
        raise HTTPException(status_code=503, detail="M3_ALS model is not available.")
    if not app_state["spark"]:
        raise HTTPException(status_code=503, detail="SparkSession not available for ALS.")

    try:
        # Coba dapatkan top N rekomendasi (misal, 10)
        num_recommendations = 10
        user_recs_df = app_state["m3_als_model"].recommendForAllUsers(num_recommendations)
        
        # Coba filter untuk user_id tertentu
        # Kolom user di ALS adalah 'user' (user_id_int di data asli)
        specific_user_recs = user_recs_df.filter(col("user") == user_id).select("recommendations").first()

        if not specific_user_recs or not specific_user_recs[0]:
            logging.info(f"No recommendations found for user_id {user_id}.")
            return []

        recommendations = []
        for row in specific_user_recs[0]: # recommendations adalah array struct(item, rating)
            recommendations.append(RecommendationItem(app_id=row['item'], score=row['rating']))
        
        logging.info(f"Returning {len(recommendations)} recommendations for user_id {user_id}")
        return recommendations
        
    except Exception as e:
        # Coba cek apakah user_id tidak ada di model (cold start)
        if "User " + str(user_id) + " does not exist in training data" in str(e) or \
           "NoSuchElementException" in str(e): # Spark bisa melempar ini jika user tidak ditemukan
            logging.warning(f"User ID {user_id} not found in ALS model (cold start). Returning empty list. Error: {e}")
            return []
        logging.error(f"Error during ALS recommendation for user_id {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@app.get("/health")
async def health_check():
    # Coba periksa status model dan SparkSession
    status = {
        "spark_session_active": app_state["spark"] is not None and \
                                hasattr(app_state["spark"], '_jsc') and \
                                app_state["spark"]._jsc is not None,
        "m1_lr_model_loaded": app_state["m1_lr_model"] is not None,
        "m2_gbt_model_loaded": app_state["m2_gbt_model"] is not None,
        "m3_als_model_loaded": app_state["m3_als_model"] is not None,
    }
    all_ok = all(status.values())
    return {"status": "healthy" if all_ok else "degraded", "details": status}

