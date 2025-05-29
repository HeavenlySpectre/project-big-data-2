# spark_app/Dockerfile
FROM bde2020/spark-submit:3.3.0-hadoop3.3

USER root

# Install Python3 tools + dos2unix
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends python3-pip dos2unix \
 && rm -rf /var/lib/apt/lists/*

# Install numpy for pyspark.ml
RUN pip3 install --no-cache-dir numpy

WORKDIR /app

# Copy and normalize line endings
COPY spark_job.py /app/
RUN dos2unix /app/spark_job.py

# Use the JSON array form—each element is a string, no backslashes:
CMD ["spark-submit", \
     "--master", "spark://spark-master:7077", \
     "/app/spark_job.py", \
     "--input", "/input_parquet_data", \
     "--output", "/models_output"]
