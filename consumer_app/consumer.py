# Coba untuk membaca dari Kafka dan menyimpan sebagai Parquet
import json
import time
import os
import uuid
import argparse
import logging
from kafka import KafkaConsumer
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Coba tentukan skema dan tipe data dasar untuk DataFrame
EXPECTED_COLUMNS = ['app_id', 'user_id', 'is_recommended', 'hours', 'date']
COLUMN_DTYPES = {
    'app_id': 'str', # Coba untuk menjaga sebagai string jika ada non-numerik, Spark bisa mengkonversi nanti
    'user_id': 'str',
    'is_recommended': 'str', # Coba sebagai string, Spark akan mengkonversi ke int/float
    'hours': 'str',          # Coba sebagai string, Spark akan mengkonversi ke float
    'date': 'str'
}


def create_consumer(topic, bootstrap_servers, group_id="parquet_consumer_group"):
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='earliest', # Coba untuk mulai dari awal jika grup baru
            enable_auto_commit=True,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            consumer_timeout_ms=10000 # Coba timeout untuk melepaskan loop jika tidak ada pesan
        )
        logging.info(f"KafkaConsumer created for topic '{topic}' and group '{group_id}'.")
        return consumer
    except Exception as e:
        logging.error(f"Error creating Kafka consumer: {e}")
        raise

def flush_buffer_to_parquet(buffer, output_path):
    if not buffer:
        return
    
    try:
        df = pd.DataFrame(buffer)
        
        # Coba pastikan semua kolom yang diharapkan ada, isi dengan None jika tidak
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = None
        
        # Coba urutkan kolom dan konversi tipe
        df = df[EXPECTED_COLUMNS]
        for col, dtype in COLUMN_DTYPES.items():
            try:
                if dtype == 'float64_safe': # Coba konversi aman ke float
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                logging.warning(f"Could not convert column {col} to {dtype}: {e}. Leaving as object.")

        # Coba buat nama file Parquet yang unik
        timestamp = time.strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4().hex)[:8]
        file_name = f"batch_{timestamp}_{unique_id}.parquet"
        file_path = os.path.join(output_path, file_name)
        
        os.makedirs(output_path, exist_ok=True) # Coba pastikan direktori output ada
        df.to_parquet(file_path, index=False, engine='pyarrow')
        logging.info(f"Flushed {len(buffer)} records to {file_path}")
    except Exception as e:
        logging.error(f"Error flushing buffer to Parquet: {e}")
    finally:
        buffer.clear()

def consume_messages(consumer, output_path, batch_size_rows, batch_timeout_min):
    buffer = []
    last_flush_time = time.time()
    
    logging.info(f"Starting consumption. Batch rows: {batch_size_rows}, Batch timeout: {batch_timeout_min} min.")

    try:
        while True: # Coba loop terus menerus, akan dihentikan oleh Docker jika perlu
            for message in consumer: # consumer_timeout_ms akan melepaskan loop ini jika tidak ada pesan
                if message and message.value:
                    # Coba pastikan message.value adalah dict
                    if isinstance(message.value, dict):
                        buffer.append(message.value)
                    else:
                        logging.warning(f"Received non-dict message: {message.value}")
                
                current_time = time.time()
                time_elapsed_seconds = current_time - last_flush_time
                
                if len(buffer) >= batch_size_rows or (buffer and time_elapsed_seconds >= batch_timeout_min * 60):
                    logging.info(f"Flush condition met: {len(buffer)} rows or {time_elapsed_seconds/60:.2f} mins elapsed.")
                    flush_buffer_to_parquet(buffer, output_path) # buffer dikosongkan di dalam fungsi ini
                    last_flush_time = current_time
            
            # Coba cek flush timeout bahkan jika tidak ada pesan baru (jika buffer tidak kosong)
            current_time = time.time()
            if buffer and (current_time - last_flush_time >= batch_timeout_min * 60):
                logging.info(f"Flush condition met (timeout with non-empty buffer): {len(buffer)} rows.")
                flush_buffer_to_parquet(buffer, output_path)
                last_flush_time = current_time
            
            logging.debug("Consumer loop iteration finished. Waiting for messages or timeout.")
            # Coba beri jeda singkat jika tidak ada pesan untuk menghindari CPU spinning ketat
            if not buffer : time.sleep(1)


    except KeyboardInterrupt:
        logging.info("Consumer process interrupted by user.")
    except Exception as e:
        logging.error(f"An error occurred during Kafka consumption: {e}")
    finally:
        # Coba flush sisa buffer saat keluar
        if buffer:
            logging.info(f"Flushing remaining {len(buffer)} records before exiting.")
            flush_buffer_to_parquet(buffer, output_path)
        if consumer:
            consumer.close()
            logging.info("Kafka consumer closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Parquet Consumer")
    parser.add_argument("--topic", default="game.reviews", help="Kafka topic to consume from")
    parser.add_argument("--bootstrap_servers", default="kafka:9093", help="Kafka bootstrap servers")
    parser.add_argument("--output_path", default="/data", help="Output directory for Parquet files") # Akan menjadi /data/batch_*.parquet
    parser.add_argument("--batch_rows", type=int, default=500000, help="Number of rows per Parquet file")
    parser.add_argument("--batch_timeout_min", type=int, default=5, help="Max time in minutes before flushing")

    args = parser.parse_args()
    logging.info(f"Consumer starting with config: {args}")

    kafka_consumer = None
    try:
        kafka_consumer = create_consumer(args.topic, args.bootstrap_servers.split(','))
        consume_messages(kafka_consumer, args.output_path, args.batch_rows, args.batch_timeout_min)
    except Exception as e:
        logging.critical(f"Consumer failed to start or run: {e}")