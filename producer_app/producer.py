# Coba untuk membaca CSV dan mengirim pesan ke Kafka
import csv
import json
import time
import random
import argparse
import logging
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_producer(bootstrap_servers):
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            linger_ms=10, # Coba untuk batch kecil sebelum mengirim
            batch_size=16384 * 2, # Coba 32KB batch size
            retries=3,
            acks='all' # Coba untuk memastikan pesan terkirim
        )
        logging.info("KafkaProducer created successfully.")
        return producer
    except Exception as e:
        logging.error(f"Error creating Kafka producer: {e}")
        raise

def read_and_send(producer, topic, csv_file_path, sleep_range_ms):
    logging.info(f"Starting to read from {csv_file_path} and send to topic '{topic}'")
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            if not fieldnames:
                logging.error("CSV file is empty or header is missing.")
                return
            logging.info(f"CSV Headers: {fieldnames}")
            
            count = 0
            for row in reader:
                try:
                    # Coba untuk membersihkan dan konversi data dasar jika perlu
                    # Contoh: 'app_id, user_id, is_recommended, hours, date'
                    message = {
                        "app_id": row.get("app_id"),
                        "user_id": row.get("user_id"),
                        "is_recommended": row.get("is_recommended"),
                        "hours": row.get("hours"),
                        "date": row.get("date")
                    }
                    producer.send(topic, value=message)
                    count += 1
                    if count % 1000 == 0: # Coba log setiap 1000 pesan
                        logging.info(f"Sent {count} messages to Kafka topic '{topic}'. Last message: {message}")
                        producer.flush() # Coba flush secara periodik

                    sleep_ms = random.randint(sleep_range_ms[0], sleep_range_ms[1])
                    time.sleep(sleep_ms / 1000.0)

                except Exception as e:
                    logging.error(f"Error processing/sending row: {row}. Error: {e}")
            
            producer.flush() # Coba flush sisa pesan
            logging.info(f"Finished sending {count} messages from {csv_file_path} to topic '{topic}'.")

    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_file_path}")
    except Exception as e:
        logging.error(f"An error occurred during file reading or Kafka sending: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka CSV Producer")
    parser.add_argument("--file", default="/datasets_to_read/realrecommendations.csv", help="Path to the CSV file")
    parser.add_argument("--topic", default="game.reviews", help="Kafka topic to send messages to")
    parser.add_argument("--bootstrap_servers", default="kafka:9093", help="Kafka bootstrap servers")
    parser.add_argument("--sleep_min_ms", type=int, default=10, help="Minimum sleep time in ms")
    parser.add_argument("--sleep_max_ms", type=int, default=200, help="Maximum sleep time in ms")

    args = parser.parse_args()

    logging.info(f"Producer starting with config: {args}")

    kafka_producer = None
    try:
        kafka_producer = create_producer(args.bootstrap_servers.split(','))
        read_and_send(kafka_producer, args.topic, args.file, (args.sleep_min_ms, args.sleep_max_ms))
    except Exception as e:
        logging.critical(f"Producer failed to start or run: {e}")
    finally:
        if kafka_producer:
            kafka_producer.close()
            logging.info("Kafka producer closed.")