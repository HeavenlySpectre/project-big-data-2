# Tugas Big Data

Sistem ini dibuat untuk mensimulasikan stream dari data proses untuk model Machine Learning hingga deployment API. Didesain khusus untuk Sistem Rekomendasi Game Steam.

Dikerjakan Oleh:
- Athalla Barka Fadhil - 5027231018
- Kevin Anugerah Faza - 5027231027
- Azza Farichi Tjahjono - 5027231071

## Tech Stack
- Apache Kafka (data streaming)
- Apache Spark (data processing & ML training)
- Docker (containerization)
- FastAPI (model serving)
- Apache Hadoop (distributed storage)

## Alur Sistem

Dataset → Kafka Producer → Kafka Server → Kafka Consumer → Batch Files → Spark Script → ML Models → API → User Responses

* **Dataset $\rightarrow$ Pembacaan Sekuensial oleh Kafka Producer dari Data Game Steam (File CSV)**
    * Proses dimulai dengan **Kafka Producer** yang membaca dataset game Steam secara berurutan dari file-file CSV yang tersedia.
* **Kafka Producer $\rightarrow$ Mengalirkan Data Baris demi Baris dengan Penundaan Acak dari `realrecommendations.csv`**
    * **Kafka Producer** mengirimkan data ke Kafka Server baris demi baris, khususnya dari file `realrecommendations.csv`. Setiap pengiriman data akan diberikan jeda waktu acak untuk mensimulasikan aliran data *streaming* yang realistis.
* **Kafka Server $\rightarrow$ Menerima dan Menyimpan Data Streaming secara Berkelanjutan di Topik `game.reviews`**
    * **Kafka Server** berfungsi sebagai *broker* pusat yang menerima dan menyimpan data *streaming* yang dikirimkan oleh Kafka Producer. Data ini akan disimpan dalam sebuah topik bernama `game.reviews`.
* **Kafka Consumer $\rightarrow$ Membaca Data dan Membuat Batch di Volume `/data`**
    * **Kafka Consumer** bertugas membaca data yang terus-menerus mengalir dari Kafka Server. Data ini kemudian akan dikelompokkan menjadi *batch-batch* dan disimpan di dalam volume `/data`.
* **File Batch $\rightarrow$ Disimpan sebagai File Parquet Berdasarkan Jumlah/Jendela Waktu**
    * *Batch-batch* data yang telah dibuat oleh Kafka Consumer akan disimpan sebagai file-file berformat **Parquet**. Penentuan ukuran *batch* ini dapat berdasarkan jumlah data yang diterima atau berdasarkan rentang waktu tertentu (jendela waktu).
* **Spark Script $\rightarrow$ Melatih 3 Model ML dari Batch dengan Fitur yang Ditingkatkan**
    * **Spark Script** akan mengambil *batch-batch* data yang disimpan, yang telah diperkaya dengan fitur-fitur tambahan. Script ini akan melatih **tiga model *Machine Learning*** yang berbeda.
* **Model ML $\rightarrow$ Dihasilkan per Batch/Segmen Data (M1_LR, M2_GBT, M3_ALS)**
    * Model-model *Machine Learning* akan dihasilkan berdasarkan setiap *batch* atau segmen data yang diproses. Tiga model yang spesifik adalah **M1_LR (Logistic Regression), M2_GBT (Gradient-Boosted Trees), dan M3_ALS (Alternating Least Squares)**.
* **API $\rightarrow$ Menyajikan Prediksi Model melalui *Endpoint* FastAPI**
    * Terakhir, sebuah **API** yang dibangun menggunakan **FastAPI** akan menyediakan *endpoint* untuk menyajikan prediksi dari model-model ML yang telah dilatih kepada pengguna.

## Dataset yang digunakan

### Ringkasan Dataset: Game Recommendations on Steam, A dataset of games, users and reviews for building recommendation systems

Dataset ini berisi lebih dari **41 juta rekomendasi pengguna (ulasan) yang telah dibersihkan dan diproses sebelumnya** dari Steam Store, sebuah platform online terkemuka untuk membeli dan mengunduh video game, DLC, dan konten terkait game lainnya. Selain itu, dataset ini juga mengandung informasi rinci tentang game dan *add-on*.

### Konten

Dataset ini terdiri dari **tiga entitas utama**:

* **`games.csv`**: Sebuah tabel berisi informasi game (atau *add-on*) seperti peringkat, harga dalam Dolar AS ($), tanggal rilis, dan lain-lain. Detail non-tabular tambahan tentang game, seperti deskripsi dan tag, ada dalam **file metadata**.
* **`users.csv`**: Sebuah tabel berisi informasi publik profil pengguna: jumlah produk yang dibeli dan ulasan yang dipublikasikan.
* **`recommendations.csv`**: Sebuah tabel ulasan pengguna: apakah pengguna merekomendasikan suatu produk atau tidak. Tabel ini merepresentasikan hubungan banyak-ke-banyak antara entitas game dan entitas pengguna.

Dataset ini **tidak mengandung informasi pribadi pengguna** di Platform Steam. Semua ID pengguna telah dianonimkan melalui *pipeline* prapemrosesan. Semua data yang dikumpulkan dapat diakses oleh masyarakat umum.

Dataset ini dikumpulkan dari **Steam Official Store**. Semua hak atas gambar *thumbnail* dataset adalah milik **Valve Corporation**.

Dataset ini menggunakan lisensi [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

### Sumber Resmi

Dataset ini kami dapatkan melalui Kaggle.
[Link Dataset](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam/data)

## Cara Menjalankan Sistem

### Pre-Requisites

- Docker Desktop (Windows/Mac) atau Docker Engine (Linux)
Docker Compose v2.0+
- RAM minimal 8GB (direkomendasikan 16GB)
- Storage kosong minimal 10GB
- Port bebas: 8000, 8080, 8081, 9092, 2181, 9000, 9864, 8088

### Langkah - Langkah

1. Jalankan seluruh sistem

```
docker-compose up -d
```

2. Monitoring proses training dan API (tidak perlu dilakukan jika menggunakan Docker Desktop)

```
docker-compose logs -f spark-trainer

docker-compose logs -f api
```
3. Jika container API berhenti sebelum training selesai, jalankan kembali menggunakan command
```
docker-compose restart api
```

### Cara membaca hasil Machine Learning

1. Untuk model Logistic Regression dan GBT Classifier, semakin tinggi angkanya (mendekati 1) maka akan semakin direkomendasikan.

2. Untuk model ALS, dia akan memberikan output berupa 10 rekomendasi game yang ditujukan kepada pengguna terkait berdasarkan ID Pengguna.

## Dokumentasi

![Logistic Regression](/img/LRmodel.png)

![GBTClassifier](/img/GBTClassifiermodel.png)

![ALSmodel](/img/ALSmodel.png)



   
