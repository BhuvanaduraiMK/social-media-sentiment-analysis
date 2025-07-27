# Scalable Data Processing for Social Media Sentiment Analysis

This project demonstrates how to perform large-scale sentiment analysis on Twitter data using Apache Spark with Docker, PySpark, and machine learning models. It is designed to be scalable and capable of processing large datasets efficiently.

---

## 📌 Project Objectives

1. **Data Collection** – Used Kaggle Twitter dataset (Sentiment140)
2. **Scalable Processing** – Performed using Apache Spark with Docker
3. **Data Storage** – In-memory processing using Spark DataFrames
4. **Data Preprocessing** – Cleaning, tokenization using NLTK
5. **Sentiment Analysis** – Applied ML models to classify tweets as positive/negative
6. **Result Evaluation** – Compared accuracy of multiple models

---

## 🧰 Technologies Used

- Python 3.x
- Apache Spark (via Bitnami Docker image)
- PySpark
- scikit-learn
- NLTK
- pandas
- Docker Engine
- Jupyter Notebook / Python Scripts

---

## 🗃️ Dataset

Used **Sentiment140 Twitter dataset** from Kaggle:  
🔗 [https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

> ⚠️ Dataset file (`training.csv`) not included in this repo. Please download manually from Kaggle and place it in your working directory.

---

## 📁 Project Structure
social-media-sentiment-analysis/
│
├── classifiers/
│ ├── logistic_regression.py
│ ├── naive_bayes.py
│ ├── rf_classifier.py
│ ├── gbt_classifier.py
│
├── data/
│ └── training.csv (local only)
│
├── load_data.py
├── requirements.txt
├── .gitignore
├── spark_docker_notes.txt
└── README.md

---

## ⚙️ How to Run the Project (using Docker + Bitnami Spark)

> ⚠️ You must have Docker installed and running.

### 🧰 Step 1: Download and Prepare Dataset

1. Download the file `training.csv` from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).
2. Move it into the `data/` folder inside your project directory.

---

### 🐳 Step 2: Run Spark Container with Docker


docker run -it --rm \
  -v $(pwd):/auth/bitnami/spark \
  bitnami/spark pyspark

  #-v C:\Users\Bhuvana\Desktop\social-media-sentiment-analysis:/auth/bitnami/spark

# Once inside the Spark container shell
python /auth/bitnami/spark/load_data.py
python /auth/bitnami/spark/classifiers/logistic_regression.py
python /auth/bitnami/spark/classifiers/naive_bayes.py
python /auth/bitnami/spark/classifiers/rf_classifier.py


| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 0.67     |
| Naive Bayes         | 0.71     |
| Random Forest       | 0.91     |


🙋 Author
Bhuvanadurai M.
Final Year B.E (CSE - Data Science)
Annamalai University
GitHub: https://github.com/BhuvanaduraiMK
LinkedIn:https://www.linkedin.com/in/bhuvanadurai-m-1312a7248/
