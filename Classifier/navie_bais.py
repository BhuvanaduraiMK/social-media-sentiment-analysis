from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace, col, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# ✅ 1. Create Spark Session
spark = SparkSession.builder.appName("TwitterSentimentAnalysis").getOrCreate()

# ✅ 2. Load & Assign Column Names
columns = ["tweet_id", "category", "sentiment", "tweet_text"]
df = spark.read.csv("/opt/bitnami/spark/twitter_training.csv", header=False, inferSchema=True).toDF(*columns)

# ✅ 3. Preprocessing: Convert to Lowercase & Remove Special Characters
df = df.withColumn("clean_tweet", lower(regexp_replace(df.tweet_text, "[^a-zA-Z0-9 ]", "")))

# ✅ 4. Tokenization
# ✅ Handle null values before tokenization
df = df.withColumn("clean_tweet", when(col("clean_tweet").isNull(), "").otherwise(col("clean_tweet")))

# ✅ Tokenization
tokenizer = Tokenizer(inputCol="clean_tweet", outputCol="words")
df = tokenizer.transform(df)


# ✅ 5. Stopword Removal
stopword_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df = stopword_remover.transform(df)

# ✅ 6. Feature Extraction: TF-IDF with numFeatures=5000
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=5000)
df = hashingTF.transform(df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df)
df = idf_model.transform(df)

# ✅ 7. Convert Sentiment Labels to Numeric
indexer = StringIndexer(inputCol="sentiment", outputCol="label")
df = indexer.fit(df).transform(df)

# ✅ 8. Balance Dataset (Oversampling)
majority_class = df.filter(df.label == 0)
minority_classes = df.filter(df.label != 0)

# Oversample minority classes
balanced_df = majority_class.union(minority_classes.sample(withReplacement=True, fraction=2.0, seed=42))

# ✅ 9. Assemble Features for Model Training
assembler = VectorAssembler(inputCols=["features"], outputCol="final_features")
df = assembler.transform(balanced_df)

# ✅ 10. Train-Test Split (80-20)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# ✅ 11. Naïve Bayes Model Training
nb = NaiveBayes(featuresCol="final_features", labelCol="label", smoothing=1.0)
model = nb.fit(train_data)

# ✅ 12. Predictions on Test Data
predictions = model.transform(test_data)

# ✅ 13. Evaluate Model Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Model Accuracy: {accuracy:.2f}")

# ✅ 14. Hyperparameter Tuning with CrossValidator
param_grid = ParamGridBuilder().addGrid(nb.smoothing, [0.5, 1.0, 1.5]).build()
cross_val = CrossValidator(estimator=nb, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
cv_model = cross_val.fit(train_data)

# ✅ 15. Evaluate Best Model
cv_predictions = cv_model.transform(test_data)
cv_accuracy = evaluator.evaluate(cv_predictions)

print(f"Best Model Accuracy After Tuning: {cv_accuracy:.2f}")
