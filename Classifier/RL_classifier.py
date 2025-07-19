from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace, col, when
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create Spark Session with Higher Memory
spark = SparkSession.builder \
    .appName("BetterSentimentAnalysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Load Dataset
columns = ["tweet_id", "category", "sentiment", "tweet_text"]
df = spark.read.csv("/opt/bitnami/spark/twitter_training.csv", header=False, inferSchema=True)
df = df.toDF(*columns)

# Text Preprocessing
df = df.withColumn("clean_tweet", lower(regexp_replace(df.tweet_text, "[^a-zA-Z0-9 ]", "")))

# Tokenization
tokenizer = Tokenizer(inputCol="clean_tweet", outputCol="words")
df = df.withColumn("clean_tweet", when(col("clean_tweet").isNull(), "").otherwise(col("clean_tweet")))
df = tokenizer.transform(df)

# Convert text to numerical features (Increase Features to 5000)
hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=5000)
df = hashingTF.transform(df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df)
df = idf_model.transform(df)

# Convert labels into numerical form
indexer = StringIndexer(inputCol="sentiment", outputCol="label")
df = indexer.fit(df).transform(df)

# Assemble Features
assembler = VectorAssembler(inputCols=["features"], outputCol="final_features")
df = assembler.transform(df)

# Cache Data
df = df.persist()

# Split Data (80% Train, 20% Test)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Logistic Regression (Better for Text Data)
lr = LogisticRegression(featuresCol="final_features", labelCol="label", maxIter=20, regParam=0.01)

# Hyperparameter Tuning (Optimize `regParam`)
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .build()

crossval = CrossValidator(estimator=lr,estimatorParamMaps=paramGrid,evaluator=MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy"),numFolds=2)  

# Train the Model
cv_model = crossval.fit(train_data)

# Make Predictions
predictions = cv_model.transform(test_data)

#Evaluate Model Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f" Improved Model Accuracy: {accuracy:.2f}")
