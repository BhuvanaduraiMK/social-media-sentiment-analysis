from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, regexp_replace
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler

# Create Spark Session
spark = SparkSession.builder.appName("TwitterSentimentAnalysis").getOrCreate()

# Define correct column names manually
columns = ["tweet_id", "category", "sentiment", "tweet_text"]

# Load CSV dataset with correct column names
df = spark.read.csv("/opt/bitnami/spark/twitter_training.csv", header=False, inferSchema=True)

# Assign column names
df = df.toDF(*columns)

# Show first 5 rows
df.show(5)

# Print schema
df.printSchema()

# Total tweet count
print("Total Tweets:", df.count())

# Display unique sentiments
df.select("sentiment").distinct().show()

df.groupBy("sentiment").count().show()

df.filter(df.sentiment == "Positive").show(5)



df = df.withColumn("clean_tweet", lower(regexp_replace(df.tweet_text, "[^a-zA-Z0-9 ]", "")))
df.select("tweet_text", "clean_tweet").show(5)



tokenizer = Tokenizer(inputCol="clean_tweet", outputCol="words")
# Replace null values with empty string
df = df.withColumn("clean_tweet", when(col("clean_tweet").isNull(), "").otherwise(col("clean_tweet")))

# Apply Tokenizer
df = tokenizer.transform(df)
df.select("clean_tweet", "words").show(5, truncate=False)

# Convert words column into numerical features using HashingTF
hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=1000)
df = hashingTF.transform(df)

# Apply IDF to scale the word frequencies
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df)
df = idf_model.transform(df)

# Show the processed dataset
df.select("words", "features").show(5, truncate=False)

# Convert sentiment labels into numerical labels
indexer = StringIndexer(inputCol="sentiment", outputCol="label")
df = indexer.fit(df).transform(df)

# Show labeled data
df.select("sentiment", "label").distinct().show()

# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# from pyspark.ml.feature import VectorAssembler

# Assemble features
assembler = VectorAssembler(inputCols=["features"], outputCol="final_features")
df = assembler.transform(df)

# Split the data into training (80%) and test (20%)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train the Logistic Regression model
lr = LogisticRegression(featuresCol="final_features", labelCol="label", maxIter=10)
model = lr.fit(train_data)

# Predictions on test data
predictions = model.transform(test_data)

# Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Model Accuracy: {accuracy:.2f}")
