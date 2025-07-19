from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TwitterSentimentAnalysis").getOrCreate()

print(spark)
    