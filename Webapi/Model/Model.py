from .mongoRetreive import retriveData
import pandas as pd
from pyspark.sql import dataframe as pysparkDataFrame
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.serializers import AutoBatchedSerializer, PickleSerializer
import nltk
from nltk.corpus import stopwords
from pyspark.conf import SparkConf
from pyspark import SparkContext# from mlflow.pyspark.ml import autolog
import os
conf = SparkConf().set("spark.sql.broadcastTimeout", "36000")  # Increase timeout value to 10 hours (36000 seconds)
sc = SparkContext.getOrCreate()
numCores = sc.defaultParallelism
print("number of cores", numCores)

spark = SparkSession.builder \
    .master('"spark://spark-master:7077"') \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.extraJavaOptions", "-XX:+PrintGCDetails -XX:+PrintGCTimeStamps") \
    .appName('my-cool-app') \
    .getOrCreate()

def getData():
    df = retriveData()
    return spark.createDataFrame(df)

def dataPipeline():
    regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
    stoppingWords = ["http","https","amp","rt","t","c","the"] 
    stoppingWordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(stoppingWords)
    countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=0)
    stringIndexing = StringIndexer(inputCol = "category", outputCol = "label")
    pipeline = Pipeline(stages=[regexTokenizer, stoppingWordsRemover, countVectors, stringIndexing])
    return pipeline

def trainModel():
    dataFrame = getData()
    pipeline = dataPipeline()
    pipelineFit = pipeline.fit(dataFrame)
    dataSet = pipelineFit.transform(dataFrame)
    mappedLabels = mapLabelandTopics(dataSet)
    (trainingData, testData) = dataSet.randomSplit([0.7, 0.3], seed = 100)
    logisticRegression = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=15, regParam=0.3, elasticNetParam=0)
    LRModel = logisticRegression.fit(trainingData)
    evaluate(LRModel, testData)
    return LRModel, pipelineFit, mappedLabels

def evaluate(LRModel, testData):
    predictions = LRModel.transform(testData)
    evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction")
    evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    evaluatorRecall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    evaluatorF1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    accuracy = evaluatorAccuracy.evaluate(predictions)
    precision = evaluatorPrecision.evaluate(predictions)
    recall = evaluatorRecall.evaluate(predictions)
    f1Score = evaluatorF1.evaluate(predictions)
    with open("./results.txt", 'w') as f:
        f.write(",".join([ str(i) for i in [accuracy, precision, recall, f1Score]]))

    print(accuracy, precision, recall, f1Score)
    
def predictTopic(Model, pipelineFit, params):
    params = spark.createDataFrame(params)
    testData = pipelineFit.transform(params)
    LRPredictions = Model.transform(testData)
    return LRPredictions.collect()[-1]['prediction']

def validate(LRModel, trainingData, testData):
    paramGrid = (ParamGridBuilder()
             .addGrid(LRModel.regParam, [0.1, 0.3, 0.5]) # regularization parameter
             .addGrid(LRModel.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
             .build())

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    cv = CrossValidator(estimator=LRModel, \
                        estimatorParamMaps=paramGrid, \
                        evaluator=evaluator, \
                        numFolds=5)
    cvModel = cv.fit(trainingData)
    predictions = cvModel.transform(testData)
    evaluator.evaluate(predictions)

def mapLabelandTopics(dataSet):
    labels = []
    Mappedlabels = {}
    for data in dataSet.collect():
        if(data['label'] not in labels):
            labels.append(data['label'])
            Mappedlabels[int(data['label'])]= data['category']
    return Mappedlabels