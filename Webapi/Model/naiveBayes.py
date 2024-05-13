from .mongoRetreive import retriveData
import pandas as pd
from pyspark.sql import dataframe as pysparkDataFrame
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import nltk
from nltk.corpus import stopwords
from pyspark.conf import SparkConf
from pyspark import SparkContext
from itertools import product
nltk.download('stopwords')
conf = SparkConf().set("spark.sql.broadcastTimeout", "36000") 
sc = SparkContext.getOrCreate()
numCores = sc.defaultParallelism
print("number of cores", numCores)

spark = SparkSession.builder \
    .master('"spark://spark-master:7077"') \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.extraJavaOptions", "-XX:+PrintGCDetails -XX:+PrintGCTimeStamps") \
    .appName('my-cool-app') \
    .getOrCreate()

hyperParametersRanges = {
    'smoothing': [0.0, 0.5, 1.0, 2.0],
    'modelType': ['multinomial', 'bernoulli', 'complement'],
}



paramGrids = [dict(zip(hyperParametersRanges.keys(), values))
              for values in product(*hyperParametersRanges.values())]

def GetDataSet(fromFile: bool = False, fileName: str = "8CatNewsData.csv") -> pd.DataFrame:
    print("Pulling Data")
    if (fromFile):
        dataFrame = pd.read_csv(fileName)
    else:
        dataFrame = retriveData()
    print("Data pulled")
    dataFrame = dataFrame.dropna()
    return dataFrame

def ConvertTextToLowerCase(text: str) -> str:
    return text.lower()

def DataPipeline():
    regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
    stoppingWords = stopwords.words('english')
    stoppingWords = [ConvertTextToLowerCase(
        stoppingWord) for stoppingWord in stoppingWords] + ["http","https","amp","rt","t","c","the"] 
    stoppingWordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(stoppingWords)
    countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=0)
    stringIndexing = StringIndexer(inputCol = "category", outputCol = "label")
    pipeline = Pipeline(stages=[regexTokenizer, stoppingWordsRemover, countVectors, stringIndexing])
    return pipeline

def MapLabelandTopics(dataSet):
    labels = []
    mappedlabels = {}
    for data in dataSet.collect():
        if(data['label'] not in labels):
            labels.append(data['label'])
            mappedlabels[int(data['label'])]= data['category']
    return mappedlabels

def evaluate(model, testData):
    predictions = model.transform(testData)
    evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction")
    evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    evaluatorRecall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    evaluatorF1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    accuracy = evaluatorAccuracy.evaluate(predictions)
    precision = evaluatorPrecision.evaluate(predictions)
    recall = evaluatorRecall.evaluate(predictions)
    f1Score = evaluatorF1.evaluate(predictions)
    with open("./results2.txt", 'w') as f:
        f.write(",".join([ str(i) for i in [accuracy, precision, recall, f1Score]]))

    return [accuracy, precision, recall, f1Score]

def TrainModel(dataSet,paramGrid ):
    (trainingData, testData) = dataSet.randomSplit([0.7, 0.2], seed = 42)
    logisticRegression = NaiveBayes(**paramGrid)
    model = logisticRegression.fit(trainingData)
    [accuracy, precision, recall, f1Score] = evaluate(model, testData)
    return model, [accuracy, precision, recall, f1Score]

def FindBestNBModel(paramGrids= paramGrids):
    dataSet = GetDataSet(fromFile=False)
    dataSet = spark.createDataFrame(dataSet)
    dataPipeline = DataPipeline()
    pipelineFit = dataPipeline.fit(dataSet)
    dataSet = pipelineFit.transform(dataSet)
    mappedLabels = MapLabelandTopics(dataSet)
    models = []
    accuracies = []
    precisions = []
    recalls= []
    f1Scores= []
    for paramGrid in paramGrids:
        try:
            model, [accuracy, precision, recall, f1Score] = TrainModel(dataSet, paramGrid)
            models.append(model)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1Scores.append(f1Score)
            break
            if (accuracy > 95):
                break
        except:
            pass
    model = models[accuracies.index(max(accuracies))]
    precision = precisions[accuracies.index(max(accuracies))]
    recall = recalls[accuracies.index(max(accuracies))]
    f1Scores = f1Scores[accuracies.index(max(accuracies))]
    return model, dataPipeline, mappedLabels, max(accuracies), precision, recall, f1Scores


def NBModelPredict(model, dataPipeline, params):
    params = spark.createDataFrame(params)
    testData = dataPipeline.transform(params)
    RFCPredictions = model.transform(testData)
    return RFCPredictions.collect()[-1]['prediction']

NBModel,NBDataPipeline, NBMappedLabels, NBAccuracy, NBPrecision, NBRecall, NBF1Scores= FindBestNBModel()
