import warnings
import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer

from helpers.eda import *
from helpers.data_prep import *

# Steps
# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Feature Engineering
# 4. Gradient Boosted Tree Classifier Model
# 5. Model Tuning




warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

findspark.init(r"C:\Users\ithalat\spark\spark-3.1.2-bin-hadoop3.2")

spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_giris") \
    .getOrCreate()

sc = spark.sparkContext

# sc.stop()

##################################################
# Reading data
##################################################

spark_df = spark.read.csv("datasets/churn2.csv", header=True, inferSchema=True)

##################################################
# Exploratory Data Analysis
##################################################
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])

print("Shape: ", (spark_df.count(), len(spark_df.columns)))
# Shape:  (10000, 14)

spark_df.columns
spark_df.toPandas().info()

#  #   Column           Non-Null Count  Dtype
# ---  ------           --------------  -----
#  0   ROWNUMBER        10000 non-null  int32
#  1   CUSTOMERID       10000 non-null  int32
#  2   SURNAME          10000 non-null  object
#  3   CREDITSCORE      10000 non-null  int32
#  4   GEOGRAPHY        10000 non-null  object --- country
#  5   GENDER           10000 non-null  object
#  6   AGE              10000 non-null  int32
#  7   TENURE           10000 non-null  int32  --- customer age
#  8   BALANCE          10000 non-null  float64
#  9   NUMOFPRODUCTS    10000 non-null  int32
#  10  HASCRCARD        10000 non-null  int32
#  11  ISACTIVEMEMBER   10000 non-null  int32
#  12  ESTIMATEDSALARY  10000 non-null  float64
#  13  EXITED           10000 non-null  int32
# dtypes: float64(2), int32(9), object(3)

spark_df.toPandas().head()

spark_df.toPandas().isnull().sum()
# there are no null values

check_df(spark_df.toPandas())

spark_df.toPandas().describe()

#        ROWNUMBER  CUSTOMERID  CREDITSCORE      AGE   TENURE   BALANCE  \
# count   10000.00    10000.00     10000.00 10000.00 10000.00  10000.00
# mean     5000.50 15690940.57       650.53    38.92     5.01  76485.89
# std      2886.90    71936.19        96.65    10.49     2.89  62397.41
# min         1.00 15565701.00       350.00    18.00     0.00      0.00
# 25%      2500.75 15628528.25       584.00    32.00     3.00      0.00
# 50%      5000.50 15690738.00       652.00    37.00     5.00  97198.54
# 75%      7500.25 15753233.75       718.00    44.00     7.00 127644.24
# max     10000.00 15815690.00       850.00    92.00    10.00 250898.09
#        NUMOFPRODUCTS  HASCRCARD  ISACTIVEMEMBER  ESTIMATEDSALARY   EXITED
# count       10000.00   10000.00        10000.00         10000.00 10000.00
# mean            1.53       0.71            0.52        100090.24     0.20
# std             0.58       0.46            0.50         57510.49     0.40
# min             1.00       0.00            0.00            11.58     0.00
# 25%             1.00       0.00            0.00         51002.11     0.00
# 50%             1.00       1.00            1.00        100193.91     0.00
# 75%             2.00       1.00            1.00        149388.25     0.00
# max             4.00       1.00            1.00        199992.48     1.00

num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']
spark_df.select("age").distinct().count()
# Out[53]: 70

spark_df.select(num_cols).describe().toPandas().transpose()



for col in num_cols:
    spark_df.select(col).distinct().show()


for col in cat_cols:
    spark_df.select(col).distinct().show()

for col in [col.lower() for col in num_cols]:
    spark_df.groupby("exited").agg({col: "mean"}).show()


##################################################
# Data Preprocessing & Feature Engineering
##################################################

############################
# Missing Values
############################

from pyspark.sql.functions import when, count, col

spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T

spark_df.sort("estimatedsalary").show(5)
# when we check there are no 0 values on the column

############################
# Feature Interaction
############################

spark_df = spark_df.withColumn('est_sal/credit_score', spark_df.estimatedsalary / spark_df.creditscore)
spark_df = spark_df.withColumn('tenure/age', spark_df.tenure / spark_df.age)
spark_df = spark_df.withColumn('balance/est_sal', spark_df.balance / spark_df.estimatedsalary)
spark_df = spark_df.withColumn('montly_sal', spark_df.estimatedsalary / 12)
spark_df.show(5)

# because we want to use qcut method we transform the spark df to pandas df
to_pd_df = spark_df.toPandas()

to_pd_df["CreditsScore"] = pd.qcut(to_pd_df['creditscore'], 5, labels=[1, 2, 3, 4, 5])
to_pd_df["AgeScore"] = pd.qcut(to_pd_df['age'], 4, labels=[1, 2, 3, 4])
to_pd_df["BalanceScore"] = pd.qcut(to_pd_df['balance'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
to_pd_df["EstSalaryScore"] = pd.qcut(to_pd_df['estimatedsalary'], 5, labels=[1, 2, 3, 4, 5])

to_pd_df = to_pd_df.drop(["creditscore"], axis=1)
to_pd_df = to_pd_df.drop(["tenure"], axis=1)
to_pd_df = to_pd_df.drop(["balance"], axis=1)
to_pd_df = to_pd_df.drop(["estimatedsalary"], axis=1)

# after finish the feature iteraction we return to spark environment
spark_df = spark.createDataFrame(to_pd_df)
spark_df.show(5)


############################
# Label Encoding
############################


indexer = StringIndexer(inputCol="gender", outputCol="gender_label")
indexer.fit(spark_df).transform(spark_df).show(5)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("gender_label", temp_sdf["gender_label"].cast("integer"))
spark_df = spark_df.drop('gender')

indexer = StringIndexer(inputCol="geography", outputCol="geography_label")
indexer.fit(spark_df).transform(spark_df).show(5)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("geography_label", temp_sdf["geography_label"].cast("integer"))
spark_df = spark_df.drop('geography')


############################
# TARGET'ın Tanımlanması
############################

# TARGET'ın tanımlanması
stringIndexer = StringIndexer(inputCol='exited', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))

spark_df.show(5)

############################
# Feature'ların Tanımlanması
############################
spark_df.columns

cols = ['age',
        'numofproducts',
        'hascrcard',
        'isactivemember',
        'est_sal/credit_score',
        'tenure/age',
        'balance/est_sal',
        'montly_sal',
        'CreditsScore',
        'AgeScore',
        'BalanceScore',
        'EstSalaryScore',
        'gender_label',
        'geography_label',
        ]

# Vectorize independent variables.
va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)
va_df.show()

# Final df
final_df = va_df.select("features", "label")
final_df.show(5)


# StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
final_df = scaler.fit(final_df).transform(final_df)

# Split the dataset into test and train sets.
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))

############################
# Gradient Boosted Tree Classifier
############################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)
y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()
# Out[35]: 0.8446411012782694


############################
# Model Tuning
############################

evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6,8])
              .addGrid(gbm.maxBins, [20, 28])
              .addGrid(gbm.maxIter, [10])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)


y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()
# Out[11]: 0.8544739429695182





