# Databricks notebook source
# MAGIC %md
# MAGIC # w261 Final Project - Predict Flight Delays: Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC 25   
# MAGIC Justin Trobec, Jeff Li, Sonya Chen, Karthik Srinivasan
# MAGIC Spring 2021, section 5, Team 25

# COMMAND ----------

# Install some helper libraries

!pip install -U seaborn
!pip install geopy
!pip install dtreeviz[pyspark]
!pip install -U mlflow

# COMMAND ----------

# MAGIC %sh
# MAGIC git clone https://github.com/sllynn/spark-xgboost.git;
# MAGIC cd spark-xgboost;
# MAGIC pip install -e .;

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Import Packages

# COMMAND ----------

## imports

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from functools import reduce
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from dtreeviz.models.spark_decision_tree import ShadowSparkTree
from dtreeviz import trees
import mlflow
from scipy.stats import ttest_ind
from sparkxgb import XGBoostClassifier, XGBoostRegressor

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, DecisionTreeClassificationModel, GBTClassifier
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer, MinMaxScaler, RobustScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType, TimestampType
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, sum, avg, max, count, countDistinct, weekofyear, to_timestamp, date_format, to_date, lit, lag, unix_timestamp, expr, ceil, floor, when, hour, dayofweek, month, trim, explode, array, expr, coalesce

from distutils.version import LooseVersion


print(sns.__version__)
sqlContext = SQLContext(sc)

# COMMAND ----------

# MAGIC %md
# MAGIC # __Section 2__ - Algorithm Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In all of the models explored below, we use the following steps:
# MAGIC 
# MAGIC * Load joined data from database tables (Here we use data uploaded to the group25 database).
# MAGIC * Filter down to pre-selected columns. These columns have been downselected based on previous EDA.
# MAGIC * Split data into train, validation and test sets.
# MAGIC * Address imbalance in train data by minority class oversampling.
# MAGIC * Normalize numerical columns using train data.
# MAGIC * Vectorize and index the variables into sparse feature formats
# MAGIC * Train and validate model candidates using hyperparameter tuning.
# MAGIC * Conduct error analyses of model results.

# COMMAND ----------

# MAGIC %md
# MAGIC # __Section 3__ - EDA & Challenges

# COMMAND ----------

spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")

# COMMAND ----------

# MAGIC %sql
# MAGIC REFRESH TABLE group25.weather_airlines_utc_main_imputed_v3;

# COMMAND ----------

df_merged = sqlContext.table("group25.weather_airlines_utc_main_imputed_v3")
print('Total number of records in the table:{}'.format(df_merged.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We use a map of airline to airline names for better visual identification.

# COMMAND ----------

df_airline_names = sqlContext.table("group25.iata_to_airline_mapping")
display(df_airline_names)
airline_name_dict = {}
for index,rows in df_airline_names.toPandas().iterrows():
  airline_name_dict[rows['IATA_CODE']] = rows['AIRLINE']

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We downselect the features and segregate them into categorical or numerical variables. This is done to enable Vector assembling and one hot encoding later.

# COMMAND ----------

#Specify columns of interest

airline_numerical_columns = [
  'NORMALIZED_OUTBOUND_COUNT_2H',
  'NORMALIZED_OUTBOUND_COUNT_3H',
  'NORMALIZED_OUTBOUND_COUNT_4H',
  'NORMALIZED_OUTBOUND_COUNT_5H',
  'NORMALIZED_OUTBOUND_COUNT_6H',
  'NORMALIZED_INBOUND_COUNT_2H',
  'NORMALIZED_INBOUND_COUNT_3H',
  'NORMALIZED_INBOUND_COUNT_4H',
  'NORMALIZED_INBOUND_COUNT_5H',
  'NORMALIZED_INBOUND_COUNT_6H',
  'NORMALIZED_DIVERTED_OUTBOUND_COUNT_2H',
  'NORMALIZED_DIVERTED_OUTBOUND_COUNT_3H',
  'NORMALIZED_DIVERTED_OUTBOUND_COUNT_4H',
  'NORMALIZED_DIVERTED_OUTBOUND_COUNT_5H',
  'NORMALIZED_DIVERTED_OUTBOUND_COUNT_6H',
  'NORMALIZED_DIVERTED_INBOUND_COUNT_2H',
  'NORMALIZED_DIVERTED_INBOUND_COUNT_3H',
  'NORMALIZED_DIVERTED_INBOUND_COUNT_4H',
  'NORMALIZED_DIVERTED_INBOUND_COUNT_5H',
  'NORMALIZED_DIVERTED_INBOUND_COUNT_6H',
  'NORMALIZED_DELAY_OUTBOUND_COUNT_2H',
  'NORMALIZED_DELAY_OUTBOUND_COUNT_3H',
  'NORMALIZED_DELAY_OUTBOUND_COUNT_4H',
  'NORMALIZED_DELAY_OUTBOUND_COUNT_5H',
  'NORMALIZED_DELAY_OUTBOUND_COUNT_6H',
  'NORMALIZED_DELAY_INBOUND_COUNT_2H',
  'NORMALIZED_DELAY_INBOUND_COUNT_3H',
  'NORMALIZED_DELAY_INBOUND_COUNT_4H',
  'NORMALIZED_DELAY_INBOUND_COUNT_5H',
  'NORMALIZED_DELAY_INBOUND_COUNT_6H',
  'PREV_DEP_TIMEDELTA'
]

airline_time_columns = [
  'DEP_UTC_TIMESTAMP',
  'DEP_LOCAL_TIMESTAMP',
  'PREV_DEP_UTC_TIMESTAMP',
  'FL_DATE',
  'FL_DATETIME' 
]

airline_categorical_columns = [
  'MONTH',
  'DAY_OF_WEEK',
  'OP_UNIQUE_CARRIER',
  'call_sign_dep',
  'DEP_LOCAL_HOUR'
]

weather_categorical_columns = [
  'WND_DIRECTION_ANGLE_AVG_DIR',
  'MV_THUNDERSTORM',
  'MV_SHOWERS',
  'MV_SAND_OR_DUST',
  'MV_BLOWINGSNOW',
  'MV_FOG'
]

weather_numerical_columns = [
  'AA1_DEPTH_DIMENSION_AVG',
  'AA1_PERIOD_QUANTITY_AVG',
  'GA1_COVERAGE_CODE_AVG',
  'GA1_BASE_HEIGHT_DIMENSION_AVG',
  'WND_SPEED_AVG',
  'TMP_AIR_TEMPERATURE_AVG',
  'SLP_ATMOSPHERIC_PRESSURE_AVG'
]

pr_numerical_columns = [
  'ORIGIN_PR',
  'DEST_PR'
]

pr_non_imputed_numerical_columns = [
  'PR_DIFF',
  'PR_AA_ORIGIN',
  'PR_AA_DEST',
  'PR_AA_DIFF',
  'PR_AAD_ORIGIN',
  'PR_AAD_DEST',
  'PR_AAD_DIFF',
  'PR_AADD_ORIGIN',
  'PR_AADD_DEST',
  'PR_AADD_DIFF',
  'PR_FL',
  'PR_FLD',
  'PR_FLDH',
  'PREV_DEP_DELAY'
]

label_column = ['DEP_DEL15']

# Gather categorical and numerical columns
categorical_columns = airline_categorical_columns + weather_categorical_columns
numerical_columns = airline_numerical_columns + weather_numerical_columns + pr_numerical_columns

# COMMAND ----------

# MAGIC %md
# MAGIC # __Section 4__ - Algorithm Implementation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Construct ML Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data split: Train, Validation and Test
# MAGIC 
# MAGIC We split the data into three components, train, validation and test. The train data consists of all flights and weather data obtained between 2015 and 2017. Additionally, we use page ranks for airports that were also computed between 2015 and 2017. We ensure that we have no data leakage by joining the weather and engineered features using time windows lagged by 2 hours. This helps us construct each record that contains data 2 hours prior to departure. The same concept is used for validation and test datasets. However for validation and test datasets, we use page ranks that were computed on train datasets. This is done since the page ranks computed were aggregated on an hourly basis and not by date, and computing new page ranks on the 2018/2019 datasets would have resulted in data leakage.

# COMMAND ----------

# create time windows for data splitting
train_dates = ("2015-01-01",  "2017-12-31")
val_dates = ("2018-01-01",  "2018-12-31")
test_dates = ("2019-01-01",  "2019-12-31")
train_date_from, train_date_to = [to_date(lit(s)).cast(TimestampType()) for s in train_dates]
val_date_from, val_date_to = [to_date(lit(s)).cast(TimestampType()) for s in val_dates]
test_date_from, test_date_to = [to_date(lit(s)).cast(TimestampType()) for s in test_dates]

# Filter the dataset based on the time windows
df_train_all = df_merged.filter(df_merged.DEP_UTC_TIMESTAMP > train_date_from) \
                    .filter(df_merged.DEP_UTC_TIMESTAMP <= train_date_to) \
                    .filter(df_merged.TAIL_NUM.isNotNull())

df_val_all = df_merged.filter(df_merged.DEP_UTC_TIMESTAMP > val_date_from) \
                  .filter(df_merged.DEP_UTC_TIMESTAMP <= val_date_to) \
                  .filter(df_merged.TAIL_NUM.isNotNull())

df_test_all = df_merged.filter(df_merged.DEP_UTC_TIMESTAMP > test_date_from) \
                   .filter(df_merged.DEP_UTC_TIMESTAMP <= test_date_to)

# Create a categorical/boolean variable for identifying departure status of the previous departure of the same flight (tail_num)  
df_train_all = df_train_all.withColumn("PREV_DEP_DELAY_BOOL", when(df_train_all.PREV_DEP_DELAY > 15, 'Delayed').otherwise('Not_Delayed'))
df_val_all = df_val_all.withColumn("PREV_DEP_DELAY_BOOL", when(df_val_all.PREV_DEP_DELAY > 15, 'Delayed').otherwise('Not_Delayed'))
df_test_all = df_test_all.withColumn("PREV_DEP_DELAY_BOOL", when(df_test_all.PREV_DEP_DELAY > 15, 'Delayed').otherwise('Not_Delayed'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We then use the selected columns to filter down the datasets

# COMMAND ----------

# Set the categorical columns to be considered for the modeling
categorical_columns = []
categorical_columns += airline_categorical_columns
categorical_columns += [col+'_IMPUTE' for col in weather_categorical_columns]
categorical_columns += ['PREV_DEP_DELAY_BOOL']

# Set the numerical columns to be considered for the modeling
numerical_columns = []
numerical_columns += [col+'_IMPUTE' for col in weather_numerical_columns]
numerical_columns += [col+'_IMPUTE' for col in airline_numerical_columns]
numerical_columns += [col+'_IMPUTE' for col in pr_numerical_columns]
numerical_columns += pr_non_imputed_numerical_columns

# Select a few 
hue_columns = ['call_sign_dep', 'OP_UNIQUE_CARRIER', 'Airline', 'DEP_LOCAL_HOUR', 'DAY_OF_WEEK', 'MONTH']


all_columns = categorical_columns + numerical_columns + ['DEP_DEL15']

df_train_all = df_train_all.select(*all_columns)
df_val_all = df_val_all.select(*all_columns)
df_test_all = df_test_all.select(*all_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Training on partial dataset
# MAGIC 
# MAGIC We choose to train on a subset of 5M records and evaluate on 500K records. This is done for reducing the train times and gathering insights faster.

# COMMAND ----------

NUM_TRAIN_SAMPLES = 5000000
NUM_VAL_SAMPLES = 500000
NUM_TEST_SAMPLES = 500000

df_train = df_train_all.sample(NUM_TRAIN_SAMPLES/df_train_all.count(), 123)
df_train = df_train.withColumn("DEP_DEL15",df_train.DEP_DEL15.cast("int"))
df_val = df_val_all.sample(NUM_VAL_SAMPLES/df_val_all.count(), 123)
df_val = df_val.withColumn("DEP_DEL15", df_val.DEP_DEL15.cast("int"))
df_test = df_test_all.sample(NUM_TEST_SAMPLES/df_test_all.count(), 123)
df_test = df_test.withColumn("DEP_DEL15", df_test.DEP_DEL15.cast("int"))

negative_ratio = df_train.filter(df_train.DEP_DEL15 == 0).count()/df_train.count()
df_train = df_train.withColumn("classWeights", when(df_train.DEP_DEL15 == 1,negative_ratio).otherwise(1-negative_ratio))
df_val = df_val.withColumn("classWeights", when(df_val.DEP_DEL15 == 1,negative_ratio).otherwise(1-negative_ratio))
df_test = df_test.withColumn("classWeights", when(df_test.DEP_DEL15 == 1,negative_ratio).otherwise(1-negative_ratio))

df_train = df_train.withColumn("PREV_DEP_DELAY_BOOL", when(df_train.PREV_DEP_DELAY > 15, 'Delayed').otherwise('Not_Delayed'))
df_val = df_val.withColumn("PREV_DEP_DELAY_BOOL", when(df_val.PREV_DEP_DELAY > 15, 'Delayed').otherwise('Not_Delayed'))
df_test = df_test.withColumn("PREV_DEP_DELAY_BOOL", when(df_test.PREV_DEP_DELAY > 15, 'Delayed').otherwise('Not_Delayed'))



# COMMAND ----------

# Get a list of airports for plotting
airports = df_train.select("call_sign_dep").toPandas().call_sign_dep.unique()
airports[:5]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Minority Oversampling: Construct Balanced Data for Train
# MAGIC 
# MAGIC The training data is heavily imbalanced. Flights with delay only constitute \\( \approx 20\% \\) of the flights. This results in the model being biased towards predicting "No Delay" more often. We seek to remedy this situation by oversampling the minority class, in this case the "No Delay" class. the simplest oversampling mechanism is to resample from the minority class until the classes have been balanced. In this work, we do this by simply duplicating the minority class 4 fold.

# COMMAND ----------

print('Data Count (before balancing): {}'.format(df_train.count()))
df_train_majority = df_train.filter(df_train.DEP_DEL15 == 0)
df_train_minority = df_train.filter(df_train.DEP_DEL15 == 1)

ratio = int(df_train_majority.count()/df_train_minority.count())

print("Initial imbalance ratio: {}".format(ratio))

# duplicate the minority rows
print('Minority Class Count (before balancing): {}'.format(df_train_minority.count()))
oversampled_minority = df_train_minority.withColumn("dummy", explode(array([lit(x) for x in range(ratio)]))).drop('dummy')
print('Minority Class Count (after balancing): {}'.format(oversampled_minority.count()))

# combine both oversampled minority rows and previous majority rows 
df_train_oversampled = df_train_majority.union(oversampled_minority)
print('Data Count (after balancing): {}'.format(df_train_oversampled.count()))
ratio = (df_train_oversampled.filter(df_train.DEP_DEL15==1).count()/df_train_oversampled.filter(df_train.DEP_DEL15==0).count())
print("Oversampled imbalance ratio: {}".format(ratio))

print('Number of train samples:{}, Number of oversampled train samples:{}'.format(df_train.count(), df_train_oversampled.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Plot Variables
# MAGIC 
# MAGIC We randomly select 50,000 samples to plot our features of interest and their effect on the departure delay.

# COMMAND ----------

# Set widget at top of page for drill downs during plotting
dbutils.widgets.removeAll()
dbutils.widgets.dropdown("Numerical Column For Plotting", numerical_columns[0], [str(col) for col in numerical_columns])
dbutils.widgets.dropdown("Categorical Column For Plotting", categorical_columns[0], [str(col) for col in categorical_columns + ['Airline']])
dbutils.widgets.dropdown("Hue Column For Plotting", hue_columns[0], [str(col) for col in hue_columns])
dbutils.widgets.dropdown("Airport Column For Plotting", airports[0], [str(col) for col in airports])
dbutils.widgets.dropdown("Weather Column For Plotting", numerical_columns[0], [str(col) for col in numerical_columns])
# dbutils.widgets.dropdown("OverSampling", "ClassWeighted", [str(col) for col in ['ClassWeighted', 'BootStrapping', 'NoOverSampling']])

# COMMAND ----------

# Extract a pandas dataframe of 50,000 samples for easier plotting
NUM_PLOT_SAMPLES = 50000 #50,000
df_airlines = df_train.sample(NUM_PLOT_SAMPLES/df_train.count(), 123)
df_airlines_pandas = df_airlines.toPandas()
df_airlines_pandas.loc[:,'Airline'] = [airline_name_dict[rows['OP_UNIQUE_CARRIER']] for index, rows in df_airlines_pandas.iterrows()]
print(df_airlines_pandas.shape)
df_airlines_pandas.loc[:,"DEP_DEL15"] = df_airlines_pandas.DEP_DEL15.astype(str)
df_airlines_pandas.loc[df_airlines_pandas.DEP_DEL15=='0',"DEP_DEL15"] = 'Not_Delayed'
df_airlines_pandas.loc[df_airlines_pandas.DEP_DEL15=='1',"DEP_DEL15"] = 'Delayed'

# COMMAND ----------

# Display the plotting dataframe
df_airlines_pandas.head()

# COMMAND ----------

# Plot of weather data
col_w = dbutils.widgets.get("Weather Column For Plotting")
col_hue = dbutils.widgets.get("Hue Column For Plotting") 
col_airports = dbutils.widgets.get("Airport Column For Plotting")

plt.figure(figsize=(8,8))
g = sns.histplot(x=col_w, data=df_airlines_pandas, bins=20)
g.set(ylim=(None, None))

# COMMAND ----------

# Plot of numerical feature at a chosen airport
col_w = dbutils.widgets.get("Numerical Column For Plotting")
col_hue = dbutils.widgets.get("Hue Column For Plotting") 
col_airports = dbutils.widgets.get("Airport Column For Plotting")
 
if len(col_w.split('_IMPUTE')) > 1:
  col_w_name = col_w.split('_IMPUTE')[0]
else:
  col_w_name = col_w
  
plt.figure(figsize=(8,8))
g = sns.boxplot(y=col_w, x=col_hue,  data=df_airlines_pandas.loc[df_airlines_pandas.call_sign_dep==col_airports,:], showfliers=False)
g.set(ylim=(None, None))
plt.xticks(rotation=90)
plt.ylabel(col_w_name)
plt.title(col_hue + ' vs ' + col_w_name + ' at ' + col_airports)

# COMMAND ----------

col_n = dbutils.widgets.get("Numerical Column For Plotting")
col_hue = dbutils.widgets.get("Hue Column For Plotting") 
col_airports = dbutils.widgets.get("Airport Column For Plotting")
print(col_airports)
# for col in numerical_columns:
#   print(col)
plt.figure(figsize=(8,8))
g = sns.boxplot(y=col_n, x="DEP_DEL15",  hue = col_hue, data=df_airlines_pandas, showfliers = False)  
g.set(ylim=(None, None))
plt.title('{} vs DEPARTURE DELAY'.format(col_n.split('_IMPUTE')[0]))
g.set(ylabel='')

# COMMAND ----------

# for col in categorical_columns:
col = dbutils.widgets.get("Categorical Column For Plotting")
# col_n = dbutils.widgets.get("Numerical Column For Plotting")
plt.figure()
g = sns.catplot(x = col, hue="DEP_DEL15", kind="count", data=df_airlines_pandas)
plt.xticks(rotation=90)

# COMMAND ----------

col_n = dbutils.widgets.get("Numerical Column For Plotting")
col_hue = dbutils.widgets.get("Hue Column For Plotting") 
col_airports = dbutils.widgets.get("Airport Column For Plotting")

ctr = 1
num_plots = len(numerical_columns)
ncols = 3
nrows = np.ceil(num_plots/ncols)

# Set figsize here
fig, axes = plt.subplots(nrows=np.int(nrows), ncols=ncols, figsize=(24,48))

# flatten axes for easy iterating
for i, ax in enumerate(axes.flatten()):
  if i < num_plots:
    column = numerical_columns[i]
    if len(column.split('classVec')) > 1:
      col_name = column.split('classVec')[0]
    elif len(column.split('_IMPUTE')) > 1:
      col_name = column.split('_IMPUTE')[0]
    else:
      col_name = column
    g = sns.boxplot(y=column, x="DEP_DEL15",  data=df_airlines_pandas, showfliers = False, ax=ax)
    g.set(ylim=(None, None))
    ax.set_title('{} vs DELAYS'.format(col_name))
    ax.set_ylabel('')
#     sns.boxplot(x= data.iloc[:, i],  orient='v' , ax=ax)

fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Scaling/Normalization
# MAGIC 
# MAGIC We use the train data alone to scale the numerical columns. This is done to avoid leakage from unseen data. We opt to use the Robust Scaler, where centering and scaling statistics is based on percentiles and are therefore not influenced by a few number of very large marginal outliers. The resulting range for each transformed numerical feature, is therefore larger than for the other standard scalers (such as Standar/MinMax).

# COMMAND ----------

# Compile all columns for modeling
all_columns = categorical_columns + numerical_columns + ['DEP_DEL15']

# COMMAND ----------

# Transform all features into a vector using VectorAssembler
columns_to_scale = numerical_columns
assemblers = [VectorAssembler(inputCols=[col], outputCol=col + "_vec") for col in columns_to_scale]
scalers = [RobustScaler(inputCol=col + "_vec", outputCol=col + "_scaled") for col in columns_to_scale]

column_names = all_columns + ['classWeights']
scalerModel = {}
sampling_methods = ['BootStrapping', 'NoOversampling', 'ClassWeighted']

train_data = {}
train_data['BootStrapping'] = df_train_oversampled.select(*column_names)
train_data['NoOversampling'] = df_train.select(*column_names)
train_data['ClassWeighted'] = df_train.select(*column_names)

for sampling_type in sampling_methods:
  pipeline = Pipeline(stages=assemblers + scalers)
  scalerModel[sampling_type] = pipeline.fit(train_data[sampling_type])


# COMMAND ----------

stages = [] # stages in our Pipeline
for categoricalCol in categorical_columns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index").setHandleInvalid("keep")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
        from pyspark.ml.feature import OneHotEncoderEstimator
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    else:
        from pyspark.ml.feature import OneHotEncoder
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]

# Label the output column
label_stringIdx = StringIndexer(inputCol="DEP_DEL15", outputCol="label")
stages += [label_stringIdx]

#Assemble the categorical and numerical features 
assemblerInputs = [c + "classVec" for c in categorical_columns] + [c + "_scaled" for c in columns_to_scale]
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


# COMMAND ----------

# MAGIC %md
# MAGIC #### Transform train and test splits

# COMMAND ----------

# Scale and transform train and test dataframes
column_names = all_columns + ['classWeights']

pipelineModel_train = {}
df_train_scaled = {}
df_val_scaled = {}
df_test_scaled = {}
preppedDataDF_train = {}
preppedDataDF_val = {}
preppedDataDF_test = {}


for sampling_type in sampling_methods:
  df_train_scaled[sampling_type] = scalerModel[sampling_type].transform(train_data[sampling_type])
  df_val_scaled[sampling_type]  = scalerModel[sampling_type].transform(df_val.select(*column_names))
  df_test_scaled[sampling_type]  = scalerModel[sampling_type].transform(df_test.select(*column_names))
  
  partialPipeline = Pipeline().setStages(stages)
  pipelineModel_train[sampling_type] = partialPipeline.fit(df_train_scaled[sampling_type])
  preppedDataDF_train[sampling_type] = pipelineModel_train[sampling_type].transform(df_train_scaled[sampling_type]).cache()
  preppedDataDF_val[sampling_type] = pipelineModel_train[sampling_type].transform(df_val_scaled[sampling_type]).cache()
  preppedDataDF_test[sampling_type] = pipelineModel_train[sampling_type].transform(df_test_scaled[sampling_type]).cache()

  print('Number of training samples in {}: {}'.format(sampling_type, preppedDataDF_train[sampling_type].count()))
  print('Number of validation samples in {}: {}'.format(sampling_type, preppedDataDF_val[sampling_type].count()))
  print('Number of test samples in {}: {}'.format(sampling_type, preppedDataDF_test[sampling_type].count()))

# COMMAND ----------

# write prepped dataframes to tables

for sampling_type in sampling_methods:  
  preppedDataDF_train[sampling_type].createOrReplaceTempView("mytempTable")
  table_name = 'group25.data_train_main_prepped_' + sampling_type
  sqlContext.sql("DROP TABLE IF EXISTS {}".format(table_name));
  sqlContext.sql("create table {} as select * from mytempTable".format(table_name));

  table_name = 'group25.data_val_main_prepped_' + sampling_type
  preppedDataDF_val[sampling_type].createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS {}".format(table_name));
  sqlContext.sql("create table {} as select * from mytempTable".format(table_name));

  table_name = 'group25.data_test_main_prepped_' + sampling_type  
  preppedDataDF_test[sampling_type].createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS {}".format(table_name));
  sqlContext.sql("create table {} as select * from mytempTable".format(table_name));


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Read prepped data 

# COMMAND ----------

preppedDataDF_train = {}
preppedDataDF_val = {}
preppedDataDF_test = {}
sampling_methods = ['BootStrapping', 'NoOversampling', 'ClassWeighted']

for sampling_type in sampling_methods:
  table_name = 'group25.data_train_main_prepped_' + sampling_type
  preppedDataDF_train[sampling_type] = sqlContext.table(table_name).cache()

  table_name = 'group25.data_val_main_prepped_' + sampling_type
  preppedDataDF_val[sampling_type]= sqlContext.table(table_name).cache()

  table_name = 'group25.data_test_main_prepped_' + sampling_type  
  preppedDataDF_test[sampling_type] = sqlContext.table(table_name).cache()
  
  print('Number of training samples in {}: {}'.format(sampling_type, preppedDataDF_train[sampling_type].count()))
  print('Number of validation samples in {}: {}'.format(sampling_type, preppedDataDF_val[sampling_type].count()))
  print('Number of test samples in {}: {}'.format(sampling_type, preppedDataDF_test[sampling_type].count()))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Models

# COMMAND ----------

# MAGIC %md
# MAGIC #### Metrics
# MAGIC 
# MAGIC We implement the F-beta measure to evaluate the performance of the model. In the formula below, \\( \beta >1 \\) results in a score that weights more on recall while a \\( \beta < 1 \\) weights precision more. The end business use case dictates the best measure to be used. In this work, we report F1, F_0.5 and F2 scores for comparing various models. 
# MAGIC 
# MAGIC $$F_{\beta} = (1+\beta^2)* \frac{Pr\cdot Rec}{\beta^2 \cdot Pr + Rec}$$
# MAGIC 
# MAGIC where \\( Pr \\) and \\( Rec \\) are precision and recall, respectively.

# COMMAND ----------

import time
sleep_time = 0
def getFMeasures(pr, rec, beta):
  beta_sq = beta*beta
  if beta_sq*pr+rec > 1e-16:
    score = (1+beta_sq)*(pr*rec/(beta_sq*pr+rec))
  else:
    score = 0
    
  return score

def assessModelPerformance(model, data, prefix):
  predictions = model.transform(data)

  # Compute raw scores on the test set
  predictionAndLabels = predictions[["label", "prediction"]].rdd.map(lambda x: (x[0], x[1]))

  # Instantiate metrics object
  metrics = BinaryClassificationMetrics(predictionAndLabels)

  # Area under precision-recall curve
  print("Area under {}PR = {}".format(prefix, round(metrics.areaUnderPR,3)))

  # Area under ROC curve
  print("Area under {}ROC  = {}".format(prefix, round(metrics.areaUnderROC,3)))

  predictionAndTarget = predictions.select("label", "prediction")
  predictionAndTargetNumpy = np.array((predictionAndTarget.collect()))
  
  # Calculate Precision and Recall
  precision = precision_score(predictionAndTargetNumpy[:,0], predictionAndTargetNumpy[:,1])
  recall = recall_score(predictionAndTargetNumpy[:,0], predictionAndTargetNumpy[:,1])

  #Compute AUC
  auc = roc_auc_score(predictionAndTargetNumpy[:,0], predictionAndTargetNumpy[:,1])

  # Compute evaluation metrics
  acc = accuracy_score(predictionAndTargetNumpy[:,0], predictionAndTargetNumpy[:,1])
  f1 = f1_score(predictionAndTargetNumpy[:,0], predictionAndTargetNumpy[:,1])
  f_0p5 = getFMeasures(precision, recall, 0.5)
  f_2p0 = getFMeasures(precision, recall, 2.0)
  
  print('{}Accuracy: {}\n{}F1-Score: {}\n{}F0.5-Score: {}\n{}F2-Score: {}\n{}Precision: {}\n{}Recall: {}\n{}AUC: {}'.format(prefix, round(acc,3), 
                                                                                                                            prefix, round(f1,3), 
                                                                                                                            prefix, round(f_0p5,3), 
                                                                                                                            prefix,round(f_2p0,3), 
                                                                                                                            prefix,round(precision,3), 
                                                                                                                            prefix, round(recall,3), 
                                                                                                                            prefix, round(auc,3)))
  return (acc, f1, precision, recall, auc, f_0p5, f_2p0)

def runAssessments(model, data_prefix, dataset, iteration, hyperparameters, sampling_type, model_name):
    print('-----------------' + data_prefix + '--------------------')

    print('Starting model assessment...')
    acc, f1, precision, recall, auc, f_0p5, f_2p0 = assessModelPerformance(model, dataset, data_prefix)
    print('Finished model assessment...')

    results = {
      'Model': model_name,
      'HyperParameter': hyperparameters,
      'Sampling': sampling_type,
      'Accuracy': round(acc,3), 
      'F1-score': round(f1,3), 
      'Precision': round(precision,3), 
      'Recall': round(recall,3), 
      'AUC': round(auc,3), 
      'F0.5-score': round(f_0p5,3), 
      'F2-score': round(f_2p0,3),
      'Prefix': data_prefix
    }
    print(results)
    df_temp = pd.DataFrame(results, index = [iteration])

    print('---------------------------------------------------')

      
    return df_temp

# COMMAND ----------

def insertIntoTable(input):
  notebook_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
  sqlContext.sql("""
    CREATE TABLE IF NOT EXISTS group25.experiment_results (
      RecordTimeStamp timestamp,
      Results varchar(10000),
      Notebook varchar(1000)
      )
  """);
  
  sqlContext.sql("""
    INSERT INTO group25.experiment_results
    VALUES (CURRENT_TIMESTAMP(), '{}', '{}')
  """.format(input, notebook_name))
  

# COMMAND ----------

import ast

def checkIfRecordExists(model_name, hyperparameters, sampling_method):
  df_exp_results = sqlContext.sql("""
    SELECT * FROM group25.experiment_results
  """)
  
  df_exp_results_pandas = df_exp_results.toPandas()

  delimiters = ["\"Model\":",
                "\"HyperParameter\":",
                "\"Sampling\":", 
                "\"Accuracy\":", 
                "\"F1-score\":", 
                "\"Precision\":", 
                "\"Recall\":", 
                "\"AUC\":", 
                "\"F0.5-score\":", 
                "\"F2-score\":", 
                "\"Prefix\":"]
  
  replace_dict = {
    '\"maxDepth\"': '\\"maxDepth\\"',
    '\"eta\"': '\\"eta\\"',
    '\"regParam\"': '\\"regParam\\"',
    '\"elasticNetParam\"': '\\"elasticNetParam\\"', 
    '\"threshold\"': '\\"threshold\\"'
  }

  res_list = []
  for item in df_exp_results_pandas.Results.values:
    try:
      part0 = item.split("\"Prefix\":")
      prefix = [v for k,v in json.loads(str(part0[1]).replace("}}","}")).items()][0]
      if prefix == 'Val':
        if model_name in part0[0]:
            res_list.append(item)
    except:
      continue
      
  return pd.DataFrame({'Results':res_list})

hyperparameters = {'eta': '0.001', 'maxDepth': '8'}
res = checkIfRecordExists('DecisionTrees', hyperparameters , 'BootStrapping')
display(res)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*) FROM group25.experiment_results

# COMMAND ----------

def getErrorPlots(predictions, columns):
  predictions = predictions.withColumn("Error", abs(predictions.label - predictions.prediction).cast("string"))
  num_plots = len(columns)
  ncols = 3
  nrows = np.ceil(num_plots/ncols)

  # Set figsize here
  fig, axes = plt.subplots(nrows=np.int(nrows), ncols=ncols, figsize=(24,48))

  # if you didn't set the figsize above you can do the following
  # fig.set_size_inches(12, 5)

  # flatten axes for easy iterating
  for i, ax in enumerate(axes.flatten()):
    if i < num_plots:
      column = columns[i]
      if len(column.split('classVec')) > 1:
        col_name = column.split('classVec')[0]
      elif len(column.split('_IMPUTE')) > 1:
        col_name = column.split('_IMPUTE')[0]
      else:
        col_name = column
      g = sns.boxplot(y=column, x="Error",  data=df_airlines_pandas, showfliers = False, ax=ax)
      g.set(ylim=(None, None))
      ax.set_title('{} vs Prediction Errors'.format(col_name))
      ax.set_ylabel('')
  #     sns.boxplot(x= data.iloc[:, i],  orient='v' , ax=ax)

  fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Logistic Regression
# MAGIC 
# MAGIC We use logistic regression to predict the outcome variable 'DEP_DEL15'. We use 10-fold cross validation on the train data (Y2015-2017) to minimize overfitting. The model is validated on the validation test (Y2018).

# COMMAND ----------

#class weights
import json

regParams = [0.3]
elasticNetParams = [0.5]

model_name = 'LogisticRegression'
evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR")
numFolds = 10

df_model_results = pd.DataFrame()
iteration = 0

best_model = {}
for regParam in regParams:
  for elasticNetParam in elasticNetParams:
    for sampling_method in ['BootStrapping']: #['ClassWeighted']: #sampling_methods:
      lrModel = None
      
      dataset = {'Train': preppedDataDF_train[sampling_method], 'Val': preppedDataDF_val[sampling_method]}
      
      if sampling_method =='ClassWeighted':
        lr = LogisticRegression(labelCol="label", 
                                     featuresCol="features", 
                                     weightCol="classWeights", 
                                     maxIter=100)        
      elif sampling_method=='BootStrapping':
        lr = LogisticRegression(labelCol="label", 
                                     featuresCol="features", 
                                     maxIter=100)
      elif sampling_method=='NoOversampling':
        lr = LogisticRegression(labelCol="label", 
                                     featuresCol="features", 
                                     maxIter=100)

      # Construct pipeline to setup cross validation
      pipeline = Pipeline(stages=[lr])
      
      # Create parameter grid search
      paramGrid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [regParam])
        .addGrid(lr.elasticNetParam, [elasticNetParam])
        .build()
      )
      
      # Cross validate
      lrModel = CrossValidator(
          estimator=pipeline,
          evaluator=evaluator,
          estimatorParamMaps=paramGrid,
          collectSubModels=True,
          numFolds=numFolds).fit(preppedDataDF_train[sampling_method])
      
      # Select the best threshold
      fMeasure =lrModel.bestModel.stages[0].summary.fMeasureByThreshold
      maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
      bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
                              .select('threshold').head()['threshold']
      lrModel.bestModel.stages[0].setThreshold(bestThreshold)
      
      prefixes = ['Train', 'Val']
      for data_prefix in prefixes:
        hyperparameters = json.dumps({'regParam':str(regParam), 'elasticNetParam': str(elasticNetParam), 'threshold': bestThreshold})
        df_temp = runAssessments(lrModel, data_prefix, dataset[data_prefix], iteration, hyperparameters, sampling_method)
        df_model_results = df_model_results.append(df_temp)

        print('---------------------------------------------------')

        iteration += 1
        if data_prefix == 'Val':
          if not best_model:
            best_model = {'val_f1_score': df_temp[['F1-score']].values[0], 'model': lrModel, 'data': df_temp}
          else:
            if best_model['val_f1_score'] < df_temp[['F1-score']].values[0]:
              best_model = {'val_f1_score': df_temp[['F1-score']].values[0], 'model': lrModel, 'data': df_temp}

print(best_model)
display(df_model_results)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Plot Confusion Matrix
# MAGIC 
# MAGIC Here, we plot the confusion matrix for the results of the logistic regression.

# COMMAND ----------



predictions = best_model['model'].transform(dataset['Val'])
y_test = list(predictions.select("label").toPandas()['label'].astype(int))
y_pred = list(predictions.select("prediction").toPandas()['prediction'].astype(int))

data = {'y_test':    y_test,
        'y_pred': y_pred
        }

df = pd.DataFrame(data, columns=['y_test','y_pred'])
cm = pd.crosstab(df['y_test'], df['y_pred'], rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(cm/len(y_test), annot=True)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Decision Trees

# COMMAND ----------

model_name = 'DecisionTrees'
evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR")
numFolds = 10

df_model_dt_results = pd.DataFrame()
iteration = 0
depths = [5,10]

best_model = {}
for depth in depths:
  for sampling_method in ['BootStrapping']: #['ClassWeighted']: #sampling_methods:
    dtModel = None

    dataset = {'Train': preppedDataDF_train[sampling_method], 'Val': preppedDataDF_val[sampling_method]}

    if sampling_method =='ClassWeighted':
      dt = DecisionTreeClassifier(labelCol="label", 
                                       featuresCol="features",
                                       weightCol="classWeights") 
    elif sampling_method=='BootStrapping':
      dt = DecisionTreeClassifier(labelCol="label", 
                                       featuresCol="features") 
    elif sampling_method=='NoOversampling':
      dt = DecisionTreeClassifier(labelCol="label", 
                                       featuresCol="features") 

    # Construct pipeline to setup cross validation
    pipeline = Pipeline(stages=[dt])

    # Create parameter grid search
    paramGrid = (
      ParamGridBuilder()
      .addGrid(dt.maxDepth, [depth])
      .build()
    )

    # Cross validate
    dtModel = CrossValidator(
        estimator=pipeline,
        evaluator=evaluator,
        estimatorParamMaps=paramGrid,
        collectSubModels=True,
        numFolds=numFolds).fit(preppedDataDF_train[sampling_method])


    prefixes = ['Train', 'Val']
    for data_prefix in prefixes:
      hyperparameters = json.dumps({'maxDepth': str(depth)})
      df_temp = runAssessments(dtModel, data_prefix, dataset[data_prefix], iteration, hyperparameters, sampling_method)
      df_model_dt_results = df_model_dt_results.append(df_temp)

      print('---------------------------------------------------')

      iteration += 1
      if data_prefix == 'Val':
        if not best_model:
          best_model = {'val_f1_score': df_temp[['F1-score']].values[0], 'model': dtModel, 'data': df_temp}
        else:
          if best_model['val_f1_score'] < df_temp[['F1-score']].values[0]:
            best_model = {'val_f1_score': df_temp[['F1-score']].values[0], 'model': dtModel, 'data': df_temp}

print(best_model)
display(df_model_dt_results)

# COMMAND ----------

print(dtModel.toDebugString)

def _get_root_node(tree: DecisionTreeClassificationModel):
    return tree._call_java('rootNode')
  
dtModel._call_java('rootNode').toString()

# COMMAND ----------


# spark_dtree = ShadowSparkTree(dtModel, predictions['features'], predictions['prediction'], feature_names='features', target_name='prediction', class_names=[0, 1])
# trees.dtreeviz(spark_dtree, fancy=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest

# COMMAND ----------

tree_list = [100, 500, 1000]
max_depth_list = [3, 4, 5]
model_name = 'RandomForest'
best_rf_model = {}
df_model_rf_results = pd.DataFrame()
iteration = 0
evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR")

for trees in tree_list:
  for depth in max_depth_list:
    for sampling_method in sampling_methods:
      rfModel = None
      dataset = {'Train': preppedDataDF_train[sampling_method], 'Val': preppedDataDF_val[sampling_method]}
      
      if sampling_method =='ClassWeighted':
        rf = RandomForestClassifier(labelCol="label", 
                                    featuresCol="features", 
                                    weightCol="classWeights", 
                                    maxDepth=depth, numTrees=trees)

      elif sampling_method=='BootStrapping':
        rf = RandomForestClassifier(labelCol="label", 
                                    featuresCol="features",  
                                    maxDepth=depth, numTrees=trees)
      elif sampling_method=='NoOversampling':
        rf = RandomForestClassifier(labelCol="label", 
                                    featuresCol="features",  
                                    maxDepth=depth, numTrees=trees)
      # Construct pipeline to setup cross validation
      pipeline = Pipeline(stages=[rf])

      # Create parameter grid search
      paramGrid = (
        ParamGridBuilder()
        .addGrid(rf.maxDepth, [depth])
        .addGrid(rf.numTrees, [trees])
        .build()
      )

      # Cross validate
      rfModel = CrossValidator(
          estimator=pipeline,
          evaluator=evaluator,
          estimatorParamMaps=paramGrid,
          collectSubModels=True,
          numFolds=numFolds).fit(preppedDataDF_train[sampling_method])

      prefixes = ['Train', 'Val']
      for data_prefix in prefixes:
        hyperparameters = json.dumps({'numTrees':str(trees), 'maxDepth': str(depth)})
        df_temp = runAssessments(rfModel, data_prefix, dataset[data_prefix], iteration, hyperparameters, sampling_method)
        df_model_rf_results = df_model_rf_results.append(df_temp)

        print('---------------------------------------------------')

        iteration += 1
        if data_prefix == 'Val':
          if not best_rf_model:
            best_rf_model = {'val_f1_score': df_temp[['F1-score']].values[0], 'model': rfModel, 'data': df_temp}
          else:
            if best_rf_model['val_f1_score'] < df_temp[['F1-score']].values[0]:
              best_rf_model = {'val_f1_score': df_temp[['F1-score']].values[0], 'model': rfModel, 'data': df_temp}

print(best_rf_model)
display(df_model_rf_results)


# COMMAND ----------

import re
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))
  
sampling_method = 'BootStrapping'
df_importance = ExtractFeatureImp(rfModel.bestModel.stages[0].featureImportances, preppedDataDF_train[sampling_method], "features")
# df_importance.head(50)

var_name = []
for features in df_importance.name.values[:]:
  if len(features.split('classVec')) > 1:
    var_name.append(features.split('classVec')[0])
  elif len(features.split('_IMPUTE')) > 1:
    var_name.append(features.split('_IMPUTE')[0])
  else:
    var_name.append(features)
    print('Did not split variable {}'.format(features))
df_importance.name = var_name
df_importance.head(50)

# COMMAND ----------

display(df_model_rf_results)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Gradient Boosted Trees

# COMMAND ----------

#class weights

tree_list = [3,5,7, 9, 11]
depths = [3, 4, 5]
model_name = 'GradientBoostedTrees'
evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR")
numFolds =5
df_model_gbt_results = pd.DataFrame()
iteration = 0
best_gbt_model ={}

sampling_methods = ['BootStrapping']

for sampling_method in sampling_methods:
  for trees in tree_list:
    for depth in depths:

      gbtModel = None
      
      dataset = {'Train': preppedDataDF_train[sampling_method], 'Val': preppedDataDF_val[sampling_method]}
      
      # Fit model to prepped data
      if sampling_method =='ClassWeighted':
        gbt = GBTClassifier(labelCol="label", featuresCol="features", weightCol="classWeights")
      elif sampling_method=='BootStrapping':
        gbt = GBTClassifier(labelCol="label", featuresCol="features")
      elif sampling_method=='NoOversampling':
        gbt = GBTClassifier(labelCol="label", featuresCol="features")

      # Construct pipeline to setup cross validation
      pipeline = Pipeline(stages=[gbt])

      # Create parameter grid search
      paramGrid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [depth])
        .addGrid(gbt.maxIter, [trees])
        .build()
      )

      # Cross validate
      gbtModel = CrossValidator(
          estimator=pipeline,
          evaluator=evaluator,
          estimatorParamMaps=paramGrid,
          collectSubModels=True,
          numFolds=numFolds).fit(preppedDataDF_train[sampling_method])

      prefixes = ['Train', 'Val']
      for data_prefix in prefixes:
        hyperparameters = json.dumps({'maxIter':str(trees), 'maxDepth':str(depth)})
        df_temp = runAssessments(gbtModel, data_prefix, dataset[data_prefix], iteration, hyperparameters, sampling_method, model_name)
        df_model_gbt_results = df_model_gbt_results.append(df_temp)

        print('---------------------------------------------------')

        iteration += 1
        if data_prefix == 'Val':
          if not best_gbt_model:
            best_gbt_model = {'val_f1_score': df_temp[['F1-score']].values[0], 'model': gbtModel, 'data': df_temp}
          else:
            if best_gbt_model['val_f1_score'] < df_temp[['F1-score']].values[0]:
              best_gbt_model = {'val_f1_score': df_temp[['F1-score']].values[0], 'model': gbtModel, 'data': df_temp}
            
print(best_gbt_model)
display(df_model_gbt_results)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Extreme Gradient Boosting

# COMMAND ----------

# eXtreme Gradient Boosting
etas = [1e-1, 1e-2, 1e-3]
depths = [4, 8, 12]
model_name = 'eXtremeGradientBoostedTrees'
numFolds = 5
df_model_xgb_results = pd.DataFrame()
iteration = 0
best_xgb_model = {}
sampling_methods = ['BootStrapping', 'NoOversampling', 'ClassWeighted']

xgbParams = dict(
  eta=0.1,
  maxDepth=2,
  missing=0.0,
  objective="binary:logistic",
  numRound=5,
  numWorkers=6
)

evaluator = BinaryClassificationEvaluator(
  rawPredictionCol="rawPrediction",
  labelCol="label"
)

for sampling_method in sampling_methods:
  for depth in depths:
    for eta in etas:

      dataset = {'Train': preppedDataDF_train[sampling_method], 'Val': preppedDataDF_val[sampling_method]}
      xgbModel = None
      # Fit model to prepped data
      if sampling_method =='ClassWeighted':
        xgb = XGBoostClassifier(**xgbParams).setFeaturesCol("features").setLabelCol("label").setWeightCol("classWeights")
      elif sampling_method=='BootStrapping':
        xgb = XGBoostClassifier(**xgbParams).setFeaturesCol("features").setLabelCol("label")
      elif sampling_method=='NoOversampling':
        xgb = XGBoostClassifier(**xgbParams).setFeaturesCol("features").setLabelCol("label")

      # Construct pipeline to setup cross validation
      pipeline = Pipeline(stages=[xgb])

      # Create parameter grid search
      paramGrid = (
        ParamGridBuilder()
        .addGrid(xgb.eta, [eta])
        .addGrid(xgb.maxDepth, [depth])
        .build()
      )

      # Cross validate
      xgbModel = CrossValidator(
          estimator=pipeline,
          evaluator=evaluator,
          estimatorParamMaps=paramGrid,
          collectSubModels=True,
          numFolds=numFolds).fit(preppedDataDF_train[sampling_method])

      prefixes = ['Train', 'Val']
      for data_prefix in prefixes:
        hyperparameters = json.dumps({'eta':str(eta), 'maxDepth': str(depth)})
        df_temp = runAssessments(xgbModel, data_prefix, dataset[data_prefix], iteration, hyperparameters, sampling_method, model_name)
        df_model_xgb_results = df_model_xgb_results.append(df_temp)

        print('---------------------------------------------------')

        iteration += 1
        if data_prefix == 'Val':
          if not best_xgb_model:
            best_xgb_model = {'val_f1_score': df_temp[['F1-score']].values[0], 'model': xgbModel, 'data': df_temp}
          else:
            if best_xgb_model['val_f1_score'] < df_temp[['F1-score']].values[0]:
              best_xgb_model = {'val_f1_score': df_temp[['F1-score']].values[0], 'model': xgbModel, 'data': df_temp}
            
print(best_xgb_model)
display(df_model_xgb_results)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Test Dataset Evaluation

# COMMAND ----------

