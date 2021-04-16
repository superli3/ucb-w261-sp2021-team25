# Databricks notebook source
# MAGIC %md
# MAGIC # w261 Final Project - Join Airline, Weather and PageRank Tables

# COMMAND ----------

# MAGIC %md
# MAGIC 25   
# MAGIC Justin Trobec, Jeff Li, Sonya Chen, Karthik Srinivasan
# MAGIC Spring 2021, section 5, Team 25

# COMMAND ----------

# MAGIC %md
# MAGIC ## Description
# MAGIC 
# MAGIC This notebook joins all of our data together from pagerank, weather, and processed airliens tables together. It also performs the imputations necessaary downstream

# COMMAND ----------

!pip install -U seaborn
!pip install geopy
! pip install dtreeviz[pyspark]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Import Packages

# COMMAND ----------

## imports

from pyspark.sql import functions as f
from pyspark.sql.functions import col, sum, avg, max, count, countDistinct, weekofyear, to_timestamp, date_format, to_date, lit, lag, unix_timestamp, expr, ceil, floor, when, hour, dayofweek, month, trim, explode, array, expr, coalesce, isnan

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType, TimestampType
from pyspark.sql import SQLContext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pyspark
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer, MinMaxScaler, RobustScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from distutils.version import LooseVersion
from pyspark.ml import Pipeline
from pyspark.sql.window import Window

from pandas.tseries.holiday import USFederalHolidayCalendar
import datetime
from functools import reduce
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from dtreeviz.models.spark_decision_tree import ShadowSparkTree
from dtreeviz import trees

print(sns.__version__)
sqlContext = SQLContext(sc)

# COMMAND ----------

#Specify columns of interest


airline_numerical_impute_zero = [
    'PREV_DEP_TIMEDELTA',
  'COUNT_FLIGHTS',
  'NUM_DELAYS',
  'FRACTION_DELAYS',
    'DIVERTED_OUTBOUND_COUNT',
  'DIVERTED_OUTBOUND_COUNT_0H',
  'DIVERTED_OUTBOUND_COUNT_1H',
  'DIVERTED_OUTBOUND_COUNT_2H',
  'DIVERTED_OUTBOUND_COUNT_3H',
  'DIVERTED_OUTBOUND_COUNT_4H',
  'DIVERTED_OUTBOUND_COUNT_5H',
  'DIVERTED_OUTBOUND_COUNT_6H',
  'NORMALIZED_DIVERTED_OUTBOUND_COUNT_0H',
  'NORMALIZED_DIVERTED_OUTBOUND_COUNT_1H',
  'NORMALIZED_DIVERTED_OUTBOUND_COUNT_2H',
  'NORMALIZED_DIVERTED_OUTBOUND_COUNT_3H',
  'NORMALIZED_DIVERTED_OUTBOUND_COUNT_4H',
  'NORMALIZED_DIVERTED_OUTBOUND_COUNT_5H',
  'NORMALIZED_DIVERTED_OUTBOUND_COUNT_6H',
  'DIVERTED_INBOUND_COUNT',
  'DIVERTED_INBOUND_COUNT_0H',
  'DIVERTED_INBOUND_COUNT_1H',
  'DIVERTED_INBOUND_COUNT_2H',
  'DIVERTED_INBOUND_COUNT_3H',
  'DIVERTED_INBOUND_COUNT_4H',
  'DIVERTED_INBOUND_COUNT_5H',
  'DIVERTED_INBOUND_COUNT_6H',
  'NORMALIZED_DIVERTED_INBOUND_COUNT_0H',
  'NORMALIZED_DIVERTED_INBOUND_COUNT_1H',
  'NORMALIZED_DIVERTED_INBOUND_COUNT_2H',
  'NORMALIZED_DIVERTED_INBOUND_COUNT_3H',
  'NORMALIZED_DIVERTED_INBOUND_COUNT_4H',
  'NORMALIZED_DIVERTED_INBOUND_COUNT_5H',
  'NORMALIZED_DIVERTED_INBOUND_COUNT_6H',
]

airline_numerical_columns = [
  'MAX_MEDIAN_OUTBOUND',
  'MAX_MEDIAN_INBOUND',
  'OUTBOUND_COUNT',
  'OUTBOUND_COUNT_0H',
  'OUTBOUND_COUNT_1H',
  'OUTBOUND_COUNT_2H',
  'OUTBOUND_COUNT_3H',
  'OUTBOUND_COUNT_4H',
  'OUTBOUND_COUNT_5H',
  'OUTBOUND_COUNT_6H',
  'NORMALIZED_OUTBOUND_COUNT_0H',
  'NORMALIZED_OUTBOUND_COUNT_1H',
  'NORMALIZED_OUTBOUND_COUNT_2H',
  'NORMALIZED_OUTBOUND_COUNT_3H',
  'NORMALIZED_OUTBOUND_COUNT_4H',
  'NORMALIZED_OUTBOUND_COUNT_5H',
  'NORMALIZED_OUTBOUND_COUNT_6H',
  'INBOUND_COUNT',
  'INBOUND_COUNT_0H',
  'INBOUND_COUNT_1H',
  'INBOUND_COUNT_2H',
  'INBOUND_COUNT_3H',
  'INBOUND_COUNT_4H',
  'INBOUND_COUNT_5H',
  'INBOUND_COUNT_6H',
  'NORMALIZED_INBOUND_COUNT_0H',
  'NORMALIZED_INBOUND_COUNT_1H',
  'NORMALIZED_INBOUND_COUNT_2H',
  'NORMALIZED_INBOUND_COUNT_3H',
  'NORMALIZED_INBOUND_COUNT_4H',
  'NORMALIZED_INBOUND_COUNT_5H',
  'NORMALIZED_INBOUND_COUNT_6H',
  'DELAY_OUTBOUND_COUNT_0H',
  'DELAY_OUTBOUND_COUNT_1H',
  'DELAY_OUTBOUND_COUNT_2H',
  'DELAY_OUTBOUND_COUNT_3H',
  'DELAY_OUTBOUND_COUNT_4H',
  'DELAY_OUTBOUND_COUNT_5H',
  'DELAY_OUTBOUND_COUNT_6H',
  'NORMALIZED_DELAY_OUTBOUND_COUNT_0H',
  'NORMALIZED_DELAY_OUTBOUND_COUNT_1H',
  'NORMALIZED_DELAY_OUTBOUND_COUNT_2H',
  'NORMALIZED_DELAY_OUTBOUND_COUNT_3H',
  'NORMALIZED_DELAY_OUTBOUND_COUNT_4H',
  'NORMALIZED_DELAY_OUTBOUND_COUNT_5H',
  'NORMALIZED_DELAY_OUTBOUND_COUNT_6H',
  'DELAY_INBOUND_COUNT_0H',
  'DELAY_INBOUND_COUNT_1H',
  'DELAY_INBOUND_COUNT_2H',
  'DELAY_INBOUND_COUNT_3H',
  'DELAY_INBOUND_COUNT_4H',
  'DELAY_INBOUND_COUNT_5H',
  'DELAY_INBOUND_COUNT_6H',
  'NORMALIZED_DELAY_INBOUND_COUNT_0H',
  'NORMALIZED_DELAY_INBOUND_COUNT_1H',
  'NORMALIZED_DELAY_INBOUND_COUNT_2H',
  'NORMALIZED_DELAY_INBOUND_COUNT_3H',
  'NORMALIZED_DELAY_INBOUND_COUNT_4H',
  'NORMALIZED_DELAY_INBOUND_COUNT_5H',
  'NORMALIZED_DELAY_INBOUND_COUNT_6H',
]

airline_time_columns = [
  'DEP_UTC_TIMESTAMP',
  'DEP_LOCAL_TIMESTAMP',
  'PREV_DEP_UTC_TIMESTAMP',
  'FL_DATE',
  'FL_DATETIME',
  'ARR_LOCAL_TIMESTAMP',
  'ARR_TIME_BLK',
  'DIV3_LONGEST_GTIME',
  'FL_DATETIMEHOUR_PLUS_2',
  'LONGEST_ADD_GTIME',
  'DIV2_TOTAL_GTIME',
  'ARR_TIME',
  'DIV4_LONGEST_GTIME',
  'ARR_TIME_DIFF',
  'DEP_LOCAL_TIMESTAMP_PLUS_2',
  'DIV1_LONGEST_GTIME',
  'FL_DATETIMEHOUR',
  'AIR_TIME',
  'ARR_UTC_TIMESTAMP_ADJUSTED',
  'ARR_TIME_DIFF2',
  'DIV3_TOTAL_GTIME',
  'DIV4_TOTAL_GTIME',
  'DIV5_TOTAL_GTIME',
  'DIV_ACTUAL_ELAPSED_TIME',
  'DIV5_LONGEST_GTIME',
  'ARR_UTC_TIMESTAMP',
  'CRS_ELAPSED_TIME'
  'DIV2_LONGEST_GTIME',
  'PREV_DEP_TIMEDELTA',
  'DEP_TIME_BLK',
  'DEP_TIME',
  'FIRST_DEP_TIME',
  'CRS_DEP_TIME',
  'TOTAL_ADD_GTIME',
  'CRS_ARR_TIME',
  'DIV1_TOTAL_GTIME',
  'ACTUAL_ELAPSED_TIME',
  'DIV2_LONGEST_GTIME',
  'DATE_DELTA4H',
  'DATE_DELTA2H',
  'CRS_ELAPSED_TIME',
]

airline_categorical_columns = [
  'MONTH',
  'DAY_OF_WEEK',
  'OP_UNIQUE_CARRIER',
  'call_sign_dep',
  'DEP_LOCAL_HOUR',
  'PREV_DEP_DELAY'
]

weather_join_columns = [
  'CALL_SIGN',
  'DATE1',
  'HOUR',
  'DATE_DELTA2H',
  'HOUR_DELTA2H',
  'DATE_DELTA4H',
  'HOUR_DELTA4H' 
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
  'AJ1_DIMENSION_AVG',
  'GA1_COVERAGE_CODE_AVG',
  'GA1_BASE_HEIGHT_DIMENSION_AVG',
  'KA1_MIN_PERIOD_QUANTITY_AVG',
  'KA1_MIN_AIR_TEMPERATURE_AVG',
  'KA1_MAX_PERIOD_QUANTITY_AVG',
  'KA1_MAX_AIR_TEMPERATURE_AVG',
  'WND_SPEED_AVG',
  'TMP_AIR_TEMPERATURE_AVG',
  'SLP_ATMOSPHERIC_PRESSURE_AVG'
]

pr_numerical_columns = [
  'ORIGIN_PR',
  'DEST_PR'
]

categorical_columns = airline_categorical_columns + weather_categorical_columns + pr_numerical_columns
numerical_columns = airline_numerical_columns + weather_numerical_columns + pr_numerical_columns

# COMMAND ----------

# MAGIC %md
# MAGIC # __Section 1__ - Load Data From Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from group25.baseline_model_data;

# COMMAND ----------

# MAGIC %sql
# MAGIC REFRESH TABLE group25.baseline_model_data;

# COMMAND ----------

# MAGIC %sql
# MAGIC REFRESH TABLE group25.weather_main

# COMMAND ----------

# MAGIC %sql
# MAGIC REFRESH TABLE group25.airports_pr_train

# COMMAND ----------

# Load airlines table
df = sqlContext.table("group25.baseline_model_data")


# COMMAND ----------

df.count()

# COMMAND ----------

df_weather = sqlContext.table("group25.weather_main")

# COMMAND ----------

df_weather.count()

# COMMAND ----------

df_pr = sqlContext.table("group25.airports_pr_train")

# COMMAND ----------

df_pr.count()
display(df_pr)

# COMMAND ----------

df_airline_names = sqlContext.table("group25.iata_to_airline_mapping")
display(df_airline_names)
airline_name_dict = {}
for index,rows in df_airline_names.toPandas().iterrows():
  airline_name_dict[rows['IATA_CODE']] = rows['AIRLINE']

# COMMAND ----------

def clean_data(df):
  #drop rows with all null values
  print(df.count())
  df = df.na.drop("all")
  # remove null target columns values
  print(df.count())
  df = df.na.drop(subset=["DEP_DEL15"])  
  print(df.count())
  return df

# Filter the columns
def filter_columns(df, filtered_columns):
  df = df.select(*filtered_columns)
  return df

def add_timedelta_columns(df):
  df = df.withColumn("DEP_LOCAL_TIMESTAMP_PLUS_2", df.DEP_LOCAL_TIMESTAMP + expr('INTERVAL 2 HOURS'))
  df = df.withColumn("DEP_LOCAL_HOUR_PLUS_2", hour(df.DEP_LOCAL_TIMESTAMP_PLUS_2))
  df = df.withColumn("MONTH_PLUS_2", month(df.DEP_LOCAL_TIMESTAMP_PLUS_2))
  df = df.withColumn("DAY_OF_WEEK_PLUS_2", dayofweek(df.DEP_LOCAL_TIMESTAMP_PLUS_2))
  df = df.withColumn("FL_DATETIMEHOUR", (unix_timestamp("FL_DATETIME") + col("DEP_LOCAL_HOUR")*60*60).cast('timestamp'))  
  df = df.withColumn("FL_DATETIMEHOUR_PLUS_2", df.FL_DATETIMEHOUR + expr('INTERVAL 2 HOURS') )
  return df

def add_flight_delay_propagation(df):
  df = df.withColumn('PREV_DEP_DELAY', 
                     lag(df['DEP_DELAY']).over(Window.partitionBy("TAIL_NUM") \
                                         .orderBy("FL_DATETIME", "DEP_UTC_TIMESTAMP"))) \
         .withColumn('PREV_DEP_UTC_TIMESTAMP', 
                     lag(df['DEP_UTC_TIMESTAMP']).over(Window.partitionBy("TAIL_NUM") \
                                                  .orderBy("FL_DATETIME", "DEP_UTC_TIMESTAMP"))) \
         .na.fill(value=0,subset=["PREV_DEP_DELAY"])
  df = df.withColumn("PREV_DEP_TIMEDELTA", 
                     (col("DEP_UTC_TIMESTAMP").cast("long")- col("PREV_DEP_UTC_TIMESTAMP").cast("long"))/60.0)
  df = df.withColumn("PREV_DEP_CUTOFF", when(col("PREV_DEP_TIMEDELTA") >= 120.0, 1).otherwise(0)) \
         .na.fill(value=0,subset=["PREV_DEP_TIMEDELTA"])
  df = df.withColumn("PREV_DEP_DELAY", when(col("PREV_DEP_CUTOFF")==0, 0).otherwise(col("PREV_DEP_DELAY")))
  return df

def add_past_performance(df):
    df_past_performance = df.groupBy("FL_DATETIMEHOUR_PLUS_2", "OP_UNIQUE_CARRIER", "call_sign_dep") \
                                    .agg(count("DEP_DEL15").alias("COUNT_FLIGHTS"), sum("DEP_DEL15").alias("NUM_DELAYS"), (sum("DEP_DEL15")/count("DEP_DEL15")).alias("FRACTION_DELAYS")) \
                                    .orderBy("call_sign_dep", "FL_DATETIMEHOUR_PLUS_2", "OP_UNIQUE_CARRIER")
    return df_past_performance

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Clean Airlines Data

# COMMAND ----------

display(df)

df = clean_data(df).cache()
df = add_timedelta_columns(df).cache()
df = add_flight_delay_propagation(df).cache()
df_past_performance = add_past_performance(df).cache()
# df = filter_columns(df, airline_filtered_columns).cache()

df.createOrReplaceTempView("airlinesTable") 
df_past_performance.createOrReplaceTempView("pastPerformanceTable")

# df_airline = sqlContext.sql("""
#   SELECT ta.*, tb.COUNT_FLIGHTS, tb.NUM_DELAYS, tb.FRACTION_DELAYS 
#   FROM airlinesTable as ta
#   LEFT JOIN pastPerformanceTable as tb
#   ON ta.OP_UNIQUE_CARRIER = tb.OP_UNIQUE_CARRIER
#   AND ta.FL_DATETIMEHOUR = tb.FL_DATETIMEHOUR_PLUS_2
#   AND ta.CALL_SIGN_DEP = tb.CALL_SIGN_DEP
# """)

# display(df_airline)

# COMMAND ----------

df_airline = sqlContext.sql("""
  SELECT ta.*, tb.COUNT_FLIGHTS, tb.NUM_DELAYS, tb.FRACTION_DELAYS 
  FROM airlinesTable as ta
  LEFT JOIN pastPerformanceTable as tb
  ON ta.OP_UNIQUE_CARRIER = tb.OP_UNIQUE_CARRIER
  AND ta.FL_DATETIMEHOUR = tb.FL_DATETIMEHOUR_PLUS_2
  AND ta.CALL_SIGN_DEP = tb.CALL_SIGN_DEP
""")

display(df_airline)

# COMMAND ----------

display(df_past_performance.where("call_sign_dep==\'KORD\'").where("OP_UNIQUE_CARRIER==\'UA\'").orderBy("FL_DATETIMEHOUR_PLUS_2"))

# COMMAND ----------

display(df.where("call_sign_dep==\'KORD\'").where("OP_UNIQUE_CARRIER==\'UA\'").orderBy("FL_DATETIMEHOUR"))

# COMMAND ----------

display(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Clean Weather Data

# COMMAND ----------

weather_filter_columns = weather_numerical_columns + weather_categorical_columns + weather_join_columns

df_weather_subset = df_weather.select(*weather_filter_columns)
df_weather_subset = df_weather_subset.withColumn('CALL_SIGN', trim(df_weather_subset.CALL_SIGN))
df_weather_subset = df_weather_subset.withColumn('FL_DATETIMEHOUR_PLUS_2',  (unix_timestamp("DATE_DELTA2H") + col("HOUR_DELTA2H")*60*60).cast('timestamp'))

display(df_weather_subset)
# imputed_df_weather = imputation_method1(weather_categorical_columns, weather_numerical_columns, df_weather, 'date1')
# display(imputed_df_weather)

# COMMAND ----------

display(df_weather_subset.where("CALL_SIGN==\'KORD\'").orderBy("FL_DATETIMEHOUR_PLUS_2"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Join Airline and Weather Tables
# MAGIC Here we join the airline table with the weather table lagged by 2 hours

# COMMAND ----------

# df = df.withColumn("FL_DATE", to_date(df.DEP_UTC_TIMESTAMP))
df_airline.createOrReplaceTempView("airline_temp") 
df_weather_subset.createOrReplaceTempView("weather_temp") 
df_merged = sqlContext.sql("""
  select a.*, 
     b.DATE_DELTA2H,
     b.HOUR_DELTA2H,
     b.DATE_DELTA4H,
     b.HOUR_DELTA4H,
     b.WND_DIRECTION_ANGLE_AVG_DIR,
     b.MV_THUNDERSTORM,
     b.MV_SHOWERS,
     b.MV_SAND_OR_DUST,
     b.MV_BLOWINGSNOW,
     b.MV_FOG,
     b.AA1_DEPTH_DIMENSION_AVG,
     b.AA1_PERIOD_QUANTITY_AVG,
     b.AJ1_DIMENSION_AVG,
     b.GA1_COVERAGE_CODE_AVG,
     b.GA1_BASE_HEIGHT_DIMENSION_AVG,
     b.KA1_MIN_PERIOD_QUANTITY_AVG,
     b.KA1_MIN_AIR_TEMPERATURE_AVG,
     b.KA1_MAX_PERIOD_QUANTITY_AVG,
     b.KA1_MAX_AIR_TEMPERATURE_AVG,
     b.WND_SPEED_AVG,
     b.TMP_AIR_TEMPERATURE_AVG,
     b.SLP_ATMOSPHERIC_PRESSURE_AVG
  FROM 
  (
    select * from airline_temp
  ) as a
  left join ( select * from weather_temp) as b
  on a.call_sign_dep=b.call_sign
  and a.dep_utc_hour = b.hour_delta2h
  and a.fl_date = b.date_delta2h
""")

display(df_merged)

# COMMAND ----------

df_merged.count()

# COMMAND ----------

display(df_merged)

# COMMAND ----------

# join pr's

df_merged.createOrReplaceTempView("weather_airline_temp")
df_pr.createOrReplaceTempView("pr_temp")

df_merged_pr = sqlContext.sql("""
  select ta.*, tbo.pagerank as ORIGIN_PR, tbd.pagerank as DEST_PR
  from weather_airline_temp ta
  left join pr_temp tbo
  on ta.origin = tbo.airport
  left join pr_temp tbd
  on ta.dest = tbd.airport""")

display(df_merged_pr)

# COMMAND ----------

df_merged_pr.createOrReplaceTempView("mytempTable") 
sqlContext.sql("DROP TABLE IF EXISTS group25.weather_airlines_utc_main_v1");
sqlContext.sql("create table group25.weather_airlines_utc_main_v1 as select * from mytempTable");

# COMMAND ----------

 df_merged_pr = sqlContext.sql("select * from group25.weather_airlines_utc_main_v1")
  
  
column_statistics_list = list(set(df_merged_pr.columns) - set(airline_time_columns))

for i in column_statistics_list:
  if "TIME" in i or "DATE" in i:
    print("'" + i+ "',")

# COMMAND ----------

df_merged_pr.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in column_statistics_list]).show()

# COMMAND ----------

# for i in categorical_columns:
#   if i in df_merged_pr.columns:
#     continue
#   else:
#     print(i)
    
# print('***********')
    
# for i in numerical_columns:
#   if i in df_merged_pr.columns:
#     continue
#   else:
#     print(i)

# COMMAND ----------

# MAGIC %md
# MAGIC #### compute nulls

# COMMAND ----------

#df_merged_pr = sqlContext.sql("select * from group25.weather_airlines_utc_main_v1")
  
  
column_statistics_list = airline_numerical_columns + weather_numerical_columns + pr_numerical_columns + airline_categorical_columns + weather_categorical_columns + airline_numerical_impute_zero

for i in column_statistics_list:
  if "TIME" in i or "DATE" in i:
    print("'" + i+ "',")
    
nulls_for_each_column = df_merged_pr.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in column_statistics_list])
nulls_for_each_column = nulls_for_each_column.withColumn('Total_Records', lit(df_merged_pr.count()))

# COMMAND ----------

nulls_for_each_column.createOrReplaceTempView("mytempTable") 
sqlContext.sql("DROP TABLE IF EXISTS group25.nulls_for_each_column_prior_to_impute");
sqlContext.sql("create table group25.nulls_for_each_column_prior_to_impute as select * from mytempTable");




# COMMAND ----------

nulls_for_each_column_pandas = nulls_for_each_column.toPandas().transpose()

# COMMAND ----------

print(nulls_for_each_column_pandas)

# COMMAND ----------

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

nulls = nulls_for_each_column_pandas.reset_index().sort_values(by=0, ascending = False)

nulls['Percentage of Records Missing'] = nulls[0] / (nulls[nulls['index'] == 'Total_Records'][0].values[0])


nulls.rename(columns = {'index':'Feature', 0: 'Num_of_Nulls'}, inplace = True)

print(nulls)

print(nulls.columns)


# COMMAND ----------

plt.figure(figsize=(24,24))
plt.xticks(rotation=90)
g = sns.barplot(x='Feature', y="Percentage of Records Missing", data=nulls)
g.set(ylim=(None, None))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Impute NULLs

# COMMAND ----------

# categorical_columns = airline_categorical_columns + weather_categorical_columns
# numerical_columns = airline_numerical_columns + weather_numerical_columns
# #join_columns = airline_join_columns + weather_join_columns

# imputed_df = imputation_using_medians(categorical_columns, numerical_columns, df_merged_pr, 'DATE1')
# display(imputed_df)

# COMMAND ----------

# #Discard unwanted columns

# unwanted_columns = [
  
# ]

# COMMAND ----------

# imputed_df.createOrReplaceTempView("mytempTable") 
# sqlContext.sql("DROP TABLE IF EXISTS group25.weather_airlines_utc_main_imputed_v1");
# sqlContext.sql("create table group25.weather_airlines_utc_main_imputed_v1 as select * from mytempTable");

# COMMAND ----------

df_merged_pr.createOrReplaceTempView("weather_airline_pr_temp")
df2 = sqlContext.sql("""
select * from weather_airline_pr_temp
""")
display(df2)


# COMMAND ----------

df_merged_pr.createOrReplaceTempView("weather_airline_pr_temp")
df2 = sqlContext.sql("""
select * from weather_airline_pr_temp
""")
display(df2)

for i in categorical_columns:
  if i in df2.columns:
    continue
  else:
    print(i)

print('***********')

for i in numerical_columns:
  if i in df2.columns:
    continue
  else:
    print(i)

query_string1 = ""
for i in airline_numerical_columns + weather_numerical_columns + pr_numerical_columns:
  query_string1 += "percentile_approx(" + i+", 0.5) as " + i + "_median1, "
#print(query_string[:-2])  
query_string2 = ""
for i in airline_numerical_columns + weather_numerical_columns + pr_numerical_columns:
  query_string2 += "percentile_approx(" + i+", 0.5) as " + i + "_median2, "
query_string3 = ""
for i in airline_numerical_columns + weather_numerical_columns + pr_numerical_columns:
  query_string3 += "percentile_approx(" + i+", 0.5) as " + i + "_median3, "
query_string4 = ""
for i in airline_numerical_columns + weather_numerical_columns + pr_numerical_columns:
  query_string4 += "percentile_approx(" + i+", 0.5) as " + i + "_median4, "
  
query =   """select 
    call_sign_dep, 
    MONTH, 
    DEP_LOCAL_HOUR, 
    {0}
    from weather_airline_pr_temp
    group by 1, 2, 3""".format(query_string1[:-2])

query2 =   """select 
    call_sign_dep, 
    MONTH, 
    {0}
    from weather_airline_pr_temp
    group by 1, 2""".format(query_string2[:-2])

query3 =   """select 
    call_sign_dep, 
    {0}
    from weather_airline_pr_temp
    group by 1""".format(query_string3[:-2])

query4 =   """select 
    {0}
    from weather_airline_pr_temp""".format(query_string4[:-2])
df_granularity1 = sqlContext.sql(query).cache()
df_granularity2 = sqlContext.sql(query2).cache()
df_granularity3 = sqlContext.sql(query3).cache()
df_granularity4 = sqlContext.sql(query4).collect()[0][0]


for i in set(airline_numerical_columns + weather_numerical_columns + pr_numerical_columns):
  df2 = df2.withColumn(i+'_median4', lit(df_granularity4))

df2 = df2.join(df_granularity1,['call_sign_dep', 'MONTH', 'DEP_LOCAL_HOUR'])
df2 = df2.join(df_granularity2,['call_sign_dep', 'MONTH'])
df2 = df2.join(df_granularity3,['call_sign_dep'])

# COMMAND ----------

df3 = df2
# Perform imputation. 
for i in airline_numerical_columns + weather_numerical_columns + pr_numerical_columns:
  df3 = df3.withColumn( i + '_IMPUTE', coalesce(df3[i], df3[i + "_median1"], df3[i + "_median2"], df3[ i + "_median3"], df3[i+'_median4']))

for j in airline_categorical_columns + weather_categorical_columns + pr_numerical_columns + airline_numerical_impute_zero:
  df3 = df3.withColumn(j + '_IMPUTE', coalesce(df3[j], lit(0)))

display(df3)

# COMMAND ----------

df3.createOrReplaceTempView("mytempTable") 
sqlContext.sql("DROP TABLE IF EXISTS group25.weather_airlines_utc_main_imputed_v2");
sqlContext.sql("create table group25.weather_airlines_utc_main_imputed_v2 as select * from mytempTable");

# COMMAND ----------

print(airline_numerical_columns + weather_numerical_columns + pr_numerical_columns)

# COMMAND ----------

query_string1

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from group25.weather_airlines_utc_main_imputed_v2

# COMMAND ----------

# %sql

# SELECT t1.*, 
# t1.ORIGIN_PR - t1.DEST_PR as PR_DIFF,

# COALESCE(aa_o.pageRank, 0) as PR_AA_ORIGIN,
# COALESCE(aa_d.pageRank, 0) as PR_AA_DEST,
# COALESCE(aa_o.pageRank - aa_d.pageRank, 0) as PR_AA_DIFF,

# COALESCE(aad_o.pageRank, 0) as PR_AAD_ORIGIN,
# COALESCE(aad_d.pageRank, 0) as PR_AAD_DEST,
# COALESCE(aad_o.pageRank - aad_d.pageRank, 0) as PR_AAD_DIFF,

# COALESCE(aadd_o.pageRank, 0) as PR_AADD_ORIGIN,
# COALESCE(aadd_d.pageRank, 0) as PR_AADD_DEST,
# COALESCE(aadd_o.pageRank - aadd_d.pageRank, 0) as PR_AADD_DIFF,

# COALESCE(f.pageRank, 0) AS PR_FL,
# COALESCE(fd.pageRank, 0) AS PR_FLD,
# COALESCE(fdh.pageRank, 0) AS PR_FLDH

# FROM
# group25.weather_airlines_utc_main_imputed_v2 t1
# LEFT JOIN group25.AIRPORTS_AIRLINE_PR_TRAIN aa_o ON aa_o.airport = t1.ORIGIN AND aa_o.airline = t1.OP_CARRIER
# LEFT JOIN group25.AIRPORTS_AIRLINE_PR_TRAIN aa_d ON aa_d.airport = t1.DEST AND aa_d.airline = t1.OP_CARRIER
# LEFT JOIN group25.AIRPORTS_AIRLINE_DOW_PR_TRAIN aad_o ON aad_o.airport = t1.ORIGIN AND aad_o.airline = t1.OP_CARRIER AND aad_o.DAY_OF_WEEK = t1.DAY_OF_WEEK
# LEFT JOIN group25.AIRPORTS_AIRLINE_DOW_PR_TRAIN aad_d ON aad_d.airport = t1.DEST AND aad_d.airline = t1.OP_CARRIER AND aad_d.DAY_OF_WEEK = t1.DAY_OF_WEEK
# LEFT JOIN group25.AIRPORTS_AIRLINE_DOW_DELAYS_PR_TRAIN aadd_o ON aadd_o.airport = t1.ORIGIN AND aadd_o.airline = t1.OP_CARRIER AND aadd_o.DAY_OF_WEEK = t1.DAY_OF_WEEK
# LEFT JOIN group25.AIRPORTS_AIRLINE_DOW_DELAYS_PR_TRAIN aadd_d ON aadd_d.airport = t1.DEST AND aadd_d.airline = t1.OP_CARRIER AND aadd_d.DAY_OF_WEEK = t1.DAY_OF_WEEK
# LEFT JOIN group25.FLIGHTS_PR_TRAIN f ON f.ORIGIN = t1.ORIGIN AND f.DEST = t1.DEST
# LEFT JOIN group25.FLIGHTS_DOW_PR_TRAIN fd ON fd.ORIGIN = t1.ORIGIN AND fd.DEST = t1.DEST AND fd.DAY_OF_WEEK = t1.DAY_OF_WEEK
# LEFT JOIN group25.FLIGHTS_DOW_HOUR_PR_TRAIN fdh ON fdh.ORIGIN = t1.ORIGIN AND fdh.DEST = t1.DEST AND fdh.DAY_OF_WEEK = t1.DAY_OF_WEEK AND fdh.HOUR = t1.DEP_UTC_HOUR
# LIMIT 1000


# COMMAND ----------

df4 = sqlContext.sql("""SELECT t1.*, 
  COALESCE(t1.ORIGIN_PR - t1.DEST_PR, 0) as PR_DIFF,

  COALESCE(aa_o.pageRank, 0) as PR_AA_ORIGIN,
  COALESCE(aa_d.pageRank, 0) as PR_AA_DEST,
  COALESCE(aa_o.pageRank - aa_d.pageRank, 0) as PR_AA_DIFF,

  COALESCE(aad_o.pageRank, 0) as PR_AAD_ORIGIN,
  COALESCE(aad_d.pageRank, 0) as PR_AAD_DEST,
  COALESCE(aad_o.pageRank - aad_d.pageRank, 0) as PR_AAD_DIFF,

  COALESCE(aadd_o.pageRank, 0) as PR_AADD_ORIGIN,
  COALESCE(aadd_d.pageRank, 0) as PR_AADD_DEST,
  COALESCE(aadd_o.pageRank - aadd_d.pageRank, 0) as PR_AADD_DIFF,

  COALESCE(f.pageRank, 0) AS PR_FL,
  COALESCE(fd.pageRank, 0) AS PR_FLD,
  COALESCE(fdh.pageRank, 0) AS PR_FLDH

  FROM
  group25.weather_airlines_utc_main_imputed_v2 t1
  LEFT JOIN group25.AIRPORTS_AIRLINE_PR_TRAIN aa_o ON aa_o.airport = t1.ORIGIN AND aa_o.airline = t1.OP_CARRIER
  LEFT JOIN group25.AIRPORTS_AIRLINE_PR_TRAIN aa_d ON aa_d.airport = t1.DEST AND aa_d.airline = t1.OP_CARRIER
  LEFT JOIN group25.AIRPORTS_AIRLINE_DOW_PR_TRAIN aad_o ON aad_o.airport = t1.ORIGIN AND aad_o.airline = t1.OP_CARRIER AND aad_o.DAY_OF_WEEK = t1.DAY_OF_WEEK
  LEFT JOIN group25.AIRPORTS_AIRLINE_DOW_PR_TRAIN aad_d ON aad_d.airport = t1.DEST AND aad_d.airline = t1.OP_CARRIER AND aad_d.DAY_OF_WEEK = t1.DAY_OF_WEEK
  LEFT JOIN group25.AIRPORTS_AIRLINE_DOW_DELAYS_PR_TRAIN aadd_o ON aadd_o.airport = t1.ORIGIN AND aadd_o.airline = t1.OP_CARRIER AND aadd_o.DAY_OF_WEEK = t1.DAY_OF_WEEK
  LEFT JOIN group25.AIRPORTS_AIRLINE_DOW_DELAYS_PR_TRAIN aadd_d ON aadd_d.airport = t1.DEST AND aadd_d.airline = t1.OP_CARRIER AND aadd_d.DAY_OF_WEEK = t1.DAY_OF_WEEK
  LEFT JOIN group25.FLIGHTS_PR_TRAIN f ON f.ORIGIN = t1.ORIGIN AND f.DEST = t1.DEST
  LEFT JOIN group25.FLIGHTS_DOW_PR_TRAIN fd ON fd.ORIGIN = t1.ORIGIN AND fd.DEST = t1.DEST AND fd.DAY_OF_WEEK = t1.DAY_OF_WEEK
  LEFT JOIN group25.FLIGHTS_DOW_HOUR_PR_TRAIN fdh ON fdh.ORIGIN = t1.ORIGIN AND fdh.DEST = t1.DEST AND fdh.DAY_OF_WEEK = t1.DAY_OF_WEEK AND fdh.HOUR = t1.DEP_UTC_HOUR
  """) 

df4.createOrReplaceTempView("mytempTable") 

sqlContext.sql("DROP TABLE IF EXISTS group25.weather_airlines_utc_main_imputed_v3");
sqlContext.sql("create table group25.weather_airlines_utc_main_imputed_v3 as select * from mytempTable");

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from group25.weather_airlines_utc_main_imputed_v3

# COMMAND ----------

