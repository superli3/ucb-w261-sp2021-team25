# Databricks notebook source
from pyspark.sql import functions as f
from pyspark.sql.functions import col, sum, avg, max, count, countDistinct, weekofyear, to_timestamp, date_format, to_date, lit, lag, unix_timestamp, expr, ceil, floor, when, hour, array
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType, TimestampType
from pyspark.sql import SQLContext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from distutils.version import LooseVersion
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
from pandas.tseries.holiday import USFederalHolidayCalendar
import datetime
from dateutil import parser, tz

spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")

# COMMAND ----------

# MAGIC %md
# MAGIC # Main Table: group25.airlines_utc_main

# COMMAND ----------

##################################################
df_airlines_utc_main = sqlContext.sql("""
SELECT * FROM group25.airlines_utc_main
""")

# COMMAND ----------

# MAGIC %md #Median

# COMMAND ----------

# MAGIC %md ##Max outbound median

# COMMAND ----------

##########################

# outbound count for each airport at each hour
# restrict to year 2015 - 2017

def compute_max_median_outbound_flights_per_airport():

  df_outbound_flights_per_airport_per_hour = sqlContext.sql("""
  SELECT
    CALL_SIGN_DEP,
    FL_DATE,
    DEP_LOCAL_HOUR,
    COUNT(*) AS hourly_outbound_count
  FROM group25.airlines_utc_main
  WHERE CALL_SIGN_DEP IS NOT NULL
  AND DEP_LOCAL_HOUR IS NOT NULL
  AND (YEAR=2015 OR YEAR=2016 OR YEAR=2017)
  GROUP BY 1, 2, 3
  ORDER BY 1, 2, 3 ASC
  """).cache()

  df_outbound_flights_per_airport_per_hour.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_outbound_flights_per_airport_per_hour");
  sqlContext.sql("create table group25.df_outbound_flights_per_airport_per_hour as select * from mytempTable");

  display(df_outbound_flights_per_airport_per_hour)


  # median of outbound
  from pyspark.sql import Window
  import pyspark.sql.functions as F

  grp_window = Window.partitionBy('CALL_SIGN_DEP', 'DEP_LOCAL_HOUR')
  magic_percentile = F.expr('percentile_approx(hourly_outbound_count, 0.5)')

  df_median_outbound_flights_per_airport = df_outbound_flights_per_airport_per_hour.withColumn('median_outbound_count', magic_percentile.over(grp_window))
  # df_median_outbound_flights_per_airport = df_median_outbound_flights_per_airport.drop('DEP_LOCAL_HOUR').drop('FL_DATE').drop('hourly_outbound_count').dropDuplicates()
  df_median_outbound_flights_per_airport = df_median_outbound_flights_per_airport.drop('FL_DATE').drop('hourly_outbound_count').dropDuplicates()


  df_median_outbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_median_outbound_flights_per_airport");
  sqlContext.sql("create table group25.df_median_outbound_flights_per_airport as select * from mytempTable");

  display(df_median_outbound_flights_per_airport.filter(df_median_outbound_flights_per_airport.CALL_SIGN_DEP=='KORD').orderBy("fl_date", "dep_local_hour"))


  ##### max median of each airport
  df_max_median_outbound_flights_per_airport = sqlContext.sql("""
  SELECT
    CALL_SIGN_DEP,
    MAX(median_outbound_count) AS max_median_outbound
  FROM group25.df_median_outbound_flights_per_airport
  GROUP BY 1
  ORDER BY 1
  """)

  df_max_median_outbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_max_median_outbound_flights_per_airport");
  sqlContext.sql("create table group25.df_max_median_outbound_flights_per_airport as select * from mytempTable");

  display(df_max_median_outbound_flights_per_airport)
  display(df_max_median_outbound_flights_per_airport.where("call_sign_dep==\'KORD\'").orderBy("CALL_SIGN_DEP"))
  return df_max_median_outbound_flights_per_airport

df_max_median_outbound_flights_per_airport = compute_max_median_outbound_flights_per_airport()

# COMMAND ----------

# display(df_max_median_outbound_flights_per_airport.filter(df_max_median_outbound_flights_per_airport.CALL_SIGN_DEP=='KORD').orderBy("CALL_SIGN_DEP"))

# COMMAND ----------

# MAGIC %md ##max_inbound_median

# COMMAND ----------

##########################

# inbound count for each airport at each hour
# restrict to year 2015 - 2017

def compute_max_median_inbound_flights_per_airport():

  df_inbound_flights_per_airport_per_hour = sqlContext.sql("""
  SELECT
    CALL_SIGN_DEP,
    FL_DATE,
    DEP_LOCAL_HOUR,
    COUNT(*) AS hourly_inbound_count
  FROM group25.airlines_utc_main
  WHERE CALL_SIGN_DEP IS NOT NULL
  AND DEP_LOCAL_HOUR IS NOT NULL
  AND (YEAR=2015 OR YEAR=2016 OR YEAR=2017)
  GROUP BY 1, 2, 3
  ORDER BY 1, 2, 3 ASC
  """).cache()

  df_inbound_flights_per_airport_per_hour.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_inbound_flights_per_airport_per_hour");
  sqlContext.sql("create table group25.df_inbound_flights_per_airport_per_hour as select * from mytempTable");

  display(df_inbound_flights_per_airport_per_hour)


  # median of inbound
  from pyspark.sql import Window
  import pyspark.sql.functions as F

  grp_window = Window.partitionBy('CALL_SIGN_DEP', 'DEP_LOCAL_HOUR')
  magic_percentile = F.expr('percentile_approx(hourly_inbound_count, 0.5)')

  df_median_inbound_flights_per_airport = df_inbound_flights_per_airport_per_hour.withColumn('median_inbound_count', magic_percentile.over(grp_window))
  # df_median_inbound_flights_per_airport = df_median_inbound_flights_per_airport.drop('DEP_LOCAL_HOUR').drop('FL_DATE').drop('hourly_inbound_count').dropDuplicates()
  df_median_inbound_flights_per_airport = df_median_inbound_flights_per_airport.drop('FL_DATE').drop('hourly_inbound_count').dropDuplicates()


  df_median_inbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_median_inbound_flights_per_airport");
  sqlContext.sql("create table group25.df_median_inbound_flights_per_airport as select * from mytempTable");

  display(df_median_inbound_flights_per_airport.filter(df_median_inbound_flights_per_airport.CALL_SIGN_DEP=='KORD').orderBy("fl_date", "dep_local_hour"))


  ##### max median of each airport
  df_max_median_inbound_flights_per_airport = sqlContext.sql("""
  SELECT
    CALL_SIGN_DEP,
    MAX(median_inbound_count) AS max_median_inbound
  FROM group25.df_median_inbound_flights_per_airport
  GROUP BY 1
  ORDER BY 1
  """)

  df_max_median_inbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_max_median_inbound_flights_per_airport");
  sqlContext.sql("create table group25.df_max_median_inbound_flights_per_airport as select * from mytempTable");

  display(df_max_median_inbound_flights_per_airport)
  display(df_max_median_inbound_flights_per_airport.where("call_sign_dep==\'KORD\'").orderBy("CALL_SIGN_DEP"))
  return df_max_median_inbound_flights_per_airport

df_max_median_inbound_flights_per_airport = compute_max_median_inbound_flights_per_airport()

# COMMAND ----------

# MAGIC %md
# MAGIC # Inbound & Outbound rolling window

# COMMAND ----------

# MAGIC %md
# MAGIC ##normalized outbound

# COMMAND ----------


#####################################################################
def compute_normalized_outbound_flights(df_max_median_outbound_flights_per_airport):

  # outbound rolling window

  hours = lambda i: i * 60 * 60

  airlines_outbound_flights_main_version_time_series_rolling_counts = None
  
  # select the dataframe
  df1 = sqlContext.sql("""
              SELECT
                call_sign_dep,
                fl_date,
                dep_local_hour,
                COUNT(*) AS outbound_count
              FROM group25.airlines_utc_main
              WHERE call_sign_dep IS NOT NULL
              AND dep_local_hour IS NOT NULL
              GROUP BY 1, 2, 3
              ORDER BY 1, 2, 3 ASC
              """)
  # make column fl_time and ast it into timestamp
  df1 = df1.withColumn("fl_time", unix_timestamp(col("fl_date").cast("timestamp")) + df1.dep_local_hour*60*60)
  df1 = df1.withColumn("fl_time", col("fl_time").cast("timestamp"))
  

  display(df1.filter(df1.call_sign_dep=='KORD').orderBy("FL_DATE", "dep_local_hour"))

  
  # calculate the rolling window
  # handle the 0th column
  df1 = df1.withColumn('outbound_count_0h', df1.outbound_count.cast('double'))
  
  # handle columns 1-6
  total_lag_time = 6
  for i in range(1, total_lag_time+1):  
    window = (Window.partitionBy("call_sign_dep", "fl_date").orderBy(col("fl_time").cast('long')).rangeBetween(-hours(int(i)), 0))
    df1 = df1.withColumn('outbound_count_' +str(i) + 'h', sum("outbound_count_0h").over(window))
  
  for i in range(total_lag_time, 0, -1):
    df1 = df1.withColumn('outbound_count_' +str(i) + 'h', col('outbound_count_' +str(i) + 'h') - col('outbound_count_' +str(i-1) + 'h'))
  
  # display(df1.filter(df1.call_sign_dep=='KORD').orderBy("fl_date", "dep_local_hour"))

  # save table
  df1.createOrReplaceTempView("mytempTable")
  airlines_outbound_flights_main_version_time_series_rolling_counts = df1
  sqlContext.sql("DROP TABLE IF EXISTS group25.airlines_outbound_flights_main_version_time_series_rolling_counts");
  sqlContext.sql("create table group25.airlines_outbound_flights_main_version_time_series_rolling_counts as select * from mytempTable");

  # display
  display(
    airlines_outbound_flights_main_version_time_series_rolling_counts
        .where("call_sign_dep==\'KORD\'")
        .orderBy("fl_date", "dep_local_hour"))



  ##### normalized outbound rolling window by max_median

    
  df1 = airlines_outbound_flights_main_version_time_series_rolling_counts

  df2 = df_max_median_outbound_flights_per_airport

  df_normalized_outbound_flights_per_airport=df1.alias("a").join(
      df2.alias("b"), df1['call_sign_dep'] == df2['call_sign_dep']
  ).select( 'a.call_sign_dep',
  'fl_date',
  'dep_local_hour',
  'outbound_count',
  'fl_time',
  'outbound_count_0h',
  'outbound_count_1h',
  'outbound_count_2h',
  'outbound_count_3h',
  'outbound_count_4h',
  'outbound_count_5h',
  'outbound_count_6h',
  'max_median_outbound'
          )

  for i in range(0, total_lag_time+1):
    df_normalized_outbound_flights_per_airport = df_normalized_outbound_flights_per_airport.withColumn(
                                                    'normalized_outbound_count_' +str(i) + 'h', 
                                                    col('outbound_count_' +str(i) + 'h').cast('int')/col('max_median_outbound')
                                                  )

  # display(df_normalized_outbound_flights_per_airport)
  display(df_normalized_outbound_flights_per_airport.filter(df_normalized_outbound_flights_per_airport.call_sign_dep=='KORD').orderBy("fl_date", "dep_local_hour"))
  
  df_normalized_outbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_normalized_outbound_flights_per_airport");
  sqlContext.sql("create table group25.df_normalized_outbound_flights_per_airport as select * from mytempTable");


#   ###### join normalized outbound rolling window with main table
#   df_left = airlines_utc_main

#   df_right = df_normalized_outbound_flights_per_airport

#   df_with_outbound=df_left.alias("a").join(
#       df_right.alias("b"), 
#       (df_left['call_sign_dep'] == df_right['call_sign_dep']) & (df_left['FL_DATETIME']== df_right['fl_time'])
#   ).select(
#   'a.*',

#   # 'b.call_sign_dep',
#   # 'b.fl_date',
#   # 'b.dep_local_hour',
#   'b.outbound_count',
#   # 'b.fl_time',
#   'b.outbound_count_0h',
#   'b.outbound_count_1h',
#   'b.outbound_count_2h',
#   'b.outbound_count_3h',
#   'b.outbound_count_4h',
#   'b.outbound_count_5h',
#   'b.outbound_count_6h',
#   'b.max_median_outbound',
#   'b.normalized_outbound_count_0h',
#   'b.normalized_outbound_count_1h',
#   'b.normalized_outbound_count_2h',
#   'b.normalized_outbound_count_3h',
#   'b.normalized_outbound_count_4h',
#   'b.normalized_outbound_count_5h',
#   'b.normalized_outbound_count_6h'
#   )


#   display(df_with_outbound)
#   df_with_outbound.createOrReplaceTempView("mytempTable") 
#   sqlContext.sql("DROP TABLE IF EXISTS group25.baseline_model_data_v1");
#   sqlContext.sql("create table group25.baseline_model_data_v1 as select * from mytempTable");


  # # showing the median outbound flight of airport KORD
  # display(df_median_outbound_flights_per_airport.where("call_sign_dep==\'KORD\'").orderBy("fl_date", "dep_local_hour"))
  return df_normalized_outbound_flights_per_airport

df_normalized_outbound_flights_per_airport = compute_normalized_outbound_flights(df_max_median_outbound_flights_per_airport)

# COMMAND ----------

# MAGIC %md ##normalized inbound

# COMMAND ----------

def compute_normalized_inbound_flights(df_max_median_inbound_flights_per_airport):

  # inbound rolling window

  hours = lambda i: i * 60 * 60

  # given a name to the rolling window
  airlines_inbound_flights_main_version_time_series_rolling_counts = None
  
  # select the dataframe
  df1 = sqlContext.sql("""
              SELECT
                call_sign_dep,
                fl_date,
                dep_local_hour,
                COUNT(*) AS inbound_count
              FROM group25.airlines_utc_main
              WHERE call_sign_dep IS NOT NULL
              AND dep_local_hour IS NOT NULL
              GROUP BY 1, 2, 3
              ORDER BY 1, 2, 3 ASC
              """)
  # make column fl_time and ast it into timestamp
  df1 = df1.withColumn("fl_time", unix_timestamp(col("fl_date").cast("timestamp")) + df1.dep_local_hour*60*60)
  df1 = df1.withColumn("fl_time", col("fl_time").cast("timestamp"))
  

  display(df1.filter(df1.call_sign_dep=='KORD').orderBy("FL_DATE", "dep_local_hour"))

  
  # calculate the rolling window
  # handle the 0th column
  df1 = df1.withColumn('inbound_count_0h', df1.inbound_count.cast('double'))
  
  # handle columns 1-6
  total_lag_time = 6
  for i in range(1, total_lag_time+1):  
    window = (Window.partitionBy("call_sign_dep", "fl_date").orderBy(col("fl_time").cast('long')).rangeBetween(-hours(int(i)), 0))
    df1 = df1.withColumn('inbound_count_' +str(i) + 'h', sum("inbound_count_0h").over(window))
  
  for i in range(total_lag_time, 0, -1):
    df1 = df1.withColumn('inbound_count_' +str(i) + 'h', col('inbound_count_' +str(i) + 'h') - col('inbound_count_' +str(i-1) + 'h'))
  
  # display(df1.filter(df1.call_sign_dep=='KORD').orderBy("fl_date", "dep_local_hour"))

  # save table
  df1.createOrReplaceTempView("mytempTable")
  airlines_inbound_flights_main_version_time_series_rolling_counts = df1
  sqlContext.sql("DROP TABLE IF EXISTS group25.airlines_inbound_flights_main_version_time_series_rolling_counts");
  sqlContext.sql("create table group25.airlines_inbound_flights_main_version_time_series_rolling_counts as select * from mytempTable");

  # display
  display(
    airlines_inbound_flights_main_version_time_series_rolling_counts
        .where("call_sign_dep==\'KORD\'")
        .orderBy("fl_date", "dep_local_hour"))



  ##### normalized inbound rolling window by max_median

  # Set df2 as rolling windown
  # set df3 as max median window
  df2 = airlines_inbound_flights_main_version_time_series_rolling_counts
  df3 = df_max_median_inbound_flights_per_airport

  # create the normalized window
  df_normalized_inbound_flights_per_airport=df2.alias("a").join(
      df3.alias("b"), df2['call_sign_dep'] == df3['call_sign_dep']
  ).select( 'a.call_sign_dep',
  'fl_date',
  'dep_local_hour',
  'inbound_count',
  'fl_time',
  'inbound_count_0h',
  'inbound_count_1h',
  'inbound_count_2h',
  'inbound_count_3h',
  'inbound_count_4h',
  'inbound_count_5h',
  'inbound_count_6h',
  'max_median_inbound'
          )

  for i in range(0, total_lag_time+1):
    df_normalized_inbound_flights_per_airport = df_normalized_inbound_flights_per_airport.withColumn(
                                                    'normalized_inbound_count_' +str(i) + 'h', 
                                                    col('inbound_count_' +str(i) + 'h').cast('int')/col('max_median_inbound')
                                                  )

  # display(df_normalized_inbound_flights_per_airport)
  display(df_normalized_inbound_flights_per_airport.filter(df_normalized_inbound_flights_per_airport.call_sign_dep=='KORD').orderBy("fl_date", "dep_local_hour"))
  
  df_normalized_inbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_normalized_inbound_flights_per_airport");
  sqlContext.sql("create table group25.df_normalized_inbound_flights_per_airport as select * from mytempTable");


#   ###### join normalized inbound rolling window with main table
#   df_left = airlines_utc_main

#   df_right = df_normalized_inbound_flights_per_airport

#   df_with_inbound=df_left.alias("a").join(
#       df_right.alias("b"), 
#       (df_left['call_sign_dep'] == df_right['call_sign_dep']) & (df_left['FL_DATETIME']== df_right['fl_time'])
#   ).select(
#   'a.*',

#   # 'b.call_sign_dep',
#   # 'b.fl_date',
#   # 'b.dep_local_hour',
#   'b.inbound_count',
#   # 'b.fl_time',
#   'b.inbound_count_0h',
#   'b.inbound_count_1h',
#   'b.inbound_count_2h',
#   'b.inbound_count_3h',
#   'b.inbound_count_4h',
#   'b.inbound_count_5h',
#   'b.inbound_count_6h',
#   'b.max_median_inbound',
#   'b.normalized_inbound_count_0h',
#   'b.normalized_inbound_count_1h',
#   'b.normalized_inbound_count_2h',
#   'b.normalized_inbound_count_3h',
#   'b.normalized_inbound_count_4h',
#   'b.normalized_inbound_count_5h',
#   'b.normalized_inbound_count_6h'
#   )


#   display(df_with_inbound)
#   df_with_inbound.createOrReplaceTempView("mytempTable") 
#   sqlContext.sql("DROP TABLE IF EXISTS group25.baseline_model_data_v1");
#   sqlContext.sql("create table group25.baseline_model_data_v1 as select * from mytempTable");


  # # showing the median inbound flight of airport KORD
  # display(df_median_inbound_flights_per_airport.where("call_sign_dep==\'KORD\'").orderBy("fl_date", "dep_local_hour"))
  return df_normalized_inbound_flights_per_airport

df_normalized_inbound_flights_per_airport = compute_normalized_inbound_flights(df_max_median_inbound_flights_per_airport)

# COMMAND ----------

# MAGIC %md #Diverted FLights

# COMMAND ----------

# MAGIC %md ##normalized diverted outbound

# COMMAND ----------


#####################################################################
def compute_normalized_diverted_outbound_flights(df_max_median_outbound_flights_per_airport):

  # diverted_outbound rolling window

  hours = lambda i: i * 60 * 60

  airlines_diverted_outbound_flights_main_version_time_series_rolling_counts = None
  
  # select the dataframe
  df1 = sqlContext.sql("""
              SELECT
                call_sign_dep,
                fl_date,
                dep_local_hour,
                COUNT(*) AS diverted_outbound_count
              FROM group25.airlines_utc_main
              WHERE call_sign_dep IS NOT NULL
              AND dep_local_hour IS NOT NULL
              AND diverted=1
              GROUP BY 1, 2, 3
              ORDER BY 1, 2, 3 ASC
              """)
  # make column fl_time and ast it into timestamp
  df1 = df1.withColumn("fl_time", unix_timestamp(col("fl_date").cast("timestamp")) + df1.dep_local_hour*60*60)
  df1 = df1.withColumn("fl_time", col("fl_time").cast("timestamp"))
  

  display(df1.filter(df1.call_sign_dep=='KORD').orderBy("FL_DATE", "dep_local_hour"))

  
  # calculate the rolling window
  # handle the 0th column
  df1 = df1.withColumn('diverted_outbound_count_0h', df1.diverted_outbound_count.cast('double'))
  
  # handle columns 1-6
  total_lag_time = 6
  for i in range(1, total_lag_time+1):  
    window = (Window.partitionBy("call_sign_dep", "fl_date").orderBy(col("fl_time").cast('long')).rangeBetween(-hours(int(i)), 0))
    df1 = df1.withColumn('diverted_outbound_count_' +str(i) + 'h', sum("diverted_outbound_count_0h").over(window))
  
  for i in range(total_lag_time, 0, -1):
    df1 = df1.withColumn('diverted_outbound_count_' +str(i) + 'h', col('diverted_outbound_count_' +str(i) + 'h') - col('diverted_outbound_count_' +str(i-1) + 'h'))
  
  # display(df1.filter(df1.call_sign_dep=='KORD').orderBy("fl_date", "dep_local_hour"))

  # save table
  df1.createOrReplaceTempView("mytempTable")
  airlines_diverted_outbound_flights_main_version_time_series_rolling_counts = df1
  sqlContext.sql("DROP TABLE IF EXISTS group25.airlines_diverted_outbound_flights_main_version_time_series_rolling_counts");
  sqlContext.sql("create table group25.airlines_diverted_outbound_flights_main_version_time_series_rolling_counts as select * from mytempTable");

  # display
  display(
    airlines_diverted_outbound_flights_main_version_time_series_rolling_counts
        .where("call_sign_dep==\'KORD\'")
        .orderBy("fl_date", "dep_local_hour"))



  ##### normalized diverted_outbound rolling window by max_median

    
  df1 = airlines_diverted_outbound_flights_main_version_time_series_rolling_counts

  df2 = df_max_median_diverted_outbound_flights_per_airport

  df_normalized_diverted_outbound_flights_per_airport=df1.alias("a").join(
      df2.alias("b"), df1['call_sign_dep'] == df2['call_sign_dep']
  ).select( 'a.call_sign_dep',
  'fl_date',
  'dep_local_hour',
  'diverted_outbound_count',
  'fl_time',
  'diverted_outbound_count_0h',
  'diverted_outbound_count_1h',
  'diverted_outbound_count_2h',
  'diverted_outbound_count_3h',
  'diverted_outbound_count_4h',
  'diverted_outbound_count_5h',
  'diverted_outbound_count_6h',
  'max_median_diverted_outbound'
          )

  for i in range(0, total_lag_time+1):
    df_normalized_diverted_outbound_flights_per_airport = df_normalized_diverted_outbound_flights_per_airport.withColumn(
                                                    'normalized_diverted_outbound_count_' +str(i) + 'h', 
                                                    col('diverted_outbound_count_' +str(i) + 'h').cast('int')/col('max_median_diverted_outbound')
                                                  )

  # display(df_normalized_diverted_outbound_flights_per_airport)
  display(df_normalized_diverted_outbound_flights_per_airport.filter(df_normalized_diverted_outbound_flights_per_airport.call_sign_dep=='KORD').orderBy("fl_date", "dep_local_hour"))
  
  df_normalized_diverted_outbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_normalized_diverted_outbound_flights_per_airport");
  sqlContext.sql("create table group25.df_normalized_diverted_outbound_flights_per_airport as select * from mytempTable");


#   ###### join normalized diverted_outbound rolling window with main table
#   df_left = airlines_utc_main

#   df_right = df_normalized_diverted_outbound_flights_per_airport

#   df_with_diverted_outbound=df_left.alias("a").join(
#       df_right.alias("b"), 
#       (df_left['call_sign_dep'] == df_right['call_sign_dep']) & (df_left['FL_DATETIME']== df_right['fl_time'])
#   ).select(
#   'a.*',

#   # 'b.call_sign_dep',
#   # 'b.fl_date',
#   # 'b.dep_local_hour',
#   'b.diverted_outbound_count',
#   # 'b.fl_time',
#   'b.diverted_outbound_count_0h',
#   'b.diverted_outbound_count_1h',
#   'b.diverted_outbound_count_2h',
#   'b.diverted_outbound_count_3h',
#   'b.diverted_outbound_count_4h',
#   'b.diverted_outbound_count_5h',
#   'b.diverted_outbound_count_6h',
#   'b.max_median_diverted_outbound',
#   'b.normalized_diverted_outbound_count_0h',
#   'b.normalized_diverted_outbound_count_1h',
#   'b.normalized_diverted_outbound_count_2h',
#   'b.normalized_diverted_outbound_count_3h',
#   'b.normalized_diverted_outbound_count_4h',
#   'b.normalized_diverted_outbound_count_5h',
#   'b.normalized_diverted_outbound_count_6h'
#   )


#   display(df_with_diverted_outbound)
#   df_with_diverted_outbound.createOrReplaceTempView("mytempTable") 
#   sqlContext.sql("DROP TABLE IF EXISTS group25.baseline_model_data_v1");
#   sqlContext.sql("create table group25.baseline_model_data_v1 as select * from mytempTable");


  # # showing the median diverted_outbound flight of airport KORD
  # display(df_median_diverted_outbound_flights_per_airport.where("call_sign_dep==\'KORD\'").orderBy("fl_date", "dep_local_hour"))
  return df_normalized_diverted_outbound_flights_per_airport

df_normalized_diverted_outbound_flights_per_airport = compute_normalized_diverted_outbound_flights(df_max_median_outbound_flights_per_airport)

# COMMAND ----------

# MAGIC %md ##normalized diverted inbound

# COMMAND ----------


#####################################################################
def compute_normalized_diverted_inbound_flights(df_max_median_inbound_flights_per_airport):

  # diverted_inbound rolling window

  hours = lambda i: i * 60 * 60

  airlines_diverted_inbound_flights_main_version_time_series_rolling_counts = None
  
  # select the dataframe
  df1 = sqlContext.sql("""
              SELECT
                call_sign_dep,
                fl_date,
                dep_local_hour,
                COUNT(*) AS diverted_inbound_count
              FROM group25.airlines_utc_main
              WHERE call_sign_dep IS NOT NULL
              AND dep_local_hour IS NOT NULL
              AND diverted=1
              GROUP BY 1, 2, 3
              ORDER BY 1, 2, 3 ASC
              """)
  # make column fl_time and ast it into timestamp
  df1 = df1.withColumn("fl_time", unix_timestamp(col("fl_date").cast("timestamp")) + df1.dep_local_hour*60*60)
  df1 = df1.withColumn("fl_time", col("fl_time").cast("timestamp"))
  

  display(df1.filter(df1.call_sign_dep=='KORD').orderBy("FL_DATE", "dep_local_hour"))

  
  # calculate the rolling window
  # handle the 0th column
  df1 = df1.withColumn('diverted_inbound_count_0h', df1.diverted_inbound_count.cast('double'))
  
  # handle columns 1-6
  total_lag_time = 6
  for i in range(1, total_lag_time+1):  
    window = (Window.partitionBy("call_sign_dep", "fl_date").orderBy(col("fl_time").cast('long')).rangeBetween(-hours(int(i)), 0))
    df1 = df1.withColumn('diverted_inbound_count_' +str(i) + 'h', sum("diverted_inbound_count_0h").over(window))
  
  for i in range(total_lag_time, 0, -1):
    df1 = df1.withColumn('diverted_inbound_count_' +str(i) + 'h', col('diverted_inbound_count_' +str(i) + 'h') - col('diverted_inbound_count_' +str(i-1) + 'h'))
  
  # display(df1.filter(df1.call_sign_dep=='KORD').orderBy("fl_date", "dep_local_hour"))

  # save table
  df1.createOrReplaceTempView("mytempTable")
  airlines_diverted_inbound_flights_main_version_time_series_rolling_counts = df1
  sqlContext.sql("DROP TABLE IF EXISTS group25.airlines_diverted_inbound_flights_main_version_time_series_rolling_counts");
  sqlContext.sql("create table group25.airlines_diverted_inbound_flights_main_version_time_series_rolling_counts as select * from mytempTable");

  # display
  display(
    airlines_diverted_inbound_flights_main_version_time_series_rolling_counts
        .where("call_sign_dep==\'KORD\'")
        .orderBy("fl_date", "dep_local_hour"))



  ##### normalized diverted_inbound rolling window by max_median

    
  df1 = airlines_diverted_inbound_flights_main_version_time_series_rolling_counts

  df2 = df_max_median_inbound_flights_per_airport

  df_normalized_diverted_inbound_flights_per_airport=df1.alias("a").join(
      df2.alias("b"), df1['call_sign_dep'] == df2['call_sign_dep']
  ).select( 'a.call_sign_dep',
  'fl_date',
  'dep_local_hour',
  'diverted_inbound_count',
  'fl_time',
  'diverted_inbound_count_0h',
  'diverted_inbound_count_1h',
  'diverted_inbound_count_2h',
  'diverted_inbound_count_3h',
  'diverted_inbound_count_4h',
  'diverted_inbound_count_5h',
  'diverted_inbound_count_6h',
  'max_median_inbound'
          )

  for i in range(0, total_lag_time+1):
    df_normalized_diverted_inbound_flights_per_airport = df_normalized_diverted_inbound_flights_per_airport.withColumn(
                                                    'normalized_diverted_inbound_count_' +str(i) + 'h', 
                                                    col('diverted_inbound_count_' +str(i) + 'h').cast('int')/col('max_median_inbound')
                                                  )

  # display(df_normalized_diverted_inbound_flights_per_airport)
  display(df_normalized_diverted_inbound_flights_per_airport.filter(df_normalized_diverted_inbound_flights_per_airport.call_sign_dep=='KORD').orderBy("fl_date", "dep_local_hour"))
  
  df_normalized_diverted_inbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_normalized_diverted_inbound_flights_per_airport");
  sqlContext.sql("create table group25.df_normalized_diverted_inbound_flights_per_airport as select * from mytempTable");


#   ###### join normalized diverted_inbound rolling window with main table
#   df_left = airlines_utc_main

#   df_right = df_normalized_diverted_inbound_flights_per_airport

#   df_with_diverted_inbound=df_left.alias("a").join(
#       df_right.alias("b"), 
#       (df_left['call_sign_dep'] == df_right['call_sign_dep']) & (df_left['FL_DATETIME']== df_right['fl_time'])
#   ).select(
#   'a.*',

#   # 'b.call_sign_dep',
#   # 'b.fl_date',
#   # 'b.dep_local_hour',
#   'b.diverted_inbound_count',
#   # 'b.fl_time',
#   'b.diverted_inbound_count_0h',
#   'b.diverted_inbound_count_1h',
#   'b.diverted_inbound_count_2h',
#   'b.diverted_inbound_count_3h',
#   'b.diverted_inbound_count_4h',
#   'b.diverted_inbound_count_5h',
#   'b.diverted_inbound_count_6h',
#   'b.max_median_inbound',
#   'b.normalized_diverted_inbound_count_0h',
#   'b.normalized_diverted_inbound_count_1h',
#   'b.normalized_diverted_inbound_count_2h',
#   'b.normalized_diverted_inbound_count_3h',
#   'b.normalized_diverted_inbound_count_4h',
#   'b.normalized_diverted_inbound_count_5h',
#   'b.normalized_diverted_inbound_count_6h'
#   )


#   display(df_with_diverted_inbound)
#   df_with_diverted_inbound.createOrReplaceTempView("mytempTable") 
#   sqlContext.sql("DROP TABLE IF EXISTS group25.baseline_model_data_v1");
#   sqlContext.sql("create table group25.baseline_model_data_v1 as select * from mytempTable");


  # # showing the median diverted_inbound flight of airport KORD
  # display(df_median_inbound_flights_per_airport.where("call_sign_dep==\'KORD\'").orderBy("fl_date", "dep_local_hour"))
  return df_normalized_diverted_inbound_flights_per_airport

df_normalized_diverted_inbound_flights_per_airport = compute_normalized_diverted_inbound_flights(df_max_median_inbound_flights_per_airport)

# COMMAND ----------

# MAGIC %md # Delay Flights

# COMMAND ----------

# MAGIC %md ##normalized delay outbound flights

# COMMAND ----------

############################################################################################
# dalay outbound flights
## calculate the number of delays per airpor per hour per airline - departure/outbound

def compute_normalized_delay_outbound_flights(df_max_median_outbound_flights_per_airport):
  #function to calculate number of seconds from number of days
  hours = lambda i: i * 60 * 60
  df_delay_outbound_flights_per_airport_per_hour = sqlContext.sql("""
          select
            call_sign_dep,
            fl_date,
            dep_local_hour,
            hour(dep_local_timestamp + INTERVAL 2 HOUR) as dep_local_hour_2h,
            sum(dep_del15) as delay_outbound_count_0h
            from group25.airlines_utc_main
            where dep_del15 is not null
            group by 1,2,3,4
              """)
  df_delay_outbound_flights_per_airport_per_hour = df_delay_outbound_flights_per_airport_per_hour.withColumn("fl_time", unix_timestamp(col("fl_date").cast("timestamp")) + df_delay_outbound_flights_per_airport_per_hour.dep_local_hour*60*60)
  df_delay_outbound_flights_per_airport_per_hour = df_delay_outbound_flights_per_airport_per_hour.withColumn("fl_time", col("fl_time").cast("timestamp"))
  display(df_delay_outbound_flights_per_airport_per_hour.filter(df_delay_outbound_flights_per_airport_per_hour.call_sign_dep=='KORD').orderBy("fl_date", "dep_local_hour"))


  # building the delays columns
  total_lag_time = 6
  for i in range(1, total_lag_time+1):  
    window = (Window.partitionBy("fl_date", "call_sign_dep").orderBy(col("fl_time").cast('long')).rangeBetween(-hours(int(i)), 0))
    df_delay_outbound_flights_per_airport_per_hour = df_delay_outbound_flights_per_airport_per_hour.withColumn('delay_outbound_count_' +str(i) + 'h', sum("delay_outbound_count_0h").over(window))

  for i in range(total_lag_time, 0, -1):
    df_delay_outbound_flights_per_airport_per_hour = df_delay_outbound_flights_per_airport_per_hour.withColumn('delay_outbound_count_' +str(i) + 'h', col('delay_outbound_count_' +str(i) + 'h') - col('delay_outbound_count_' +str(i-1) + 'h'))

  display(df_delay_outbound_flights_per_airport_per_hour.filter(df_delay_outbound_flights_per_airport_per_hour.call_sign_dep=='KORD').orderBy("fl_date", "dep_local_hour"))

  df_delay_outbound_flights_per_airport_per_hour.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_delayed_outbound_flights_per_airport_per_hour_rolling_window");
  sqlContext.sql("create table group25.df_delayed_outbound_flights_per_airport_per_hour_rolling_window as select * from mytempTable");


  ##### normalized outbound rolling window by max_median

  df2 = df_max_median_outbound_flights_per_airport

  df_normalized_delay_outbound_flights_per_airport=df_delay_outbound_flights_per_airport_per_hour.alias("a").join(
      df2.alias("b"), df_delay_outbound_flights_per_airport_per_hour['call_sign_dep'] == df2['call_sign_dep']
  ).select( 'a.call_sign_dep',
  'fl_date',
  'dep_local_hour',
  #  'dep_local_hour_2h',
  'fl_time',
  'delay_outbound_count_0h',
  'delay_outbound_count_1h',
  'delay_outbound_count_2h',
  'delay_outbound_count_3h',
  'delay_outbound_count_4h',
  'delay_outbound_count_5h',
  'delay_outbound_count_6h',
  'max_median_outbound'
          )

  for i in range(0, total_lag_time+1):
    df_normalized_delay_outbound_flights_per_airport = df_normalized_delay_outbound_flights_per_airport.withColumn('normalized_delay_outbound_count_' +str(i) + 'h', col('delay_outbound_count_' +str(i) + 'h').cast('int')/col('max_median_outbound'))

  # display(df_normalized_delay_outbound_flights_per_airport)
  display(df_normalized_delay_outbound_flights_per_airport.filter(df_normalized_delay_outbound_flights_per_airport.call_sign_dep=='KORD').orderBy("FL_DATE", "dep_local_hour"))
  
  df_normalized_delay_outbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_normalized_delay_outbound_flights_per_airport");
  sqlContext.sql("create table group25.df_normalized_delay_outbound_flights_per_airport as select * from mytempTable");


#   ###### join normalized outbound rolling window with main table
#   df_left = sqlContext.sql(
#   """
#   SELECT * FROM group25.baseline_model_data
#   """)

#   df_right = sqlContext.sql(
#   """
#   SELECT * FROM group25.df_normalized_delay_outbound_flights_per_airport
#   """)

#   df_with_delay_outbound=df_left.alias("a").join(
#       df_right.alias("b"), 
#       (df_left['call_sign_dep'] == df_left['call_sign_dep']) & (df_left['FL_DATETIME']== df_left['fl_time'])
#   ).select(
#   'a.*',
#   # df_left has no fl_datetime. It only has FL_DATE. 

#   # 'b.call_sign_dep',
#   # 'b.fl_date',
#   # 'b.dep_local_hour',
#   # 'b.delay_outbound_count',
#   # 'b.fl_time',
#   'b.delay_outbound_count_0h',
#   'b.delay_outbound_count_1h',
#   'b.delay_outbound_count_2h',
#   'b.delay_outbound_count_3h',
#   'b.delay_outbound_count_4h',
#   'b.delay_outbound_count_5h',
#   'b.delay_outbound_count_6h',
#   # 'b.max_median_outbound',
#   'b.normalized_delay_outbound_count_0h',
#   'b.normalized_delay_outbound_count_1h',
#   'b.normalized_delay_outbound_count_2h',
#   'b.normalized_delay_outbound_count_3h',
#   'b.normalized_delay_outbound_count_4h',
#   'b.normalized_delay_outbound_count_5h',
#   'b.normalized_delay_outbound_count_6h'
#   )


#   # display(df_with_delay_outbound)
#   display(df_with_delay_outbound.filter(df_with_delay_outbound.call_sign_dep=='KORD').orderBy("FL_DATE", "dep_local_hour"))

#   df_temp.createOrReplaceTempView("mytempTable") 
#   sqlContext.sql("DROP TABLE IF EXISTS group25.baseline_model_data");
#   sqlContext.sql("create table group25.baseline_model_data as select * from mytempTable");


#   # showing the median outbound flight of airport KORD
#   display(df_median_outbound_flights_per_airport.where("call_sign_dep==\'KORD\'").orderBy("fl_date", "dep_local_hour"))
  
  return df_normalized_delay_outbound_flights_per_airport

df_normalized_delay_outbound_flights_per_airport = compute_normalized_delay_outbound_flights(df_max_median_outbound_flights_per_airport)


# COMMAND ----------

# MAGIC %md ##normalized delay outbound flights

# COMMAND ----------

############################################################################################
# dalay inbound flights
## calculate the number of delays per airpor per hour per airline - departure/inbound

def compute_normalized_delay_inbound_flights(df_max_median_inbound_flights_per_airport):
  #function to calculate number of seconds from number of days
  hours = lambda i: i * 60 * 60
  df_delay_inbound_flights_per_airport_per_hour = sqlContext.sql("""
          select
            call_sign_dep,
            fl_date,
            dep_local_hour,
            hour(dep_local_timestamp + INTERVAL 2 HOUR) as dep_local_hour_2h,
            sum(dep_del15) as delay_inbound_count_0h
            from group25.airlines_utc_main
            where dep_del15 is not null
            group by 1,2,3,4
              """)
  df_delay_inbound_flights_per_airport_per_hour = df_delay_inbound_flights_per_airport_per_hour.withColumn("fl_time", unix_timestamp(col("fl_date").cast("timestamp")) + df_delay_inbound_flights_per_airport_per_hour.dep_local_hour*60*60)
  df_delay_inbound_flights_per_airport_per_hour = df_delay_inbound_flights_per_airport_per_hour.withColumn("fl_time", col("fl_time").cast("timestamp"))
  display(df_delay_inbound_flights_per_airport_per_hour.filter(df_delay_inbound_flights_per_airport_per_hour.call_sign_dep=='KORD').orderBy("fl_date", "dep_local_hour"))


  # building the delays columns
  total_lag_time = 6
  for i in range(1, total_lag_time+1):  
    window = (Window.partitionBy("fl_date", "call_sign_dep").orderBy(col("fl_time").cast('long')).rangeBetween(-hours(int(i)), 0))
    df_delay_inbound_flights_per_airport_per_hour = df_delay_inbound_flights_per_airport_per_hour.withColumn('delay_inbound_count_' +str(i) + 'h', sum("delay_inbound_count_0h").over(window))

  for i in range(total_lag_time, 0, -1):
    df_delay_inbound_flights_per_airport_per_hour = df_delay_inbound_flights_per_airport_per_hour.withColumn('delay_inbound_count_' +str(i) + 'h', col('delay_inbound_count_' +str(i) + 'h') - col('delay_inbound_count_' +str(i-1) + 'h'))

  display(df_delay_inbound_flights_per_airport_per_hour.filter(df_delay_inbound_flights_per_airport_per_hour.call_sign_dep=='KORD').orderBy("fl_date", "dep_local_hour"))

  df_delay_inbound_flights_per_airport_per_hour.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_delayed_inbound_flights_per_airport_per_hour_rolling_window");
  sqlContext.sql("create table group25.df_delayed_inbound_flights_per_airport_per_hour_rolling_window as select * from mytempTable");


  ##### normalized inbound rolling window by max_median

  df2 = df_max_median_inbound_flights_per_airport

  df_normalized_delay_inbound_flights_per_airport=df_delay_inbound_flights_per_airport_per_hour.alias("a").join(
      df2.alias("b"), df_delay_inbound_flights_per_airport_per_hour['call_sign_dep'] == df2['call_sign_dep']
  ).select( 'a.call_sign_dep',
  'fl_date',
  'dep_local_hour',
  #  'dep_local_hour_2h',
  'fl_time',
  'delay_inbound_count_0h',
  'delay_inbound_count_1h',
  'delay_inbound_count_2h',
  'delay_inbound_count_3h',
  'delay_inbound_count_4h',
  'delay_inbound_count_5h',
  'delay_inbound_count_6h',
  'max_median_inbound'
          )

  for i in range(0, total_lag_time+1):
    df_normalized_delay_inbound_flights_per_airport = df_normalized_delay_inbound_flights_per_airport.withColumn('normalized_delay_inbound_count_' +str(i) + 'h', col('delay_inbound_count_' +str(i) + 'h').cast('int')/col('max_median_inbound'))

  # display(df_normalized_delay_inbound_flights_per_airport)
  display(df_normalized_delay_inbound_flights_per_airport.filter(df_normalized_delay_inbound_flights_per_airport.call_sign_dep=='KORD').orderBy("FL_DATE", "dep_local_hour"))
  
  df_normalized_delay_inbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_normalized_delay_inbound_flights_per_airport");
  sqlContext.sql("create table group25.df_normalized_delay_inbound_flights_per_airport as select * from mytempTable");


#   ###### join normalized inbound rolling window with main table
#   df_left = sqlContext.sql(
#   """
#   SELECT * FROM group25.baseline_model_data
#   """)

#   df_right = sqlContext.sql(
#   """
#   SELECT * FROM group25.df_normalized_delay_inbound_flights_per_airport
#   """)

#   df_with_delay_inbound=df_left.alias("a").join(
#       df_right.alias("b"), 
#       (df_left['call_sign_dep'] == df_left['call_sign_dep']) & (df_left['FL_DATETIME']== df_left['fl_time'])
#   ).select(
#   'a.*',
#   # df_left has no fl_datetime. It only has FL_DATE. 

#   # 'b.call_sign_dep',
#   # 'b.fl_date',
#   # 'b.dep_local_hour',
#   # 'b.delay_inbound_count',
#   # 'b.fl_time',
#   'b.delay_inbound_count_0h',
#   'b.delay_inbound_count_1h',
#   'b.delay_inbound_count_2h',
#   'b.delay_inbound_count_3h',
#   'b.delay_inbound_count_4h',
#   'b.delay_inbound_count_5h',
#   'b.delay_inbound_count_6h',
#   # 'b.max_median_inbound',
#   'b.normalized_delay_inbound_count_0h',
#   'b.normalized_delay_inbound_count_1h',
#   'b.normalized_delay_inbound_count_2h',
#   'b.normalized_delay_inbound_count_3h',
#   'b.normalized_delay_inbound_count_4h',
#   'b.normalized_delay_inbound_count_5h',
#   'b.normalized_delay_inbound_count_6h'
#   )


#   # display(df_with_delay_inbound)
#   display(df_with_delay_inbound.filter(df_with_delay_inbound.call_sign_dep=='KORD').orderBy("FL_DATE", "dep_local_hour"))

#   df_temp.createOrReplaceTempView("mytempTable") 
#   sqlContext.sql("DROP TABLE IF EXISTS group25.baseline_model_data");
#   sqlContext.sql("create table group25.baseline_model_data as select * from mytempTable");


#   # showing the median inbound flight of airport KORD
#   display(df_median_inbound_flights_per_airport.where("call_sign_dep==\'KORD\'").orderBy("fl_date", "dep_local_hour"))
  
  return df_normalized_delay_inbound_flights_per_airport

df_normalized_delay_inbound_flights_per_airport = compute_normalized_delay_inbound_flights(df_max_median_inbound_flights_per_airport)


# COMMAND ----------

# MAGIC %md # Join all table

# COMMAND ----------

# def join_tables(airlines_utc_main, normalized_outbound):
#   baseline_model_data = airlines_utc_main.join(
#     normalized_outbound, 
#     on=["FL_DATE"],
#     how='left')
#   return baseline_model_data

# baseline_model_data = join_tables(df_airlines_utc_main, df_normalized_outbound_flights_per_airport)
# display(baseline_model_data.filter(airlines_utc_main.call_sign_dep=='KORD').orderBy("fl_date", "airlines_utc_main.DEP_LOCAL_HOUR"))


def call_join_tables(df_airlines_utc_main, df_normalized_outbound_flights_per_airport, df_normalized_delay_outbound_flights_per_airport):
  d1 = df_airlines_utc_main.filter(airlines_utc_main.call_sign_dep=='KORD')
  d2 = df_normalized_outbound_flights_per_airport.filter(airlines_utc_main.call_sign_dep=='KORD')
  d3 = df_normalized_delay_outbound_flights_per_airport.filter(airlines_utc_main.call_sign_dep=='KORD')
  return join_tables(d1, d2, d3)

baseline_model_data = call_join_tables(df_airlines_utc_main, df_normalized_outbound_flights_per_airport, df_normalized_delay_outbound_flights_per_airport, )
display(baseline_model_data.filter(airlines_utc_main.call_sign_dep=='KORD').orderBy("fl_date", "airlines_utc_main.DEP_LOCAL_HOUR"))


# COMMAND ----------

def join_tables(airlines_utc_main, normalized_outbound):
  baseline_model_data = airlines_utc_main.join(normalized_outbound, (airlines_utc_main["FL_DATE"] == normalized_outbound["FL_DATE"]) & (airlines_utc_main["FL_DATE"] == normalized_outbound["FL_DATE"]), how='left')
  return baseline_model_data

# baseline_model_data = join_tables(df_airlines_utc_main, df_normalized_outbound_flights_per_airport)
# display(baseline_model_data.filter(airlines_utc_main.call_sign_dep=='KORD').orderBy("fl_date", "airlines_utc_main.DEP_LOCAL_HOUR"))



# COMMAND ----------

def join_tables(airlines_utc_main, normalized_outbound, normalized_delay_outbound):
  baseline_model_data = airlines_utc_main.join(
      normalized_outbound, 
      (airlines_utc_main["FL_DATE"] == normalized_outbound["FL_DATE"]) & (airlines_utc_main["dep_local_hour"] == normalized_outbound["dep_local_hour"]), 
      how='left')
#   .join(
#       normalized_delay_outbound, 
#       (airlines_utc_main["FL_DATE"] == normalized_delay_outbound["FL_DATE"]) & (airlines_utc_main["dep_local_hour"] == normalized_delay_outbound["dep_local_hour"]), 
#       how='left'
#     )
  
  return baseline_model_data

# baseline_model_data = join_tables(df_airlines_utc_main, df_normalized_outbound_flights_per_airport, df_normalized_delay_outbound_flights_per_airport)
# display(baseline_model_data.filter(airlines_utc_main.call_sign_dep=='KORD').orderBy("fl_date", "airlines_utc_main.DEP_LOCAL_HOUR"))

def call_join_tables(df_airlines_utc_main, df_normalized_outbound_flights_per_airport, df_normalized_delay_outbound_flights_per_airport):
  d1 = df_airlines_utc_main.filter(airlines_utc_main.call_sign_dep=='KORD')
  d2 = df_normalized_outbound_flights_per_airport.filter(airlines_utc_main.call_sign_dep=='KORD')
  d3 = df_normalized_delay_outbound_flights_per_airport.filter(airlines_utc_main.call_sign_dep=='KORD')
  return join_tables(d1, d2, d3)

baseline_model_data = call_join_tables(df_airlines_utc_main, df_normalized_outbound_flights_per_airport, df_normalized_delay_outbound_flights_per_airport, )
display(baseline_model_data.filter(airlines_utc_main.call_sign_dep=='KORD').orderBy(airlines_utc_main.FL_DATE, "airlines_utc_main.DEP_LOCAL_HOUR"))

baseline_model_data.createOrReplaceTempView("mytempTable") 
sqlContext.sql("DROP TABLE IF EXISTS group25.baseline_model_data");
sqlContext.sql("create table group25.baseline_model_data as select * from mytempTable");

# COMMAND ----------

def join_tables(airlines_utc_main, normalized_outbound):
  baseline_model_data = airlines_utc_main.join(
    normalized_outbound, 
    ["FL_DATE", "DEP_LOCAL_HOUR"],
    how='left')
  return baseline_model_data

# baseline_model_data = join_tables(df_airlines_utc_main, df_normalized_outbound_flights_per_airport)
# display(baseline_model_data.filter(baseline_model_data.CALL_SIGN_DEP=='KORD').orderBy("fl_date", "dep_local_hour"))

def call_join_tables(df_airlines_utc_main, df_normalized_outbound_flights_per_airport, df_normalized_delay_outbound_flights_per_airport):
  d1 = df_airlines_utc_main.filter(airlines_utc_main.call_sign_dep=='KORD')
  d2 = df_normalized_outbound_flights_per_airport.filter(airlines_utc_main.call_sign_dep=='KORD')
  d3 = df_normalized_delay_outbound_flights_per_airport.filter(airlines_utc_main.call_sign_dep=='KORD')
  return join_tables(d1, d2)

baseline_model_data = call_join_tables(df_airlines_utc_main, df_normalized_outbound_flights_per_airport, df_normalized_delay_outbound_flights_per_airport, )
display(baseline_model_data.filter(airlines_utc_main.call_sign_dep=='KORD').orderBy(airlines_utc_main.FL_DATE, "airlines_utc_main.DEP_LOCAL_HOUR"))


# COMMAND ----------

