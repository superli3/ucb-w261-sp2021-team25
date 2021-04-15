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

  # compute outbound flights /airport/fl_date/hour
  df_outbound_flights_per_airport_per_hour = sqlContext.sql("""
  SELECT
    CALL_SIGN_DEP,
    FL_DATE,
    DEP_LOCAL_HOUR,
    COUNT(*) AS HOURLY_OUTBOUND_COUNT
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


  # median of outbound for each airport by each hour
  from pyspark.sql import Window
  import pyspark.sql.functions as F

  grp_window = Window.partitionBy('CALL_SIGN_DEP', 'DEP_LOCAL_HOUR')
  magic_percentile = F.expr('percentile_approx(HOURLY_OUTBOUND_COUNT, 0.5)')

  df_median_outbound_flights_per_airport = df_outbound_flights_per_airport_per_hour.withColumn('MEDIAN_OUTBOUND_COUNT', magic_percentile.over(grp_window))
  df_median_outbound_flights_per_airport = df_median_outbound_flights_per_airport.drop('FL_DATE').drop('HOURLY_OUTBOUND_COUNT').dropDuplicates()


  df_median_outbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_median_outbound_flights_per_airport");
  sqlContext.sql("create table group25.df_median_outbound_flights_per_airport as select * from mytempTable");

  display(df_median_outbound_flights_per_airport.filter(df_median_outbound_flights_per_airport.CALL_SIGN_DEP=='KORD').orderBy("FL_DATE", "DEP_LOCAL_HOUR"))


  ##### max median of each airport
  df_max_median_outbound_flights_per_airport = sqlContext.sql("""
  SELECT
    CALL_SIGN_DEP,
    MAX(MEDIAN_OUTBOUND_COUNT) AS MAX_MEDIAN_OUTBOUND
  FROM group25.df_median_outbound_flights_per_airport
  GROUP BY 1
  ORDER BY 1
  """)

  df_max_median_outbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_max_median_outbound_flights_per_airport");
  sqlContext.sql("create table group25.df_max_median_outbound_flights_per_airport as select * from mytempTable");

  display(df_max_median_outbound_flights_per_airport)
  display(df_max_median_outbound_flights_per_airport.where("CALL_SIGN_DEP==\'KORD\'").orderBy("CALL_SIGN_DEP"))
  return df_max_median_outbound_flights_per_airport

df_max_median_outbound_flights_per_airport = compute_max_median_outbound_flights_per_airport()

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from group25.df_max_median_outbound_flights_per_airport

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
    CALL_SIGN_ARR,
    FL_DATE,
    ARR_LOCAL_HOUR,
    COUNT(*) AS HOURLY_INBOUND_COUNT
  FROM group25.airlines_utc_main
  WHERE CALL_SIGN_ARR IS NOT NULLSpark
  AND ARR_LOCAL_HOUR IS NOT NULL
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

  grp_window = Window.partitionBy('CALL_SIGN_ARR', 'ARR_LOCAL_HOUR')
  magic_percentile = F.expr('percentile_approx(HOURLY_INBOUND_COUNT, 0.5)')

  df_median_inbound_flights_per_airport = df_inbound_flights_per_airport_per_hour.withColumn('MEDIAN_INBOUND_COUNT', magic_percentile.over(grp_window))
  # df_median_inbound_flights_per_airport = df_median_inbound_flights_per_airport.drop('ARR_LOCAL_HOUR').drop('FL_DATE').drop('HOURLY_INBOUND_COUNT').dropDuplicates()
  df_median_inbound_flights_per_airport = df_median_inbound_flights_per_airport.drop('FL_DATE').drop('HOURLY_INBOUND_COUNT').dropDuplicates()


  df_median_inbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_median_inbound_flights_per_airport");
  sqlContext.sql("create table group25.df_median_inbound_flights_per_airport as select * from mytempTable");

  display(df_median_inbound_flights_per_airport.filter(df_median_inbound_flights_per_airport.CALL_SIGN_ARR=='KORD').orderBy("FL_DATE", "ARR_LOCAL_HOUR"))


  ##### max median of each airport
  df_max_median_inbound_flights_per_airport = sqlContext.sql("""
  SELECT
    CALL_SIGN_ARR,
    MAX(MEDIAN_INBOUND_COUNT) AS MAX_MEDIAN_INBOUND
  FROM group25.df_median_inbound_flights_per_airport
  GROUP BY 1
  ORDER BY 1
  
  """)

  df_max_median_inbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_max_median_inbound_flights_per_airport");
  sqlContext.sql("create table group25.df_max_median_inbound_flights_per_airport as select * from mytempTable");

  display(df_max_median_inbound_flights_per_airport)
  display(df_max_median_inbound_flights_per_airport.where("CALL_SIGN_ARR==\'KORD\'").orderBy("CALL_SIGN_ARR"))
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
                CALL_SIGN_DEP,
                FL_DATE,
                DEP_LOCAL_HOUR,
                COUNT(*) AS OUTBOUND_COUNT
              FROM group25.airlines_utc_main
              WHERE CALL_SIGN_DEP IS NOT NULL
              AND DEP_LOCAL_HOUR IS NOT NULL
              GROUP BY 1, 2, 3
              ORDER BY 1, 2, 3 ASC
              
              """)
  # make column FL_TIME and Cast it into timestamp
  df1 = df1.withColumn("FL_TIME", unix_timestamp(col("FL_DATE").cast("timestamp")) + df1.DEP_LOCAL_HOUR*60*60)
  df1 = df1.withColumn("FL_TIME", col("FL_TIME").cast("timestamp"))
  

  display(df1.filter(df1.CALL_SIGN_DEP=='KORD').orderBy("FL_DATE", "DEP_LOCAL_HOUR"))

  
  # calculate the rolling window
  # handle the 0th column
  df1 = df1.withColumn('OUTBOUND_COUNT_0H', df1.OUTBOUND_COUNT.cast('double'))
  
  # handle columns 1-6
  total_lag_time = 6
  for i in range(1, total_lag_time+1):  
    window = (Window.partitionBy("CALL_SIGN_DEP", "FL_DATE").orderBy(col("FL_TIME").cast('long')).rangeBetween(-hours(int(i)), 0))
    df1 = df1.withColumn('OUTBOUND_COUNT_' +str(i) + 'H', sum("OUTBOUND_COUNT_0H").over(window))
  
  for i in range(total_lag_time, 0, -1):
    df1 = df1.withColumn('OUTBOUND_COUNT_' +str(i) + 'H', col('OUTBOUND_COUNT_' +str(i) + 'H') - col('OUTBOUND_COUNT_' +str(i-1) + 'H'))
  
  # display(df1.filter(df1.CALL_SIGN_DEP=='KORD').orderBy("FL_DATE", "DEP_LOCAL_HOUR"))

  # save table
  df1.createOrReplaceTempView("mytempTable")
  airlines_outbound_flights_main_version_time_series_rolling_counts = df1
  sqlContext.sql("DROP TABLE IF EXISTS group25.airlines_outbound_flights_main_version_time_series_rolling_counts");
  sqlContext.sql("create table group25.airlines_outbound_flights_main_version_time_series_rolling_counts as select * from mytempTable");

  # display
  display(
    airlines_outbound_flights_main_version_time_series_rolling_counts
        .where("CALL_SIGN_DEP==\'KORD\'")
        .orderBy("FL_DATE", "DEP_LOCAL_HOUR"))



  ##### normalized outbound rolling window by max_median

    
  df1 = airlines_outbound_flights_main_version_time_series_rolling_counts

  df2 = df_max_median_outbound_flights_per_airport

  df_normalized_outbound_flights_per_airport=df1.alias("a").join(
      df2.alias("b"), df1['CALL_SIGN_DEP'] == df2['CALL_SIGN_DEP'], how="left"
  ).select( 'a.CALL_SIGN_DEP',
  'FL_DATE',
  'DEP_LOCAL_HOUR',
  'OUTBOUND_COUNT',
  'FL_TIME',
  'OUTBOUND_COUNT_0H',
  'OUTBOUND_COUNT_1H',
  'OUTBOUND_COUNT_2H',
  'OUTBOUND_COUNT_3H',
  'OUTBOUND_COUNT_4H',
  'OUTBOUND_COUNT_5H',
  'OUTBOUND_COUNT_6H',
  'MAX_MEDIAN_OUTBOUND'
          )

  for i in range(0, total_lag_time+1):
    df_normalized_outbound_flights_per_airport = df_normalized_outbound_flights_per_airport.withColumn(
                                                    'NORMALIZED_OUTBOUND_COUNT_' +str(i) + 'H', 
                                                    col('OUTBOUND_COUNT_' +str(i) + 'H').cast('int')/col('MAX_MEDIAN_OUTBOUND')
                                                  )

  # display(df_normalized_outbound_flights_per_airport)
  display(df_normalized_outbound_flights_per_airport.filter(df_normalized_outbound_flights_per_airport.CALL_SIGN_DEP=='KORD').orderBy("FL_DATE", "DEP_LOCAL_HOUR"))
  
  df_normalized_outbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_normalized_outbound_flights_per_airport");
  sqlContext.sql("create table group25.df_normalized_outbound_flights_per_airport as select * from mytempTable");


  # # showing the median outbound flight of airport KORD
  # display(df_median_outbound_flights_per_airport.where("CALL_SIGN_DEP==\'KORD\'").orderBy("FL_DATE", "DEP_LOCAL_HOUR"))

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
                CALL_SIGN_ARR,
                FL_DATE,
                ARR_LOCAL_HOUR,
                COUNT(*) AS INBOUND_COUNT
              FROM group25.airlines_utc_main
              WHERE CALL_SIGN_ARR IS NOT NULL
              AND ARR_LOCAL_HOUR IS NOT NULL
              GROUP BY 1, 2, 3
              ORDER BY 1, 2, 3 ASC
              
              """)
  # make column FL_TIME and ast it into timestamp
  df1 = df1.withColumn("FL_TIME", unix_timestamp(col("FL_DATE").cast("timestamp")) + df1.ARR_LOCAL_HOUR*60*60)
  df1 = df1.withColumn("FL_TIME", col("FL_TIME").cast("timestamp"))

  display(df1.filter(df1.CALL_SIGN_ARR=='KORD').orderBy("FL_DATE", "ARR_LOCAL_HOUR"))

  
  # calculate the rolling window
  # handle the 0th column
  df1 = df1.withColumn('INBOUND_COUNT_0H', df1.INBOUND_COUNT.cast('double'))
  
  # handle columns 1-6
  total_lag_time = 6
  for i in range(1, total_lag_time+1):  
    window = (Window.partitionBy("CALL_SIGN_ARR", "FL_DATE").orderBy(col("FL_TIME").cast('long')).rangeBetween(-hours(int(i)), 0))
    df1 = df1.withColumn('INBOUND_COUNT_' +str(i) + 'H', sum("INBOUND_COUNT_0H").over(window))
  
  for i in range(total_lag_time, 0, -1):
    df1 = df1.withColumn('INBOUND_COUNT_' +str(i) + 'H', col('INBOUND_COUNT_' +str(i) + 'H') - col('INBOUND_COUNT_' +str(i-1) + 'H'))
  
  # display(df1.filter(df1.CALL_SIGN_ARR=='KORD').orderBy("FL_DATE", "ARR_LOCAL_HOUR"))

  # save table
  df1.createOrReplaceTempView("mytempTable")
  airlines_inbound_flights_main_version_time_series_rolling_counts = df1
  sqlContext.sql("DROP TABLE IF EXISTS group25.airlines_inbound_flights_main_version_time_series_rolling_counts");
  sqlContext.sql("create table group25.airlines_inbound_flights_main_version_time_series_rolling_counts as select * from mytempTable");

  # display
  display(
    airlines_inbound_flights_main_version_time_series_rolling_counts
        .where("CALL_SIGN_ARR==\'KORD\'")
        .orderBy("FL_DATE", "ARR_LOCAL_HOUR"))



  ##### normalized inbound rolling window by max_median

  # Set df2 as rolling windown
  # set df3 as max median window
  df2 = airlines_inbound_flights_main_version_time_series_rolling_counts
  df3 = df_max_median_inbound_flights_per_airport

  # create the normalized window
  df_normalized_inbound_flights_per_airport=df2.alias("a").join(
      df3.alias("b"), df2['CALL_SIGN_ARR'] == df3['CALL_SIGN_ARR'], how="left"
  ).select( 'a.CALL_SIGN_ARR',
  'FL_DATE',
  'ARR_LOCAL_HOUR',
  'INBOUND_COUNT',
  'FL_TIME',
  'INBOUND_COUNT_0H',
  'INBOUND_COUNT_1H',
  'INBOUND_COUNT_2H',
  'INBOUND_COUNT_3H',
  'INBOUND_COUNT_4H',
  'INBOUND_COUNT_5H',
  'INBOUND_COUNT_6H',
  'MAX_MEDIAN_INBOUND'
          )

  for i in range(0, total_lag_time+1):
    df_normalized_inbound_flights_per_airport = df_normalized_inbound_flights_per_airport.withColumn(
                                                    'NORMALIZED_INBOUND_COUNT_' +str(i) + 'H', 
                                                    col('INBOUND_COUNT_' +str(i) + 'H').cast('int')/col('MAX_MEDIAN_INBOUND')
                                                  )

  # display(df_normalized_inbound_flights_per_airport)
  display(df_normalized_inbound_flights_per_airport.filter(df_normalized_inbound_flights_per_airport.CALL_SIGN_ARR=='KORD').orderBy("FL_DATE", "ARR_LOCAL_HOUR"))
  
  df_normalized_inbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_normalized_inbound_flights_per_airport");
  sqlContext.sql("create table group25.df_normalized_inbound_flights_per_airport as select * from mytempTable");

  # # showing the median inbound flight of airport KORD
  # display(df_median_inbound_flights_per_airport.where("CALL_SIGN_ARR==\'KORD\'").orderBy("FL_DATE", "ARR_LOCAL_HOUR"))
  
  ##### print counts
  row_counts = [
    ('inbound counts ', df1.count()),
    ('max median inbound df ', df_max_median_inbound_flights_per_airport.count()),
    ('airlines_inbound_flights_main_version_time_series_rolling_counts ', airlines_inbound_flights_main_version_time_series_rolling_counts.count()),
    ('df_normalized_inbound_flights_per_airport ', df_normalized_inbound_flights_per_airport.count())
    ('airlines_utc_main '. df_airlins_utc_main.count())
  ]
  
  print('inbound counts ', df1.count())
  print('max median inbound df ', df_max_median_inbound_flights_per_airport.count())
  print('airlines_inbound_flights_main_version_time_series_rolling_counts ', airlines_inbound_flights_main_version_time_series_rolling_counts.count())
  print('df_normalized_inbound_flights_per_airport ', df_normalized_inbound_flights_per_airport.count())
  print('airlines_utc_main '. df_airlins_utc_main.count())

  return df_normalized_inbound_flights_per_airport, row_counts

df_normalized_inbound_flights_per_airport, normalized_inbound_row_counts = compute_normalized_inbound_flights(df_max_median_inbound_flights_per_airport)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) FROM (
# MAGIC   SELECT
# MAGIC     CALL_SIGN_ARR,
# MAGIC     FL_DATE,
# MAGIC     ARR_LOCAL_HOUR,
# MAGIC     COUNT(*) AS INBOUND_COUNT
# MAGIC   FROM group25.airlines_utc_main
# MAGIC   WHERE CALL_SIGN_ARR IS NOT NULL
# MAGIC   AND ARR_LOCAL_HOUR IS NOT NULL
# MAGIC   GROUP BY 1, 2, 3
# MAGIC   ORDER BY 1, 2, 3 ASC
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select
# MAGIC CALL_SIGN_ARR,
# MAGIC FL_DATE,
# MAGIC ARR_LOCAL_HOUR,
# MAGIC hour(dep_local_timestamp + INTERVAL 2 HOUR) as ARR_LOCAL_HOUR_2H,
# MAGIC sum(dep_del15) as DELAY_INBOUND_COUNT_0H
# MAGIC from group25.airlines_utc_main
# MAGIC where dep_del15 is not null
# MAGIC group by 1,2,3,4

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT count(*) FROM group25.airlines_utc_main

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
                CALL_SIGN_DEP,
                FL_DATE,
                DEP_LOCAL_HOUR,
                COUNT(*) AS DIVERTED_OUTBOUND_COUNT
              FROM group25.airlines_utc_main
              WHERE CALL_SIGN_DEP IS NOT NULL
              AND DEP_LOCAL_HOUR IS NOT NULL
              AND diverted=1
              GROUP BY 1, 2, 3
              ORDER BY 1, 2, 3 ASC
              
              """)
  # make column FL_TIME and ast it into timestamp
  df1 = df1.withColumn("FL_TIME", unix_timestamp(col("FL_DATE").cast("timestamp")) + df1.DEP_LOCAL_HOUR*60*60)
  df1 = df1.withColumn("FL_TIME", col("FL_TIME").cast("timestamp"))
  

  display(df1.filter(df1.CALL_SIGN_DEP=='KORD').orderBy("FL_DATE", "DEP_LOCAL_HOUR"))

  
  # calculate the rolling window
  # handle the 0th column
  df1 = df1.withColumn('DIVERTED_OUTBOUND_COUNT_0H', df1.DIVERTED_OUTBOUND_COUNT.cast('double'))
  
  # handle columns 1-6
  total_lag_time = 6
  for i in range(1, total_lag_time+1):  
    window = (Window.partitionBy("CALL_SIGN_DEP", "FL_DATE").orderBy(col("FL_TIME").cast('long')).rangeBetween(-hours(int(i)), 0))
    df1 = df1.withColumn('DIVERTED_OUTBOUND_COUNT_' +str(i) + 'H', sum("DIVERTED_OUTBOUND_COUNT_0H").over(window))
  
  for i in range(total_lag_time, 0, -1):
    df1 = df1.withColumn('DIVERTED_OUTBOUND_COUNT_' +str(i) + 'H', col('DIVERTED_OUTBOUND_COUNT_' +str(i) + 'H') - col('DIVERTED_OUTBOUND_COUNT_' +str(i-1) + 'H'))
  
  # display(df1.filter(df1.CALL_SIGN_DEP=='KORD').orderBy("FL_DATE", "DEP_LOCAL_HOUR"))

  # save table
  df1.createOrReplaceTempView("mytempTable")
  airlines_diverted_outbound_flights_main_version_time_series_rolling_counts = df1
  sqlContext.sql("DROP TABLE IF EXISTS group25.airlines_diverted_outbound_flights_main_version_time_series_rolling_counts");
  sqlContext.sql("create table group25.airlines_diverted_outbound_flights_main_version_time_series_rolling_counts as select * from mytempTable");

  # display
  display(
    airlines_diverted_outbound_flights_main_version_time_series_rolling_counts
        .where("CALL_SIGN_DEP==\'KORD\'")
        .orderBy("FL_DATE", "DEP_LOCAL_HOUR"))



  ##### normalized diverted_outbound rolling window by max_median

    
  df1 = airlines_diverted_outbound_flights_main_version_time_series_rolling_counts

  df2 = df_max_median_outbound_flights_per_airport

  df_normalized_diverted_outbound_flights_per_airport=df1.alias("a").join(
      df2.alias("b"), df1['CALL_SIGN_DEP'] == df2['CALL_SIGN_DEP'], how="left"
  ).select( 'a.CALL_SIGN_DEP',
  'FL_DATE',
  'DEP_LOCAL_HOUR',
  'DIVERTED_OUTBOUND_COUNT',
  'FL_TIME',
  'DIVERTED_OUTBOUND_COUNT_0H',
  'DIVERTED_OUTBOUND_COUNT_1H',
  'DIVERTED_OUTBOUND_COUNT_2H',
  'DIVERTED_OUTBOUND_COUNT_3H',
  'DIVERTED_OUTBOUND_COUNT_4H',
  'DIVERTED_OUTBOUND_COUNT_5H',
  'DIVERTED_OUTBOUND_COUNT_6H',
  'MAX_MEDIAN_OUTBOUND'
          )

  for i in range(0, total_lag_time+1):
    df_normalized_diverted_outbound_flights_per_airport = df_normalized_diverted_outbound_flights_per_airport.withColumn(
                                                    'NORMALIZED_DIVERTED_OUTBOUND_COUNT_' +str(i) + 'H', 
                                                    col('DIVERTED_OUTBOUND_COUNT_' +str(i) + 'H').cast('int')/col('MAX_MEDIAN_OUTBOUND')
                                                  )

  # display(df_normalized_diverted_outbound_flights_per_airport)
  display(df_normalized_diverted_outbound_flights_per_airport.filter(df_normalized_diverted_outbound_flights_per_airport.CALL_SIGN_DEP=='KORD').orderBy("FL_DATE", "DEP_LOCAL_HOUR"))
  
  df_normalized_diverted_outbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_normalized_diverted_outbound_flights_per_airport");
  sqlContext.sql("create table group25.df_normalized_diverted_outbound_flights_per_airport as select * from mytempTable");

  # # showing the median diverted_outbound flight of airport KORD
  # display(df_median_diverted_outbound_flights_per_airport.where("CALL_SIGN_DEP==\'KORD\'").orderBy("FL_DATE", "DEP_LOCAL_HOUR"))

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
                CALL_SIGN_ARR,
                FL_DATE,
                ARR_LOCAL_HOUR,
                COUNT(*) AS DIVERTED_INBOUND_COUNT
              FROM group25.airlines_utc_main
              WHERE CALL_SIGN_ARR IS NOT NULL
              AND ARR_LOCAL_HOUR IS NOT NULL
              AND diverted=1
              GROUP BY 1, 2, 3
              ORDER BY 1, 2, 3 ASC
              
              """)
  # make column FL_TIME and ast it into timestamp
  df1 = df1.withColumn("FL_TIME", unix_timestamp(col("FL_DATE").cast("timestamp")) + df1.ARR_LOCAL_HOUR*60*60)
  df1 = df1.withColumn("FL_TIME", col("FL_TIME").cast("timestamp"))
  

  display(df1.filter(df1.CALL_SIGN_ARR=='KORD').orderBy("FL_DATE", "ARR_LOCAL_HOUR"))

  
  # calculate the rolling window
  # handle the 0th column
  df1 = df1.withColumn('DIVERTED_INBOUND_COUNT_0H', df1.DIVERTED_INBOUND_COUNT.cast('double'))
  
  # handle columns 1-6
  total_lag_time = 6
  for i in range(1, total_lag_time+1):  
    window = (Window.partitionBy("CALL_SIGN_ARR", "FL_DATE").orderBy(col("FL_TIME").cast('long')).rangeBetween(-hours(int(i)), 0))
    df1 = df1.withColumn('DIVERTED_INBOUND_COUNT_' +str(i) + 'H', sum("DIVERTED_INBOUND_COUNT_0H").over(window))
  
  for i in range(total_lag_time, 0, -1):
    df1 = df1.withColumn('DIVERTED_INBOUND_COUNT_' +str(i) + 'H', col('DIVERTED_INBOUND_COUNT_' +str(i) + 'H') - col('DIVERTED_INBOUND_COUNT_' +str(i-1) + 'H'))
  
  # display(df1.filter(df1.CALL_SIGN_ARR=='KORD').orderBy("FL_DATE", "ARR_LOCAL_HOUR"))

  # save table
  df1.createOrReplaceTempView("mytempTable")
  airlines_diverted_inbound_flights_main_version_time_series_rolling_counts = df1
  sqlContext.sql("DROP TABLE IF EXISTS group25.airlines_diverted_inbound_flights_main_version_time_series_rolling_counts");
  sqlContext.sql("create table group25.airlines_diverted_inbound_flights_main_version_time_series_rolling_counts as select * from mytempTable");

  # display
  display(
    airlines_diverted_inbound_flights_main_version_time_series_rolling_counts
        .where("CALL_SIGN_ARR==\'KORD\'")
        .orderBy("FL_DATE", "ARR_LOCAL_HOUR"))



  ##### normalized diverted_inbound rolling window by max_median

    
  df1 = airlines_diverted_inbound_flights_main_version_time_series_rolling_counts

  df2 = df_max_median_inbound_flights_per_airport

  df_normalized_diverted_inbound_flights_per_airport=df1.alias("a").join(
      df2.alias("b"), df1['CALL_SIGN_ARR'] == df2['CALL_SIGN_ARR'], how="left"
  ).select( 'a.CALL_SIGN_ARR',
  'FL_DATE',
  'ARR_LOCAL_HOUR',
  'DIVERTED_INBOUND_COUNT',
  'FL_TIME',
  'DIVERTED_INBOUND_COUNT_0H',
  'DIVERTED_INBOUND_COUNT_1H',
  'DIVERTED_INBOUND_COUNT_2H',
  'DIVERTED_INBOUND_COUNT_3H',
  'DIVERTED_INBOUND_COUNT_4H',
  'DIVERTED_INBOUND_COUNT_5H',
  'DIVERTED_INBOUND_COUNT_6H',
  'MAX_MEDIAN_INBOUND'
          )

  for i in range(0, total_lag_time+1):
    df_normalized_diverted_inbound_flights_per_airport = df_normalized_diverted_inbound_flights_per_airport.withColumn(
                                                    'NORMALIZED_DIVERTED_INBOUND_COUNT_' +str(i) + 'H', 
                                                    col('DIVERTED_INBOUND_COUNT_' +str(i) + 'H').cast('int')/col('MAX_MEDIAN_INBOUND')
                                                  )

  # display(df_normalized_diverted_inbound_flights_per_airport)
  display(df_normalized_diverted_inbound_flights_per_airport.filter(df_normalized_diverted_inbound_flights_per_airport.CALL_SIGN_ARR=='KORD').orderBy("FL_DATE", "ARR_LOCAL_HOUR"))
  
  df_normalized_diverted_inbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_normalized_diverted_inbound_flights_per_airport");
  sqlContext.sql("create table group25.df_normalized_diverted_inbound_flights_per_airport as select * from mytempTable");

  # # showing the median diverted_inbound flight of airport KORD
  # display(df_median_inbound_flights_per_airport.where("CALL_SIGN_ARR==\'KORD\'").orderBy("FL_DATE", "ARR_LOCAL_HOUR"))

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
            CALL_SIGN_DEP,
            FL_DATE,
            DEP_LOCAL_HOUR,
            sum(dep_del15) as DELAY_OUTBOUND_COUNT_0H
            from group25.airlines_utc_main
            where dep_del15 is not null
            group by 1,2,3
            
              """)
  df_delay_outbound_flights_per_airport_per_hour = df_delay_outbound_flights_per_airport_per_hour.withColumn("FL_TIME", unix_timestamp(col("FL_DATE").cast("timestamp")) + df_delay_outbound_flights_per_airport_per_hour.DEP_LOCAL_HOUR*60*60)
  df_delay_outbound_flights_per_airport_per_hour = df_delay_outbound_flights_per_airport_per_hour.withColumn("FL_TIME", col("FL_TIME").cast("timestamp"))
  display(df_delay_outbound_flights_per_airport_per_hour.filter(df_delay_outbound_flights_per_airport_per_hour.CALL_SIGN_DEP=='KORD').orderBy("FL_DATE", "DEP_LOCAL_HOUR"))


  # building the delays columns
  total_lag_time = 6
  for i in range(1, total_lag_time+1):  
    window = (Window.partitionBy("FL_DATE", "CALL_SIGN_DEP").orderBy(col("FL_TIME").cast('long')).rangeBetween(-hours(int(i)), 0))
    df_delay_outbound_flights_per_airport_per_hour = df_delay_outbound_flights_per_airport_per_hour.withColumn('DELAY_OUTBOUND_COUNT_' +str(i) + 'H', sum("DELAY_OUTBOUND_COUNT_0H").over(window))

  for i in range(total_lag_time, 0, -1):
    df_delay_outbound_flights_per_airport_per_hour = df_delay_outbound_flights_per_airport_per_hour.withColumn('DELAY_OUTBOUND_COUNT_' +str(i) + 'H', col('DELAY_OUTBOUND_COUNT_' +str(i) + 'H') - col('DELAY_OUTBOUND_COUNT_' +str(i-1) + 'H'))

  display(df_delay_outbound_flights_per_airport_per_hour.filter(df_delay_outbound_flights_per_airport_per_hour.CALL_SIGN_DEP=='KORD').orderBy("FL_DATE", "DEP_LOCAL_HOUR"))

  df_delay_outbound_flights_per_airport_per_hour.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_delayed_outbound_flights_per_airport_per_hour_rolling_window");
  sqlContext.sql("create table group25.df_delayed_outbound_flights_per_airport_per_hour_rolling_window as select * from mytempTable");


  ##### normalized outbound rolling window by max_median

  df2 = df_max_median_outbound_flights_per_airport

  df_normalized_delay_outbound_flights_per_airport=df_delay_outbound_flights_per_airport_per_hour.alias("a").join(
      df2.alias("b"), df_delay_outbound_flights_per_airport_per_hour['CALL_SIGN_DEP'] == df2['CALL_SIGN_DEP'], how="left"
  ).select('a.CALL_SIGN_DEP',
  'FL_DATE',
  'DEP_LOCAL_HOUR',
  'FL_TIME',
  'DELAY_OUTBOUND_COUNT_0H',
  'DELAY_OUTBOUND_COUNT_1H',
  'DELAY_OUTBOUND_COUNT_2H',
  'DELAY_OUTBOUND_COUNT_3H',
  'DELAY_OUTBOUND_COUNT_4H',
  'DELAY_OUTBOUND_COUNT_5H',
  'DELAY_OUTBOUND_COUNT_6H',
  'MAX_MEDIAN_OUTBOUND'
          )

  for i in range(0, total_lag_time+1):
    df_normalized_delay_outbound_flights_per_airport = df_normalized_delay_outbound_flights_per_airport.withColumn('NORMALIZED_DELAY_OUTBOUND_COUNT_' +str(i) + 'H', col('DELAY_OUTBOUND_COUNT_' +str(i) + 'H').cast('int')/col('MAX_MEDIAN_OUTBOUND'))

  # display(df_normalized_delay_outbound_flights_per_airport)
  display(df_normalized_delay_outbound_flights_per_airport.filter(df_normalized_delay_outbound_flights_per_airport.CALL_SIGN_DEP=='KORD').orderBy("FL_DATE", "DEP_LOCAL_HOUR"))
  
  df_normalized_delay_outbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_normalized_delay_outbound_flights_per_airport");
  sqlContext.sql("create table group25.df_normalized_delay_outbound_flights_per_airport as select * from mytempTable");


#   # showing the median outbound flight of airport KORD
#   display(df_median_outbound_flights_per_airport.where("CALL_SIGN_DEP==\'KORD\'").orderBy("FL_DATE", "DEP_LOCAL_HOUR"))
  
  return df_normalized_delay_outbound_flights_per_airport

df_normalized_delay_outbound_flights_per_airport = compute_normalized_delay_outbound_flights(df_max_median_outbound_flights_per_airport)


# COMMAND ----------

# MAGIC %md ##normalized delay inbound flights

# COMMAND ----------

############################################################################################
# dalay inbound flights
## calculate the number of delays per airpor per hour per airline - departure/inbound

def compute_normalized_delay_inbound_flights(df_max_median_inbound_flights_per_airport):
  #function to calculate number of seconds from number of days
  hours = lambda i: i * 60 * 60
  df_delay_inbound_flights_per_airport_per_hour = sqlContext.sql("""
          select
            CALL_SIGN_ARR,
            FL_DATE,
            ARR_LOCAL_HOUR,
            sum(dep_del15) as DELAY_INBOUND_COUNT_0H
            from group25.airlines_utc_main
            where dep_del15 is not null
            group by 1,2,3
            
              """)
  df_delay_inbound_flights_per_airport_per_hour = df_delay_inbound_flights_per_airport_per_hour.withColumn("FL_TIME", unix_timestamp(col("FL_DATE").cast("timestamp")) + df_delay_inbound_flights_per_airport_per_hour.ARR_LOCAL_HOUR*60*60)
  df_delay_inbound_flights_per_airport_per_hour = df_delay_inbound_flights_per_airport_per_hour.withColumn("FL_TIME", col("FL_TIME").cast("timestamp"))
  display(df_delay_inbound_flights_per_airport_per_hour.filter(df_delay_inbound_flights_per_airport_per_hour.CALL_SIGN_ARR=='KORD').orderBy("FL_DATE", "ARR_LOCAL_HOUR"))


  # building the delays columns
  total_lag_time = 6
  for i in range(1, total_lag_time+1):  
    window = (Window.partitionBy("FL_DATE", "CALL_SIGN_ARR").orderBy(col("FL_TIME").cast('long')).rangeBetween(-hours(int(i)), 0))
    df_delay_inbound_flights_per_airport_per_hour = df_delay_inbound_flights_per_airport_per_hour.withColumn('DELAY_INBOUND_COUNT_' +str(i) + 'H', sum("DELAY_INBOUND_COUNT_0H").over(window))

  for i in range(total_lag_time, 0, -1):
    df_delay_inbound_flights_per_airport_per_hour = df_delay_inbound_flights_per_airport_per_hour.withColumn('DELAY_INBOUND_COUNT_' +str(i) + 'H', col('DELAY_INBOUND_COUNT_' +str(i) + 'H') - col('DELAY_INBOUND_COUNT_' +str(i-1) + 'H'))

  display(df_delay_inbound_flights_per_airport_per_hour.filter(df_delay_inbound_flights_per_airport_per_hour.CALL_SIGN_ARR=='KORD').orderBy("FL_DATE", "ARR_LOCAL_HOUR"))

  df_delay_inbound_flights_per_airport_per_hour.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_delayed_inbound_flights_per_airport_per_hour_rolling_window");
  sqlContext.sql("create table group25.df_delayed_inbound_flights_per_airport_per_hour_rolling_window as select * from mytempTable");


  ##### normalized inbound rolling window by max_median

  df2 = df_max_median_inbound_flights_per_airport

  df_normalized_delay_inbound_flights_per_airport=df_delay_inbound_flights_per_airport_per_hour.alias("a").join(
      df2.alias("b"), df_delay_inbound_flights_per_airport_per_hour['CALL_SIGN_ARR'] == df2['CALL_SIGN_ARR'], how="left"
  ).select( 'a.CALL_SIGN_ARR',
  'FL_DATE',
  'ARR_LOCAL_HOUR',
  'FL_TIME',
  'DELAY_INBOUND_COUNT_0H',
  'DELAY_INBOUND_COUNT_1H',
  'DELAY_INBOUND_COUNT_2H',
  'DELAY_INBOUND_COUNT_3H',
  'DELAY_INBOUND_COUNT_4H',
  'DELAY_INBOUND_COUNT_5H',
  'DELAY_INBOUND_COUNT_6H',
  'MAX_MEDIAN_INBOUND'
          )

  for i in range(0, total_lag_time+1):
    df_normalized_delay_inbound_flights_per_airport = df_normalized_delay_inbound_flights_per_airport.withColumn('NORMALIZED_DELAY_INBOUND_COUNT_' +str(i) + 'H', col('DELAY_INBOUND_COUNT_' +str(i) + 'H').cast('int')/col('MAX_MEDIAN_INBOUND'))

  # display(df_normalized_delay_inbound_flights_per_airport)
  display(df_normalized_delay_inbound_flights_per_airport.filter(df_normalized_delay_inbound_flights_per_airport.CALL_SIGN_ARR=='KORD').orderBy("FL_DATE", "ARR_LOCAL_HOUR"))
  
  df_normalized_delay_inbound_flights_per_airport.createOrReplaceTempView("mytempTable") 
  sqlContext.sql("DROP TABLE IF EXISTS group25.df_normalized_delay_inbound_flights_per_airport");
  sqlContext.sql("create table group25.df_normalized_delay_inbound_flights_per_airport as select * from mytempTable");


#   # showing the median inbound flight of airport KORD
#   display(df_median_inbound_flights_per_airport.where("CALL_SIGN_ARR==\'KORD\'").orderBy("FL_DATE", "ARR_LOCAL_HOUR"))
  
  return df_normalized_delay_inbound_flights_per_airport

df_normalized_delay_inbound_flights_per_airport = compute_normalized_delay_inbound_flights(df_max_median_inbound_flights_per_airport)


# COMMAND ----------



# COMMAND ----------

# MAGIC %md # Join all table

# COMMAND ----------

#v5
# df_airlines_utc_main, 
# df_normalized_outbound, 
# df_normalized_delay_outbound

def joining_tables():
  baseline_model_data = sqlContext.sql("""
  SELECT
    T1.*,
    T2.MAX_MEDIAN_OUTBOUND,
    T3.MAX_MEDIAN_INBOUND,

    T2.OUTBOUND_COUNT,
    T2.OUTBOUND_COUNT_0H,
    T2.OUTBOUND_COUNT_1H,
    T2.OUTBOUND_COUNT_2H,
    T2.OUTBOUND_COUNT_3H,
    T2.OUTBOUND_COUNT_4H,
    T2.OUTBOUND_COUNT_5H,
    T2.OUTBOUND_COUNT_6H,
    T2.NORMALIZED_OUTBOUND_COUNT_0H,
    T2.NORMALIZED_OUTBOUND_COUNT_1H,
    T2.NORMALIZED_OUTBOUND_COUNT_2H,
    T2.NORMALIZED_OUTBOUND_COUNT_3H,
    T2.NORMALIZED_OUTBOUND_COUNT_4H,
    T2.NORMALIZED_OUTBOUND_COUNT_5H,
    T2.NORMALIZED_OUTBOUND_COUNT_6H,

    T3.INBOUND_COUNT,
    T3.INBOUND_COUNT_0H,
    T3.INBOUND_COUNT_1H,
    T3.INBOUND_COUNT_2H,
    T3.INBOUND_COUNT_3H,
    T3.INBOUND_COUNT_4H,
    T3.INBOUND_COUNT_5H,
    T3.INBOUND_COUNT_6H,
    T3.NORMALIZED_INBOUND_COUNT_0H,
    T3.NORMALIZED_INBOUND_COUNT_1H,
    T3.NORMALIZED_INBOUND_COUNT_2H,
    T3.NORMALIZED_INBOUND_COUNT_3H,
    T3.NORMALIZED_INBOUND_COUNT_4H,
    T3.NORMALIZED_INBOUND_COUNT_5H,
    T3.NORMALIZED_INBOUND_COUNT_6H,

    T4.DIVERTED_OUTBOUND_COUNT,
    T4.DIVERTED_OUTBOUND_COUNT_0H,
    T4.DIVERTED_OUTBOUND_COUNT_1H,
    T4.DIVERTED_OUTBOUND_COUNT_2H,
    T4.DIVERTED_OUTBOUND_COUNT_3H,
    T4.DIVERTED_OUTBOUND_COUNT_4H,
    T4.DIVERTED_OUTBOUND_COUNT_5H,
    T4.DIVERTED_OUTBOUND_COUNT_6H,
    T4.NORMALIZED_DIVERTED_OUTBOUND_COUNT_0H,
    T4.NORMALIZED_DIVERTED_OUTBOUND_COUNT_1H,
    T4.NORMALIZED_DIVERTED_OUTBOUND_COUNT_2H,
    T4.NORMALIZED_DIVERTED_OUTBOUND_COUNT_3H,
    T4.NORMALIZED_DIVERTED_OUTBOUND_COUNT_4H,
    T4.NORMALIZED_DIVERTED_OUTBOUND_COUNT_5H,
    T4.NORMALIZED_DIVERTED_OUTBOUND_COUNT_6H,

    T5.DIVERTED_INBOUND_COUNT,
    T5.DIVERTED_INBOUND_COUNT_0H,
    T5.DIVERTED_INBOUND_COUNT_1H,
    T5.DIVERTED_INBOUND_COUNT_2H,
    T5.DIVERTED_INBOUND_COUNT_3H,
    T5.DIVERTED_INBOUND_COUNT_4H,
    T5.DIVERTED_INBOUND_COUNT_5H,
    T5.DIVERTED_INBOUND_COUNT_6H,
    T5.NORMALIZED_DIVERTED_INBOUND_COUNT_0H,
    T5.NORMALIZED_DIVERTED_INBOUND_COUNT_1H,
    T5.NORMALIZED_DIVERTED_INBOUND_COUNT_2H,
    T5.NORMALIZED_DIVERTED_INBOUND_COUNT_3H,
    T5.NORMALIZED_DIVERTED_INBOUND_COUNT_4H,
    T5.NORMALIZED_DIVERTED_INBOUND_COUNT_5H,
    T5.NORMALIZED_DIVERTED_INBOUND_COUNT_6H,

    T6.DELAY_OUTBOUND_COUNT_0H,
    T6.DELAY_OUTBOUND_COUNT_1H,
    T6.DELAY_OUTBOUND_COUNT_2H,
    T6.DELAY_OUTBOUND_COUNT_3H,
    T6.DELAY_OUTBOUND_COUNT_4H,
    T6.DELAY_OUTBOUND_COUNT_5H,
    T6.DELAY_OUTBOUND_COUNT_6H,
    T6.NORMALIZED_DELAY_OUTBOUND_COUNT_0H,
    T6.NORMALIZED_DELAY_OUTBOUND_COUNT_1H,
    T6.NORMALIZED_DELAY_OUTBOUND_COUNT_2H,
    T6.NORMALIZED_DELAY_OUTBOUND_COUNT_3H,
    T6.NORMALIZED_DELAY_OUTBOUND_COUNT_4H,
    T6.NORMALIZED_DELAY_OUTBOUND_COUNT_5H,
    T6.NORMALIZED_DELAY_OUTBOUND_COUNT_6H,

    T7.DELAY_INBOUND_COUNT_0H,
    T7.DELAY_INBOUND_COUNT_1H,
    T7.DELAY_INBOUND_COUNT_2H,
    T7.DELAY_INBOUND_COUNT_3H,
    T7.DELAY_INBOUND_COUNT_4H,
    T7.DELAY_INBOUND_COUNT_5H,
    T7.DELAY_INBOUND_COUNT_6H,
    T7.NORMALIZED_DELAY_INBOUND_COUNT_0H,
    T7.NORMALIZED_DELAY_INBOUND_COUNT_1H,
    T7.NORMALIZED_DELAY_INBOUND_COUNT_2H,
    T7.NORMALIZED_DELAY_INBOUND_COUNT_3H,
    T7.NORMALIZED_DELAY_INBOUND_COUNT_4H,
    T7.NORMALIZED_DELAY_INBOUND_COUNT_5H,
    T7.NORMALIZED_DELAY_INBOUND_COUNT_6H

  FROM
    group25.airlines_utc_main T1   
      LEFT JOIN group25.df_normalized_outbound_flights_per_airport T2
        ON T1.CALL_SIGN_DEP = T2.CALL_SIGN_DEP
          AND T1.FL_DATE = T2.FL_DATE
          AND T1.DEP_LOCAL_HOUR = T2.DEP_LOCAL_HOUR
  
      LEFT JOIN group25.df_normalized_inbound_flights_per_airport T3
        ON T1.CALL_SIGN_ARR = T3.CALL_SIGN_ARR
          AND T1.FL_DATE = T3.FL_DATE
          AND T1.ARR_LOCAL_HOUR = T3.ARR_LOCAL_HOUR

      LEFT JOIN group25.df_normalized_diverted_outbound_flights_per_airport T4
        ON T1.CALL_SIGN_DEP = T4.CALL_SIGN_DEP
          AND T1.FL_DATE = T4.FL_DATE
          AND T1.DEP_LOCAL_HOUR = T4.DEP_LOCAL_HOUR

      LEFT JOIN group25.df_normalized_diverted_inbound_flights_per_airport T5
        ON 
          T1.CALL_SIGN_ARR = T5.CALL_SIGN_ARR
          AND T1.FL_DATE = T5.FL_DATE
          AND T1.ARR_LOCAL_HOUR = T5.ARR_LOCAL_HOUR

      LEFT JOIN group25.df_normalized_delay_outbound_flights_per_airport T6
        ON T1.CALL_SIGN_DEP = T6.CALL_SIGN_DEP
          AND T1.FL_DATE = T6.FL_DATE
          AND T1.DEP_LOCAL_HOUR = T6.DEP_LOCAL_HOUR

      LEFT JOIN group25.df_normalized_delay_inbound_flights_per_airport T7
        ON 
          T1.CALL_SIGN_ARR = T7.CALL_SIGN_ARR
          AND T1.FL_DATE = T7.FL_DATE
          AND T1.ARR_LOCAL_HOUR = T7.ARR_LOCAL_HOUR

  WHERE T1.CALL_SIGN_DEP != "99999"
  """)
  
  # save table to db
  display(baseline_model_data)
  baseline_model_data.createOrReplaceTempView("mytempTable")
  sqlContext.sql("DROP TABLE IF EXISTS group25.baseline_model_data");
  sqlContext.sql("create table group25.baseline_model_data as select * from mytempTable");
  
  print('baseline_model_data rows count: ', baseline_model_data.count())
  
  return baseline_model_data

baseline_model_data = joining_tables()

# COMMAND ----------

# MAGIC %md # check rows counts

# COMMAND ----------

# print and compare the rows
def check_row_counts():
  print("airline utc main", df_airlines_utc_main.count()) # 31,746,841
  print('max median: ', df_max_median_outbound_flights_per_airport.count()) #334
  print('baseline_model_data rows count: ', baseline_model_data.count()) #31,746,841
  return

check_row_counts()

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) , "df_normalized_outbound_flights_per_airport" as table from group25.df_normalized_outbound_flights_per_airport
# MAGIC union 
# MAGIC select count(*), "df_normalized_inbound_flights_per_airport" as table from group25.df_normalized_inbound_flights_per_airport
# MAGIC union 
# MAGIC select count(*), "df_normalized_diverted_outbound_flights_per_airport" as table from group25.df_normalized_diverted_outbound_flights_per_airport
# MAGIC union 
# MAGIC select count(*), "df_normalized_diverted_inbound_flights_per_airport" as table from  group25.df_normalized_diverted_inbound_flights_per_airport
# MAGIC union 
# MAGIC select count(*), "df_normalized_delay_outbound_flights_per_airport" as table from group25.df_normalized_delay_outbound_flights_per_airport
# MAGIC union 
# MAGIC select count(*), "df_normalized_delay_inbound_flights_per_airport" as table from group25.df_normalized_delay_inbound_flights_per_airport
# MAGIC union 
# MAGIC select count(*), "df_normalized_delay_inbound_flights_per_airport" as table from group25.df_normalized_delay_inbound_flights_per_airport
# MAGIC union
# MAGIC select count(*), "baseline_model_data" as table from group25.baseline_model_data

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*), "df_normalized_outbound_flights_per_airport" as table from (select distinct call_sign_dep, FL_DATE, DEP_LOCAL_HOUR from group25.df_normalized_outbound_flights_per_airport where call_sign_dep != '99999' and call_sign_dep is not null and dep_local_hour is not null group by call_sign_dep, FL_DATE, DEP_LOCAL_HOUR)
# MAGIC union
# MAGIC select count(*), "df_normalized_inbound_flights_per_airport" as table from (select distinct call_sign_arr, FL_DATE, ARR_LOCAL_HOUR from group25.df_normalized_inbound_flights_per_airport where call_sign_arr != '99999' and call_sign_arr is not null and ARR_LOCAL_HOUR is not null group by call_sign_arr, FL_DATE, ARR_LOCAL_HOUR)
# MAGIC union 
# MAGIC select count(*), "df_normalized_diverted_outbound_flights_per_airport" as table from (select distinct call_sign_dep, FL_DATE, DEP_LOCAL_HOUR from group25.df_normalized_diverted_outbound_flights_per_airport where call_sign_dep != '99999' and call_sign_dep is not null and DEP_LOCAL_HOUR is not null group by call_sign_dep, FL_DATE, DEP_LOCAL_HOUR)
# MAGIC union 
# MAGIC select count(*), "df_normalized_diverted_inbound_flights_per_airport" as table from (select distinct call_sign_arr, FL_DATE, ARR_LOCAL_HOUR from group25.df_normalized_diverted_inbound_flights_per_airport where call_sign_arr != '99999' and call_sign_arr is not null and ARR_LOCAL_HOUR is not null group by call_sign_arr, FL_DATE, ARR_LOCAL_HOUR)
# MAGIC union 
# MAGIC select count(*), "df_normalized_delay_outbound_flights_per_airport" as table from (select distinct call_sign_dep, FL_DATE, DEP_LOCAL_HOUR from group25.df_normalized_delay_outbound_flights_per_airport where call_sign_dep != '99999' and call_sign_dep is not null and dep_local_hour is not null group by call_sign_dep, FL_DATE, DEP_LOCAL_HOUR)
# MAGIC union 
# MAGIC select count(*), "df_normalized_delay_inbound_flights_per_airport" as table from (select distinct call_sign_arr, FL_DATE, ARR_LOCAL_HOUR from group25.df_normalized_delay_inbound_flights_per_airport where call_sign_arr != '99999' and call_sign_arr is not null and ARR_LOCAL_HOUR is not null group by call_sign_arr, FL_DATE, ARR_LOCAL_HOUR)
# MAGIC union 
# MAGIC select count(*),  "airlines_utc_main" as table from (select distinct call_sign_dep, FL_DATE, DEP_LOCAL_HOUR from group25.airlines_utc_main where call_sign_dep != '99999' and call_sign_dep is not null and dep_local_hour is not null  group by call_sign_dep, FL_DATE, DEP_LOCAL_HOUR)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md #check the case for KORD in baseline_model_data

# COMMAND ----------

# check at the if the tables looks okay

select_cols = [
'FL_DATE',
'OP_UNIQUE_CARRIER',
'DEP_LOCAL_HOUR',
'OUTBOUND_COUNT',
'OUTBOUND_COUNT_0H',
'OUTBOUND_COUNT_1H',
'OUTBOUND_COUNT_2H',
'OUTBOUND_COUNT_3H',
'OUTBOUND_COUNT_4H',
'OUTBOUND_COUNT_5H',
'OUTBOUND_COUNT_6H',
'MAX_MEDIAN_OUTBOUND',
'NORMALIZED_OUTBOUND_COUNT_0H',
'NORMALIZED_OUTBOUND_COUNT_1H',
'NORMALIZED_OUTBOUND_COUNT_2H',
'NORMALIZED_OUTBOUND_COUNT_3H',
'NORMALIZED_OUTBOUND_COUNT_4H',
'NORMALIZED_OUTBOUND_COUNT_5H',
'NORMALIZED_OUTBOUND_COUNT_6H'
]
display(baseline_model_data.select(select_cols).filter(baseline_model_data.call_sign_dep=='KORD').filter(baseline_model_data.OP_UNIQUE_CARRIER=='UA').dropDuplicates().orderBy("FL_DATE", "DEP_LOCAL_HOUR", ))

# COMMAND ----------

df_normalized_delay_outbound_flights_per_airport.columns

# COMMAND ----------

# MAGIC %md # sampling df

# COMMAND ----------

def sampling_df():
  list_of_df = [
    df_airlines_utc_main,
    df_normalized_outbound,
    df_normalized_inbound,
    df_normalized_diverted_outbound,
    df_normalized_diverted_inbound,
    df_normalized_delay_outbound,
    df_normalized_delay_inbound,
  ]
  for df in list_of_df:
    df = df.filter(df.CALL_SIGN_DEP=='KORD')
  return

# sampling_df()
