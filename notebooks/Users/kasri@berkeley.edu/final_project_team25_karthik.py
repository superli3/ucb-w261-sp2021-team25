# Databricks notebook source
# MAGIC %md
# MAGIC # w261 Final Project - Clickthrough Rate Prediction

# COMMAND ----------

# MAGIC %md
# MAGIC 25   
# MAGIC Justin Trobec, Jeff Li, Sonya Chen, Karthik Srinivasan
# MAGIC Spring 2021, section 5, Team 25

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table of Contents
# MAGIC 
# MAGIC * __Section 1__ - Question Formulation
# MAGIC * __Section 2__ - Algorithm Explanation
# MAGIC * __Section 3__ - EDA & Challenges
# MAGIC * __Section 4__ - Algorithm Implementation
# MAGIC * __Section 5__ - Course Concepts

# COMMAND ----------

## imports

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/weather_data"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2015.parquet/"))

# COMMAND ----------

airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data/201*.parquet")
display(airlines.sample(False, 0.00001))

# COMMAND ----------

airlines.printSchema()

# COMMAND ----------

f'{airlines.count():,}'

# COMMAND ----------

#### For the first phase of the project you will focus on flights departing from two major US airports (ORD (Chicago O'Hare) and ATL (Atlanta) for the first quarter of 2015 (that is about 160k flights)

## 161507 rows
airline_dataset = airlines.filter("(ORIGIN == 'ORD' or ORIGIN == 'ATL') AND YEAR == 2015 AND QUARTER == 1").distinct().cache()


#display(airline_dataset.sample(False, 0.00001))

# COMMAND ----------

airline_dataset.take(1)

# COMMAND ----------

display(airline_dataset.describe())

# COMMAND ----------

# MAGIC %md # Weather
# MAGIC https://data.nodc.noaa.gov/cgi-bin/iso?id=gov.noaa.ncdc:C00532

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/weather_data"))

# COMMAND ----------

weather = spark.read.option("header", "true")\
                    .parquet(f"dbfs:/mnt/mids-w261/datasets_final_project/weather_data/*.parquet")

f'{weather.count():,}'

# COMMAND ----------

weather.select(f.col('CALL_SIGN')).distinct().collect()

# COMMAND ----------

display(weather.where('DATE =="DATE"'))

# COMMAND ----------

weather.printSchema()

# COMMAND ----------

display(weather.sample(False, 0.00001))

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS WEATHER;
# MAGIC 
# MAGIC CREATE TEMPORARY VIEW WEATHER 
# MAGIC USING parquet OPTIONS (path "dbfs:/mnt/mids-w261/datasets_final_project/weather_data/weather*.parquet");
# MAGIC 
# MAGIC SELECT * FROM WEATHER LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS AIRLINES_2015;
# MAGIC 
# MAGIC CREATE TEMPORARY VIEW AIRLINES_2015
# MAGIC USING PARQUET OPTIONS (path "dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2015.parquet");
# MAGIC 
# MAGIC SELECT * FROM AIRLINES_2015 LIMIT 10;
# MAGIC 
# MAGIC -- spark.sql("CREATE TEMPORARY VIEW WEATHER USING parquet OPTIONS (path \"dbfs:/mnt/mids-w261/datasets_final_project/weather_data/weather*.parquet\")")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE group25;

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS group25.PHASE_1_PROCESSED_WEATHER;
# MAGIC CREATE TABLE group25.PHASE_1_PROCESSED_WEATHER AS (
# MAGIC   SELECT * FROM WEATHER
# MAGIC   WHERE CALL_SIGN LIKE ('%ORD%') OR CALL_SIGN LIKE ('%ATL%') AND REPORT_TYPE IN ('FM-15', 'FM-16') AND DATE BETWEEN '2015-01-01' AND '2015-04-01'
# MAGIC ) 

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS group25.PHASE_1_PROCESSED_WEATHER_1;
# MAGIC CREATE TABLE group25.PHASE_1_PROCESSED_WEATHER_1 AS (
# MAGIC   SELECT * FROM WEATHER LIMIT 10
# MAGIC )

# COMMAND ----------

PHASE_1_PROCESSED_WEATHER_1.write.saveAsTable("")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS PHASE_1_RAW_TABLE;
# MAGIC CREATE TABLE PHASE_1_RAW_TABLE AS (
# MAGIC   SELECT * FROM (SELECT *, CONCAT('K', ORIGIN, ' ') AS CALL_SIGN FROM AIRLINES_2015) AS A
# MAGIC   LEFT JOIN WEATHER AS W
# MAGIC   ON W.CALL_SIGN=A.CALL_SIGN
# MAGIC   AND CAST(W.DATE AS DATE)=CAST(A.FL_DATE AS DATE)
# MAGIC   WHERE A.ORIGIN IN ('ORD', 'ATL') AND A.QUARTER=1 AND W.REPORT_TYPE IN ('FM-16', 'FM-15')
# MAGIC );
# MAGIC 
# MAGIC SELECT * FROM PHASE_1_RAW_TABLE LIMIT 10;

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM (SELECT *, CONCAT('K', ORIGIN, ' ') AS CALL_SIGN FROM AIRLINES_2015) AS A
# MAGIC LEFT JOIN 
# MAGIC (  SELECT DISTINCT
# MAGIC     STATION,
# MAGIC     CAST(DATE AS TIMESTAMP) AS DATE,
# MAGIC     SOURCE,
# MAGIC     LATITUDE,
# MAGIC     LONGITUDE,
# MAGIC     ELEVATION,
# MAGIC     NAME,
# MAGIC     REPORT_TYPE,
# MAGIC     CALL_SIGN,
# MAGIC     QUALITY_CONTROL,
# MAGIC     WND,
# MAGIC     CIG,
# MAGIC     VIS,
# MAGIC     TMP,
# MAGIC     DEW,
# MAGIC     SLP,
# MAGIC     REM,
# MAGIC     AA1,
# MAGIC     AA2,
# MAGIC     AA3,
# MAGIC     AA4,
# MAGIC     AB1,
# MAGIC     AD1,
# MAGIC     AE1,
# MAGIC     AH1,
# MAGIC     AH2,
# MAGIC     AH3,
# MAGIC     AH4,
# MAGIC     AH5,
# MAGIC     AH6,
# MAGIC     AI1,
# MAGIC     AI2,
# MAGIC     AI3,
# MAGIC     AI4,
# MAGIC     AI5,
# MAGIC     AI6,
# MAGIC     AJ1,
# MAGIC     AK1,
# MAGIC     AL1,
# MAGIC     AL2,
# MAGIC     AL3 FROM WEATHER) AS W
# MAGIC ON W.CALL_SIGN=A.CALL_SIGN
# MAGIC AND CAST(W.DATE AS DATE)=CAST(A.FL_DATE AS DATE)
# MAGIC WHERE A.ORIGIN='ORD' AND A.QUARTER=1
# MAGIC ORDER BY A.ORIGIN, A.OP_CARRIER_FL_NUM, W.DATE ASC ;

# COMMAND ----------

spark.sql("CREATE TEMPORARY VIEW AIRLINES_DATA USING parquet OPTIONS (path \"dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*.parquet\")")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM WEATHER WHERE CALL_SIGN LIKE '%KORD%' LIMIT 10
# MAGIC -- spark.sql("SELECT * FROM WEATHER LIMIT 10").show

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT REPORT_TYPE, COUNT(*) FROM WEATHER WHERE CALL_SIGN != '99999' GROUP BY REPORT_TYPE 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM WEATHER WHERE REPORT_TYPE LIKE '%SAO%' AND NAME LIKE '%CHICAGO%' LIMIT 100

# COMMAND ----------

# MAGIC %md # Stations

# COMMAND ----------

stations = spark.read.option("header", "true").csv("dbfs:/mnt/mids-w261/DEMO8/gsod/stations.csv.gz")

# COMMAND ----------

display(stations)

# COMMAND ----------

from pyspark.sql import functions as f
stations.where(f.col('call').contains('KORD')).take(10)

# COMMAND ----------

stations.select(['name', 'call']).where(f.col('call').contains('KORD')).distinct().take(100)

# COMMAND ----------

display(stations.select('name').distinct())

# COMMAND ----------

weather.select('NAME').distinct().count()

# COMMAND ----------

display(weather.select('name').distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC # __Section 1__ - Question Formulation

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC # __Section 2__ - Algorithm Explanation

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC # __Section 3__ - EDA & Challenges

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC # __Section 4__ - Algorithm Implementation

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC # __Section 5__ - Course Concepts

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

