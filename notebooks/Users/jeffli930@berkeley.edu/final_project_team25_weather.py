# Databricks notebook source
# MAGIC %md
# MAGIC # w261 Final Project - Weather

# COMMAND ----------

# MAGIC %md
# MAGIC 25   
# MAGIC Justin Trobec, Jeff Li, Sonya Chen, Karthik Srinivasan
# MAGIC Spring 2021, section 5, Team 25

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook contains the processing for Weather
# MAGIC 
# MAGIC DO NOT run this notebook end to end, it will take a long time if you do so
# MAGIC 
# MAGIC Helpful Links:
# MAGIC - https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
# MAGIC - https://www.visualcrossing.com/resources/documentation/weather-data/how-we-process-integrated-surface-database-historical-weather-data/

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Jeff Note - for additional info on these fields I suggest using Ctrl-F to look through the fields on the NOAA isd document. I used my best judgement in some of these cases,  but we may want to walk through it together.
# MAGIC 
# MAGIC ## Field Processing Notes
# MAGIC 
# MAGIC ### AA1
# MAGIC 
# MAGIC #### Filtering Criteria
# MAGIC 
# MAGIC 
# MAGIC #### Definition
# MAGIC 
# MAGIC * LIQUID-PRECIPITATION occurrence identifier
# MAGIC * The identifier that represents an episode of LIQUID-PRECIPITATION
# MAGIC 
# MAGIC #### Attributes
# MAGIC 
# MAGIC Period quantity - 0-98 scaling (99 is null), unit of hours - filter out 99
# MAGIC 
# MAGIC Depth dimension - 0-9998, 9999 is null, millimeters - filter out 9999
# MAGIC 
# MAGIC Condition code (used for filtering Only)
# MAGIC 
# MAGIC The code that denotes whether a LIQUID-PRECIPITATION depth dimension was a trace value.
# MAGIC 
# MAGIC 
# MAGIC * 1 = Measurement impossible or inaccurate
# MAGIC * 2 = Trace
# MAGIC * 3 = Begin accumulated period (precipitation amount missing until end of accumulated period)
# MAGIC * 4 = End accumulated period
# MAGIC * 5 = Begin deleted period (precipitation amount missing due to data problem)
# MAGIC * 6 = End deleted period
# MAGIC * 7 = Begin missing period
# MAGIC * 8 = End missing period
# MAGIC * E = Estimated data value (eg, from nearby station)
# MAGIC * I = Incomplete precipitation amount, excludes one or more missing reports, such as one or more 15-minute reports
# MAGIC  not included in the 1-hour precipitation total
# MAGIC * J = Incomplete precipitation amount, excludes one or more erroneous reports, such as one or more 1-hour
# MAGIC  precipitation amounts excluded from the 24-hour total
# MAGIC * 9 = Missing
# MAGIC 
# MAGIC ### AJ1
# MAGIC 
# MAGIC 
# MAGIC #### Definition
# MAGIC 
# MAGIC * SNOW-DEPTH identifier
# MAGIC 
# MAGIC #### Attributes
# MAGIC 
# MAGIC SNOW-DEPTH dimension - 0000 - 1200, filter out 9999 (missing)
# MAGIC 
# MAGIC SNOW-DEPTH condition code
# MAGIC 
# MAGIC SNOW-DEPTH quality code (filter out 2,3,6,7)
# MAGIC 
# MAGIC SNOW-DEPTH equivalent water depth dimension
# MAGIC 
# MAGIC SNOW-DEPTH equivalent water condition code
# MAGIC 
# MAGIC SNOW-DEPTH equivalent water condition quality code
# MAGIC 
# MAGIC ### GA1
# MAGIC 
# MAGIC 
# MAGIC #### Definition
# MAGIC 
# MAGIC *  SKY-COVER-LAYER identifier
# MAGIC 
# MAGIC #### Attributes
# MAGIC 
# MAGIC SKY-COVER-LAYER coverage code - 00 - 09. Filter out 10, 99
# MAGIC 
# MAGIC SKY-COVER-LAYER coverage quality code
# MAGIC 
# MAGIC SKY-COVER-LAYER base height dimension
# MAGIC 
# MAGIC SKY-COVER-LAYER base height quality code
# MAGIC 
# MAGIC SKY-COVER-LAYER cloud type code
# MAGIC 
# MAGIC SKY-COVER-LAYER cloud type quality code
# MAGIC 
# MAGIC 
# MAGIC ### KA1
# MAGIC 
# MAGIC **Important** - I separated this one out into min and max based on the EXTREME-AIR-TEMPERATURE code Field. N & O denotes a min obs. M and P denotes a max obs.
# MAGIC 
# MAGIC #### Definition
# MAGIC 
# MAGIC *  EXTREME-AIR-TEMPERATURE identifier
# MAGIC 
# MAGIC #### Attributes
# MAGIC 
# MAGIC EXTREME-AIR-TEMPERATURE period quantity 001-480.
# MAGIC 
# MAGIC EXTREME-AIR-TEMPERATURE code
# MAGIC 
# MAGIC EXTREME-AIR-TEMPERATURE air temperature - : -0932 MAX: +0618, +9999 = missing
# MAGIC 
# MAGIC EXTREME-AIR-TEMPERATURE temperature quality code - filter out 2,3,6,7
# MAGIC 
# MAGIC 
# MAGIC ### WND
# MAGIC 
# MAGIC #### Definition
# MAGIC 
# MAGIC *  WIND-OBSERVATION identifier
# MAGIC 
# MAGIC #### Attributes
# MAGIC 
# MAGIC WIND-OBSERVATION direction angle - 0-360
# MAGIC 
# MAGIC WIND-OBSERVATION direction quality code
# MAGIC 
# MAGIC WIND-OBSERVATION direction type code
# MAGIC 
# MAGIC WIND-OBSERVATION direction speed rate 0000-0900 (meters per second)
# MAGIC 
# MAGIC WIND-OBSERVATION direction speed quality code
# MAGIC 
# MAGIC 
# MAGIC ### TMP
# MAGIC 
# MAGIC ####

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Imports

# COMMAND ----------

## imports

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql.functions import sum, avg, mean, max, min, count, col,when
from pyspark.sql.functions import col
from matplotlib import pyplot as plt
import seaborn as sns
sqlContext = SQLContext(sc)

# COMMAND ----------

test_mode = False

#strings to ingest test/prod data from
test_table = "group25.PHASE_1_PROCESSED_WEATHER_test" #test table
prod_table = "group25.PHASE_1_PROCESSED_WEATHER_a" #everything sans call-sign  99999
#prod_table = "group25.WEATHER_base_table" #fm -15 fm -16
prod_table2 = "group25.WEATHER_base_table_2" # non fm 15  & fm16


if test_mode == True:
  ingestion_table = test_table
else:
  ingestion_table = prod_table
  ingestion_table2 = prod_table2

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/weather_data"))

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2015.parquet/"))

# COMMAND ----------

# MAGIC %md # Weather
# MAGIC https://data.nodc.noaa.gov/cgi-bin/iso?id=gov.noaa.ncdc:C00532

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/datasets_final_project/weather_data"))

# COMMAND ----------

dbutils.fs.mkdir()

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

# MAGIC %md
# MAGIC ## SQL QUERIES FOR DATABASE

# COMMAND ----------

# MAGIC %md 
# MAGIC ### initialize first view

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS WEATHER;
# MAGIC 
# MAGIC CREATE TEMPORARY VIEW WEATHER 
# MAGIC USING parquet OPTIONS (path "dbfs:/mnt/mids-w261/datasets_final_project/weather_data/weather*.parquet");

# COMMAND ----------

# MAGIC %sql
# MAGIC --Use this table for testing/feature generation
# MAGIC DROP TABLE IF EXISTS group25.PHASE_1_PROCESSED_WEATHER_a;
# MAGIC CREATE TABLE group25.PHASE_1_PROCESSED_WEATHER_a AS (
# MAGIC   SELECT * FROM WEATHER
# MAGIC   WHERE (DATE BETWEEN '2015-01-01' AND '2019-12-31')
# MAGIC   and CALL_SIGN != '99999'
# MAGIC ) 

# COMMAND ----------

# MAGIC %md 
# MAGIC Use this table for testing/feature generation

# COMMAND ----------

# MAGIC %sql
# MAGIC --Use this table for testing/feature generation
# MAGIC DROP TABLE IF EXISTS group25.PHASE_1_PROCESSED_WEATHER_test;
# MAGIC CREATE TABLE group25.PHASE_1_PROCESSED_WEATHER_test AS (
# MAGIC   SELECT * FROM WEATHER
# MAGIC   WHERE (CALL_SIGN LIKE ('%ORD%') OR CALL_SIGN LIKE ('%ATL%')) AND (REPORT_TYPE IN ('FM-15', 'FM-16')) AND (DATE BETWEEN '2015-01-01' AND '2015-04-01')
# MAGIC ) 

# COMMAND ----------

# MAGIC %sql
# MAGIC --DROP TABLE IF EXISTS group25.WEATHER_base_table;
# MAGIC --CREATE TABLE group25.WEATHER_base_table AS (
# MAGIC --  SELECT * FROM WEATHER
# MAGIC -- WHERE REPORT_TYPE IN ('FM-15', 'FM-16') AND DATE BETWEEN '2015-01-01' AND '2019-12-31'
# MAGIC --) 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DROP TABLE IF EXISTS group25.WEATHER_base_table_2;
# MAGIC -- CREATE TABLE group25.WEATHER_base_table_2 AS (
# MAGIC --   SELECT * FROM WEATHER
# MAGIC --  WHERE REPORT_TYPE NOT IN ('FM-15', 'FM-16') and  DATE BETWEEN '2015-01-01' AND '2019-12-31' 
# MAGIC -- ) 

# COMMAND ----------

df = sqlContext.sql("select call_sign, date, station, aa1, aa2, aa3, aa4 from group25.WEATHER_base_table where AA1 != ''")

display(df)

# COMMAND ----------

display(df.describe())

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct date, call_sign, aa1, SPLIT(AA1, ',') AS AA1_ARRAY from group25.PHASE_1_PROCESSED_WEATHER_test
# MAGIC limit 100

# COMMAND ----------

# MAGIC %md 
# MAGIC # Generate Features to load into table

# COMMAND ----------

# MAGIC %md
# MAGIC ## AA1 - Episodes of rain occurence

# COMMAND ----------

#https://stackoverflow.com/questions/39255973/split-1-column-into-3-columns-in-spark-scala

AA1_df = sqlContext.sql("""
                          select 
                            distinct
                            call_sign,
                            report_type,
                            date,
                            to_date(date) as date1,
                            month(date) as month,
                            hour(date) as hour,
                            to_date(date + INTERVAL 2 HOUR) as date_delta2h,
                            hour(date + INTERVAL 2 HOUR) as hour_delta2h,
                            to_date(date + INTERVAL 4 HOUR) as date_delta4h,
                            hour(date + INTERVAL 4 HOUR) as hour_delta4h,
                            call_sign, 
                            SPLIT(AA1, ',') AS AA1_ARRAY,
                            AA1
                          from {0}
                          where AA1 != ''
                          """.format(ingestion_table))

AA1_df1 = AA1_df.select("date",
                        "report_type",
                        "date1",
                        "call_sign",
                        "month",
                        "hour",
                        "date_delta2h",
                        "hour_delta2h",
                        "date_delta4h",
                        "hour_delta4h",
                        "AA1",
                        "AA1_array", 
                         AA1_df.AA1_ARRAY[0].alias("AA1_period_quantity"),
                         AA1_df.AA1_ARRAY[1].alias("AA1_depth_dimension"),
                         AA1_df.AA1_ARRAY[2].alias("AA1_condition_code"),
                         AA1_df.AA1_ARRAY[3].alias("AA1_quality_code"),
                       ) \
                          .filter(~AA1_df.AA1_ARRAY[3].isin(2,3,6,7) 
                                  & (AA1_df.AA1_ARRAY[0] != 99)
                                  & (AA1_df.AA1_ARRAY[1] != 9999)
                                 ) \
                          .groupBy("call_sign","date1", "hour", "date_delta2h","hour_delta2h", "date_delta4h","hour_delta4h") \
                          .agg(count("AA1_depth_dimension").alias("AA1_count"),
                               sum("AA1_depth_dimension").alias("AA1_depth_dimension_sum"),
                               avg("AA1_depth_dimension").alias("AA1_depth_dimension_avg"),
                               min("AA1_depth_dimension").alias("AA1_depth_dimension_min"),
                               max("AA1_depth_dimension").alias("AA1_depth_dimension_max"),
                               sum("AA1_period_quantity").alias("AA1_period_quantity_sum"),
                               avg("AA1_period_quantity").alias("AA1_period_quantity_avg"),
                               min("AA1_period_quantity").alias("AA1_period_quantity_min"),
                               max("AA1_period_quantity").alias("AA1_period_quantity_max"),
                            )


#write to table
AA1_df1.createOrReplaceTempView("mytempTable") 

sqlContext.sql("DROP TABLE IF EXISTS group25.AA1_df");
sqlContext.sql("create table group25.AA1_df as select * from mytempTable");

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from group25.AA1_df
# MAGIC where call_sign != "99999"
# MAGIC limit 100;

# COMMAND ----------

# MAGIC %md
# MAGIC ## AJ1 - Snow-Depth Identifier

# COMMAND ----------

#https://stackoverflow.com/questions/39255973/split-1-column-into-3-columns-in-spark-scala

AJ1_df = sqlContext.sql("""
                          select 
                            distinct
                            report_type,
                            date, 
                            to_date(date) as date1,
                            month(date) as month,
                            hour(date) as hour,
                            to_date(date + INTERVAL 2 HOUR) as date_delta2h,
                            hour(date + INTERVAL 2 HOUR) as hour_delta2h,
                            to_date(date + INTERVAL 4 HOUR) as date_delta4h,
                            hour(date + INTERVAL 4 HOUR) as hour_delta4h,
                            call_sign, 
                            SPLIT(AJ1, ',') AS AJ1_ARRAY,
                            AJ1
                          from {0}
                          where AJ1 != ''
                          """.format(ingestion_table))

AJ1_df1 = AJ1_df.select("report_type",
                        "date",
                        "date1",
                        "call_sign",
                        "month",
                        "hour",
                        "date_delta2h",
                        "hour_delta2h",
                        "date_delta4h",
                        "hour_delta4h",
                        "AJ1",
                        "AJ1_array", 
                         AJ1_df.AJ1_ARRAY[0].alias("AJ1_dimension"),  
                         AJ1_df.AJ1_ARRAY[1].alias("AJ1_coverage_code"), #filter out 2, 3, 6, 7
                         AJ1_df.AJ1_ARRAY[2].alias("AJ1_quality_code"),
                         AJ1_df.AJ1_ARRAY[3].alias("AJ1_equiv_water_depth_dimension"),
                         AJ1_df.AJ1_ARRAY[4].alias("AJ1_equiv_water_condition_code"),
                         AJ1_df.AJ1_ARRAY[4].alias("AJ1_equiv_water_condition_quality_code")
                       ).filter((AJ1_df.AJ1_ARRAY[0] != 9999)
                                & ~(AJ1_df.AJ1_ARRAY[2].isin(2,3,6,7))
                               ) \
                          .groupBy("call_sign","date1", "hour", "date_delta2h","hour_delta2h", "date_delta4h","hour_delta4h") \
                          .agg(count("AJ1_dimension").alias("AJ1_count"),
                               sum("AJ1_dimension").alias("AJ1_dimension_sum"),
                               avg("AJ1_dimension").alias("AJ1_dimension_avg"),
                               min("AJ1_dimension").alias("AJ1_dimension_min"),
                               max("AJ1_dimension").alias("AJ1_dimension_max"),
                            )


#write to table
AJ1_df1.createOrReplaceTempView("mytempTable") 

sqlContext.sql("DROP TABLE IF EXISTS group25.AJ1_df");
sqlContext.sql("create table group25.AJ1_df as select * from mytempTable");

AJ1_df1.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## GA1 - Sky-Cover-Layer

# COMMAND ----------

#https://stackoverflow.com/questions/39255973/split-1-column-into-3-columns-in-spark-scala

GA1_df = sqlContext.sql("""
                          select 
                            distinct
                            report_type,
                            date, 
                            to_date(date) as date1,
                            month(date) as month,
                            hour(date) as hour,
                            to_date(date + INTERVAL 2 HOUR) as date_delta2h,
                            hour(date + INTERVAL 2 HOUR) as hour_delta2h,
                            to_date(date + INTERVAL 4 HOUR) as date_delta4h,
                            hour(date + INTERVAL 4 HOUR) as hour_delta4h,
                            call_sign, 
                            SPLIT(GA1, ',') AS GA1_ARRAY,
                            GA1
                          from {0}
                          where GA1 != ''
                          """.format(ingestion_table))

GA1_df1 = GA1_df.select("report_type",
                        "date",
                        "date1",
                        "call_sign",
                        "month",
                        "hour",
                        "date_delta2h",
                        "hour_delta2h",
                        "date_delta4h",
                        "hour_delta4h",
                        "GA1",
                        "GA1_array", 
                         GA1_df.GA1_ARRAY[0].alias("GA1_coverage_code"),  #I think we should use this one, ranges from 1-8  oktas
                         GA1_df.GA1_ARRAY[1].alias("GA1_coverage_quality_code"), #filter out 2, 3, 6, 7
                         GA1_df.GA1_ARRAY[2].alias("GA1_base_height_dimension"),
                         GA1_df.GA1_ARRAY[3].alias("GA1_base_height_quality_code"),
                         GA1_df.GA1_ARRAY[4].alias("GA1_cloud_type_code"),
                         GA1_df.GA1_ARRAY[4].alias("GA1_cloud_type_quality_code")
                       ).filter(~GA1_df.GA1_ARRAY[1].isin(2,3,6,7) 
                               & ~GA1_df.GA1_ARRAY[0].isin(10,99)
                                & (GA1_df.GA1_ARRAY[2] != "+99999")
                                & ~GA1_df.GA1_ARRAY[3].isin(2,3,6,7)
                               ) \
                          .groupBy("call_sign","date1", "hour", "date_delta2h","hour_delta2h", "date_delta4h","hour_delta4h") \
                          .agg(count("GA1_coverage_code").alias("GA1_count"),
                               sum("GA1_coverage_code").alias("GA1_coverage_code_sum"),
                               avg("GA1_coverage_code").alias("GA1_coverage_code_avg"),
                               min("GA1_coverage_code").alias("GA1_coverage_code_min"),
                               max("GA1_coverage_code").alias("GA1_coverage_code_max"),
                               sum("GA1_base_height_dimension").alias("GA1_base_height_dimension_sum"),
                               avg("GA1_base_height_dimension").alias("GA1_base_height_dimension_avg"),
                               min("GA1_base_height_dimension").alias("GA1_base_height_dimension_min"),
                               max("GA1_base_height_dimension").alias("GA1_base_height_dimension_max")
                            )


#write to table
GA1_df1.createOrReplaceTempView("mytempTable") 

sqlContext.sql("DROP TABLE IF EXISTS group25.GA1_df");
sqlContext.sql("create table group25.GA1_df as select * from mytempTable");

GA1_df1.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## KA1 - Extreme Air Temperature
# MAGIC 
# MAGIC Note I will filter this boil this down to max and min temps since they  both appear to be captured by this dimension

# COMMAND ----------

#https://stackoverflow.com/questions/39255973/split-1-column-into-3-columns-in-spark-scala



KA1_df = sqlContext.sql("""
                          select 
                            distinct
                            report_type,
                            date, 
                            to_date(date) as date1,
                            month(date) as month,
                            hour(date) as hour,
                            to_date(date + INTERVAL 2 HOUR) as date_delta2h,
                            hour(date + INTERVAL 2 HOUR) as hour_delta2h,
                            to_date(date + INTERVAL 4 HOUR) as date_delta4h,
                            hour(date + INTERVAL 4 HOUR) as hour_delta4h,
                            call_sign, 
                            SPLIT(KA1, ',') AS KA1_ARRAY,
                            KA1
                          from {0}
                          where KA1 != ''
                          """.format(ingestion_table))

#min values

KA1_min = KA1_df.select("report_type",
                        "date",
                        "date1",
                        "call_sign",
                        "month",
                        "hour",
                        "date_delta2h",
                        "hour_delta2h",
                        "date_delta4h",
                        "hour_delta4h",
                        "KA1",
                        "KA1_array", 
                         KA1_df.KA1_ARRAY[0].alias("KA1_period_quantity"),  #0-480 scalar
                         KA1_df.KA1_ARRAY[1].alias("KA1_code"), #N and O for min
                         (KA1_df.KA1_ARRAY[2]/10).alias("KA1_air_temperature"),
                         KA1_df.KA1_ARRAY[3].alias("KA1_temperature_quality_code"),
                       ).filter(KA1_df.KA1_ARRAY[1].isin("M","P") 
                                & (KA1_df.KA1_ARRAY[0] != 999)
                                & (KA1_df.KA1_ARRAY[2] != "+9999")
                                & ~KA1_df.KA1_ARRAY[3].isin(2,3,6,7)) \
                          .groupBy("call_sign","date1", "hour", "date_delta2h","hour_delta2h", "date_delta4h","hour_delta4h") \
                          .agg(count("KA1_code").alias("KA1_min_count"),
                               sum("KA1_period_quantity").alias("KA1_min_period_quantity_sum"),
                               avg("KA1_period_quantity").alias("KA1_min_period_quantity_avg"),
                               min("KA1_period_quantity").alias("KA1_min_period_quantity_min"),
                               max("KA1_period_quantity").alias("KA1_min_period_quantity_max"),
                               sum("KA1_air_temperature").alias("KA1_min_air_temperature_sum"),
                               avg("KA1_air_temperature").alias("KA1_min_air_temperature_avg"),
                               min("KA1_air_temperature").alias("KA1_min_air_temperature_min"),
                               max("KA1_air_temperature").alias("KA1_min_air_temperature_max"),
                            )


#write to table
KA1_min.createOrReplaceTempView("mytempTable") 

sqlContext.sql("DROP TABLE IF EXISTS group25.KA1_min_df");
sqlContext.sql("create table group25.KA1_min_df as select * from mytempTable");

#KA1_min.show()

##max
KA1_max = KA1_df.select("report_type",
                        "date",
                        "date1",
                        "call_sign",
                        "month",
                        "hour",
                        "date_delta2h",
                        "hour_delta2h",
                        "date_delta4h",
                        "hour_delta4h",
                        "KA1",
                        "KA1_array", 
                         KA1_df.KA1_ARRAY[0].alias("KA1_period_quantity"),  #0-480 scalar
                         KA1_df.KA1_ARRAY[1].alias("KA1_code"), #N and O for min
                         (KA1_df.KA1_ARRAY[2]/10).alias("KA1_air_temperature"),
                         KA1_df.KA1_ARRAY[3].alias("KA1_temperature_quality_code"),
                       ).filter(KA1_df.KA1_ARRAY[1].isin("N","O") & ~KA1_df.KA1_ARRAY[3].isin(2,3,6,7)) \
                          .groupBy("call_sign","date1", "hour", "date_delta2h","hour_delta2h", "date_delta4h","hour_delta4h") \
                          .agg(count("KA1_code").alias("KA1_max_count"),
                               sum("KA1_period_quantity").alias("KA1_max_period_quantity_sum"),
                               avg("KA1_period_quantity").alias("KA1_max_period_quantity_avg"),
                               min("KA1_period_quantity").alias("KA1_max_period_quantity_min"),
                               max("KA1_period_quantity").alias("KA1_max_period_quantity_max"),
                               sum("KA1_air_temperature").alias("KA1_max_air_temperature_sum"),
                               avg("KA1_air_temperature").alias("KA1_max_air_temperature_avg"),
                               min("KA1_air_temperature").alias("KA1_max_air_temperature_min"),
                               max("KA1_air_temperature").alias("KA1_max_air_temperature_max"),
                               
                            )

#write to table
KA1_max.createOrReplaceTempView("mytempTable") 

sqlContext.sql("DROP TABLE IF EXISTS group25.KA1_max_df");
sqlContext.sql("create table group25.KA1_max_df as select * from mytempTable");

KA1_min.show()

# COMMAND ----------

# MAGIC %md

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## TMP - air tempearture

# COMMAND ----------


TMP_df = sqlContext.sql("""
                          select 
                            distinct
                            report_type,
                            date, 
                            to_date(date) as date1,
                            month(date) as month,
                            hour(date) as hour,
                            to_date(date + INTERVAL 2 HOUR) as date_delta2h,
                            hour(date + INTERVAL 2 HOUR) as hour_delta2h,
                            to_date(date + INTERVAL 4 HOUR) as date_delta4h,
                            hour(date + INTERVAL 4 HOUR) as hour_delta4h,
                            call_sign, 
                            SPLIT(TMP, ',') AS TMP_ARRAY,
                            TMP
                          from {0}
                          where TMP != ''
                          """.format(ingestion_table))

TMP_df1 = TMP_df.select("date",
                        "report_type",
                        "date1",
                        "call_sign",
                        "month",
                        "hour",
                        "date_delta2h",
                        "hour_delta2h",
                        "date_delta4h",
                        "hour_delta4h",
                        "TMP",
                        "TMP_array", 
                         (TMP_df.TMP_ARRAY[0]/10).alias("TMP_air_temperature"),
                         TMP_df.TMP_ARRAY[1].alias("TMP_air_temp_quality_code")
                       ) \
                          .filter(~TMP_df.TMP_ARRAY[1].isin(2,3,6,7) &
                                  (TMP_df.TMP_ARRAY[0] != 9999) & 
                                  (TMP_df.TMP_ARRAY[0] >= -932) & 
                                  (TMP_df.TMP_ARRAY[0] <= 618) 
                                 ) \
                          .groupBy("call_sign","date1", "hour", "date_delta2h","hour_delta2h", "date_delta4h","hour_delta4h") \
                          .agg(count("TMP_air_temperature").alias("TMP_count"),
                               sum("TMP_air_temperature").alias("TMP_air_temperature_sum"),
                               avg("TMP_air_temperature").alias("TMP_air_temperature_avg"),
                               min("TMP_air_temperature").alias("TMP_air_temperature_min"),
                               max("TMP_air_temperature").alias("TMP_air_temperature_max")
                            )


#write to table
TMP_df1.createOrReplaceTempView("mytempTable") 

sqlContext.sql("DROP TABLE IF EXISTS group25.TMP_df");
sqlContext.sql("create table group25.TMP_df as select * from mytempTable");

# COMMAND ----------

# MAGIC %sql
# MAGIC select TMP_air_temperature_avg from group25.TMP_df where call_sign != '99999' 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## WND - Wind

# COMMAND ----------

#https://stackoverflow.com/questions/39255973/split-1-column-into-3-columns-in-spark-scala


WND_df = sqlContext.sql("""
                          select 
                            distinct
                            report_type,
                            date, 
                            to_date(date) as date1,
                            month(date) as month,
                            hour(date) as hour,
                            to_date(date + INTERVAL 2 HOUR) as date_delta2h,
                            hour(date + INTERVAL 2 HOUR) as hour_delta2h,
                            to_date(date + INTERVAL 4 HOUR) as date_delta4h,
                            hour(date + INTERVAL 4 HOUR) as hour_delta4h,
                            call_sign, 
                            SPLIT(WND, ',') AS WND_ARRAY,
                            WND
                          from {0}
                          where WND != ''
                          """.format(ingestion_table))


WND_df1 = WND_df.select("report_type",
                        "date",
                        "date1",
                        "call_sign",
                        "month",
                        "hour",
                        "date_delta2h",
                        "hour_delta2h",
                        "date_delta4h",
                        "hour_delta4h",
                        "WND",
                        "WND_array", 
                         WND_df.WND_ARRAY[0].alias("WND_direction_angle"),  #0-480 scalar
                         WND_df.WND_ARRAY[1].alias("WND_direction_quality_code"), #N and O for min
                         WND_df.WND_ARRAY[2].alias("WND_obs_type_code"),
                         WND_df.WND_ARRAY[3].alias("WND_speed"),
                         WND_df.WND_ARRAY[4].alias("WND_speed_quality_code"),
                       ).filter(~WND_df.WND_ARRAY[1].isin(2,3,6,7) 
                                & (WND_df.WND_ARRAY[0] != 999)
                                & (WND_df.WND_ARRAY[3] != 9999)
                                & ~WND_df.WND_ARRAY[4].isin(2,3,6,7) 
                               ) \
                          .groupBy("call_sign","date1", "hour", "date_delta2h","hour_delta2h", "date_delta4h","hour_delta4h") \
                          .agg(count("WND").alias("WND_count"),
                               avg("WND_direction_angle").alias("WND_direction_angle_avg"),
                               min("WND_direction_angle").alias("WND_direction_angle_min"),
                               max("WND_direction_angle").alias("WND_direction_angle_max"),
                               sum("WND_speed").alias("WND_speed_sum"),
                               avg("WND_speed").alias("WND_speed_avg"),
                               min("WND_speed").alias("WND_speed_min"),
                               max("WND_speed").alias("WND_speed_max")
                            )


WND_df1 = WND_df1.withColumn("WND_direction_angle_avg_dir", f.when(col("WND_direction_angle_avg").between(0,90), "NE")
      .when(col("WND_direction_angle_avg").between(90,180), "SE")
      .when(col("WND_direction_angle_avg").between(180,270), "SW")
      .otherwise("NW"))


#write to table
WND_df1.createOrReplaceTempView("mytempTable") 

sqlContext.sql("DROP TABLE IF EXISTS group25.WND_df");
sqlContext.sql("create table group25.WND_df as select * from mytempTable");

WND_df1.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## MV - weather events

# COMMAND ----------

MV1_df = sqlContext.sql("""
                          select 
                            distinct
                            report_type,
                            date, 
                            to_date(date) as date1,
                            month(date) as month,
                            hour(date) as hour,
                            to_date(date + INTERVAL 2 HOUR) as date_delta2h,
                            hour(date + INTERVAL 2 HOUR) as hour_delta2h,
                            to_date(date + INTERVAL 4 HOUR) as date_delta4h,
                            hour(date + INTERVAL 4 HOUR) as hour_delta4h,
                            call_sign, 
                            SPLIT(MV1, ',') AS MV1_ARRAY,
                            MV1
                          from {0}
                          """.format(ingestion_table))


MV1_df1 = MV1_df.select("report_type",
                        "date",
                        "date1",
                        "call_sign",
                        "month",
                        "hour",
                        "date_delta2h",
                        "hour_delta2h",
                        "date_delta4h",
                        "hour_delta4h",
                        "MV1",
                        "MV1_array",
                        MV1_df.MV1_ARRAY[0].alias("MV1_condition_code"),  #0-480 scalar
                        MV1_df.MV1_ARRAY[1].alias("MV1_quality_condition_code"), #N and O for min
                       ).filter(~MV1_df.MV1_ARRAY[1].isin(6,7) 
                                & ~MV1_df.MV1_ARRAY[0].isin('00','99') 
                               ) \
                          .groupBy("call_sign","date1", "hour", "date_delta2h","hour_delta2h", "date_delta4h","hour_delta4h") \
                          .agg(
                                count(when(col('MV1_condition_code') == '01', True)).alias('Thunderstorm_mv1'),
                                count(when(col('MV1_condition_code') == '02', True)).alias('Showers_mv1'),
                                count(when(col('MV1_condition_code') == '03', True)).alias('Sandstorm_mv1'),
                                count(when(col('MV1_condition_code') == '04', True)).alias('Sand_dust_mv1'),
                                count(when(col('MV1_condition_code') == '05', True)).alias('Duststorm_mv1'),
                                count(when(col('MV1_condition_code') == '06', True)).alias('BlowingSnow_mv1'),
                                count(when(col('MV1_condition_code') == '07', True)).alias('BlowingSand_mv1'),
                                count(when(col('MV1_condition_code') == '08', True)).alias('BlowingDust_mv1'),
                                  count(when(col('MV1_condition_code') == '09', True)).alias('Fog_mv1')
                            )


MV2_df = sqlContext.sql("""
                          select 
                            distinct
                            report_type as report_type2,
                            date as date2, 
                            to_date(date) as date12,
                            hour(date) as hour2,
                            call_sign as call_sign2, 
                            SPLIT(MV2, ',') AS MV2_ARRAY,
                            MV2
                          from {0}
                          """.format(ingestion_table))


MV2_df1 = MV2_df.select(
                        "report_type2",
                        "date12",
                        "call_sign2",
                        "hour2",
                        "MV2",
                        "MV2_array",
                        MV2_df.MV2_ARRAY[0].alias("MV2_condition_code"),  #0-480 scalar
                        MV2_df.MV2_ARRAY[1].alias("MV2_quality_condition_code"), #N and O for min
                       ).filter(~MV2_df.MV2_ARRAY[1].isin(6,7) 
                                & ~MV2_df.MV2_ARRAY[0].isin('00','99') 
                               ) \
                          .groupBy("call_sign2","date12", "hour2") \
                          .agg(
                                count(when(col('MV2_condition_code') == '01', True)).alias('Thunderstorm_mv2'),
                                count(when(col('MV2_condition_code') == '02', True)).alias('Showers_mv2'),
                                count(when(col('MV2_condition_code') == '03', True)).alias('Sandstorm_mv2'),
                                count(when(col('MV2_condition_code') == '04', True)).alias('Sand_dust_mv2'),
                                count(when(col('MV2_condition_code') == '05', True)).alias('Duststorm_mv2'),
                                count(when(col('MV2_condition_code') == '06', True)).alias('BlowingSnow_mv2'),
                                count(when(col('MV2_condition_code') == '07', True)).alias('BlowingSand_mv2'),
                                count(when(col('MV2_condition_code') == '08', True)).alias('BlowingDust_mv2'),
                                  count(when(col('MV2_condition_code') == '09', True)).alias('Fog_mv2')
                            )


MV_df= MV1_df1.join(MV2_df1, (MV1_df1.date1 == MV2_df1.date12) & (MV1_df1.hour == MV2_df1.hour2)& (MV1_df1.call_sign == MV2_df1.call_sign2), "outer")

display(MV_df)

# COMMAND ----------

MV_df = MV_df.withColumn('MV_Thunderstorm', 
                         when((MV_df["Thunderstorm_mv1"] > 0) | (MV_df["Thunderstorm_mv2"] > 0), 1).otherwise(0))

MV_df = MV_df.withColumn('MV_Showers', 
                         when((MV_df["Showers_mv1"] > 0) | (MV_df["Showers_mv2"] > 0), 1).otherwise(0))

MV_df = MV_df.withColumn('MV_Sand_or_Dust', 
                         when((MV_df["Sandstorm_mv1"] > 0) | 
                              (MV_df["Sandstorm_mv2"] > 0)  | 
                              (MV_df["Sand_dust_mv1"] > 0)  | 
                              (MV_df["Sand_dust_mv2"] > 0) | 
                              (MV_df["Duststorm_mv1"] > 0)  | 
                              (MV_df["Duststorm_mv2"] > 0) | 
                               (MV_df["BlowingSand_mv1"] > 0)  | 
                              (MV_df["BlowingSand_mv2"] > 0) |                              
                               (MV_df["BlowingDust_mv1"] > 0)  | 
                              (MV_df["BlowingDust_mv2"] > 0)  , 1).otherwise(0))
MV_df = MV_df.withColumn('MV_BlowingSnow', 
                         when((MV_df["BlowingSnow_mv1"] > 0) | (MV_df["BlowingSnow_mv2"] > 0), 1).otherwise(0))


MV_df = MV_df.withColumn('MV_Fog', 
                         when((MV_df["Fog_mv1"] > 0) | (MV_df["Fog_mv2"] > 0), 1).otherwise(0))

# COMMAND ----------

MV_df1 = MV_df.select(
            "date1",
            "call_sign",
            "hour",
            "date_delta2h",
            "hour_delta2h",
            "date_delta4h",
            "hour_delta4h",
            'MV_Thunderstorm',
            'MV_Showers',
            'MV_Sand_or_Dust', 
            'MV_BlowingSnow',
             'MV_Fog'
            )

MV_df1.createOrReplaceTempView("mytempTable") 

sqlContext.sql("DROP TABLE IF EXISTS group25.MV_df1");
sqlContext.sql("create table group25.MV_df1 as select * from mytempTable");

display(MV_df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SLP Pressure

# COMMAND ----------

#https://stackoverflow.com/questions/39255973/split-1-column-into-3-columns-in-spark-scala


SLP_df = sqlContext.sql("""
                          select 
                            distinct
                            report_type,
                            date, 
                            to_date(date) as date1,
                            month(date) as month,
                            hour(date) as hour,
                            to_date(date + INTERVAL 2 HOUR) as date_delta2h,
                            hour(date + INTERVAL 2 HOUR) as hour_delta2h,
                            to_date(date + INTERVAL 4 HOUR) as date_delta4h,
                            hour(date + INTERVAL 4 HOUR) as hour_delta4h,
                            call_sign, 
                            SPLIT(SLP, ',') AS SLP_ARRAY,
                            SLP
                          from {0}
                          where SLP != ''
                          AND SPLIT(SLP, ',')[0] != '99999'
                          """.format(ingestion_table))


SLP_df1 = SLP_df.select("report_type",
                        "date",
                        "date1",
                        "call_sign",
                        "month",
                        "hour",
                        "date_delta2h",
                        "hour_delta2h",
                        "date_delta4h",
                        "hour_delta4h",
                        "SLP",
                        "SLP_array", 
                         (SLP_df.SLP_ARRAY[0]/10).alias("SLP_atmospheric_pressure"),  #0-480 scalar
                         SLP_df.SLP_ARRAY[1].alias("SLP_quality_code"), #N and O for min
                       ).filter(~SLP_df.SLP_ARRAY[1].isin(2,3,6,7) 
                                & (SLP_df.SLP_ARRAY[0] >= 8600)
                             & (SLP_df.SLP_ARRAY[0] <= 10900)
                               ) \
                          .groupBy("call_sign","date1", "hour", "date_delta2h","hour_delta2h", "date_delta4h","hour_delta4h") \
                          .agg(count("SLP").alias("SLP_count"),
                               avg("SLP_atmospheric_pressure").alias("SLP_atmospheric_pressure_avg"),
                               min("SLP_atmospheric_pressure").alias("SLP_atmospheric_pressure_min"),
                               max("SLP_atmospheric_pressure").alias("SLP_atmospheric_pressure_max"),
                            )

#write to table
SLP_df1.createOrReplaceTempView("mytempTable") 

sqlContext.sql("DROP TABLE IF EXISTS group25.SLP_df");
sqlContext.sql("create table group25.SLP_df as select * from mytempTable");

SLP_df1.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Concatenate individual tables into one by joining individual DFs outlined in sections above

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists group25.WEATHER_main;
# MAGIC 
# MAGIC create table group25.WEATHER_main as
# MAGIC SELECT A.CALL_SIGN,
# MAGIC        A.DATE1,
# MAGIC        A.HOUR,
# MAGIC        A.DATE_DELTA2H,
# MAGIC        A.HOUR_DELTA2H,
# MAGIC        A.DATE_DELTA4H,
# MAGIC        A.HOUR_DELTA4H,
# MAGIC        A.AA1_DEPTH_DIMENSION_AVG,
# MAGIC        A.AA1_PERIOD_QUANTITY_AVG,
# MAGIC        B.AJ1_DIMENSION_AVG,
# MAGIC        C.GA1_COVERAGE_CODE_AVG,
# MAGIC        C.GA1_BASE_HEIGHT_DIMENSION_AVG,
# MAGIC        D.KA1_MIN_PERIOD_QUANTITY_AVG,
# MAGIC         D.KA1_MIN_AIR_TEMPERATURE_AVG,
# MAGIC         E.KA1_MAX_PERIOD_QUANTITY_AVG,      
# MAGIC         E.KA1_MAX_AIR_TEMPERATURE_AVG,
# MAGIC         F.WND_DIRECTION_ANGLE_AVG_DIR,
# MAGIC        F.WND_SPEED_AVG AS WND_SPEED_AVG,
# MAGIC        G.TMP_AIR_TEMPERATURE_AVG,
# MAGIC             COALESCE(H.MV_THUNDERSTORM,0) AS MV_THUNDERSTORM,
# MAGIC             COALESCE(H.MV_SHOWERS,0) AS MV_SHOWERS,
# MAGIC             COALESCE(H.MV_SAND_OR_DUST, 0) AS MV_SAND_OR_DUST,
# MAGIC             COALESCE(H.MV_BLOWINGSNOW,0) AS MV_BLOWINGSNOW,
# MAGIC             COALESCE(H.MV_FOG,0) AS MV_FOG,
# MAGIC          J.SLP_ATMOSPHERIC_PRESSURE_AVG
# MAGIC         from 
# MAGIC           group25.AA1_df a left outer join group25.AJ1_df b
# MAGIC               on a.call_sign = b.call_sign
# MAGIC               and a.date1 = b.date1
# MAGIC               and a.hour = b.hour
# MAGIC               left outer join group25.GA1_df c
# MAGIC                   on a.call_sign = c.call_sign
# MAGIC                   and a.date1 = c.date1
# MAGIC                   and a.hour = c.hour
# MAGIC               left outer join group25.KA1_min_df d
# MAGIC                   on a.call_sign = d.call_sign
# MAGIC                   and a.date1 = d.date1
# MAGIC                   and a.hour = d.hour
# MAGIC               left outer join group25.KA1_max_df e
# MAGIC                   on a.call_sign = e.call_sign
# MAGIC                   and a.date1 = e.date1
# MAGIC                   and a.hour = e.hour
# MAGIC               left outer join group25.WND_df f
# MAGIC                   on a.call_sign = f.call_sign
# MAGIC                   and a.date1 = f.date1
# MAGIC                   and a.hour = f.hour
# MAGIC               left outer join group25.TMP_df g
# MAGIC                   on a.call_sign = g.call_sign
# MAGIC                   and a.date1 = g.date1
# MAGIC                   and a.hour = g.hour
# MAGIC               left outer join group25.MV_df1 h
# MAGIC                   on a.call_sign = h.call_sign
# MAGIC                   and a.date1 = h.date1
# MAGIC                   and a.hour = h.hour
# MAGIC               left outer join group25.SLP_df j
# MAGIC                   on a.call_sign = j.call_sign
# MAGIC                   and a.date1 = j.date1
# MAGIC                   and a.hour = j.hour
# MAGIC         where a.call_sign != "99999"
# MAGIC         order by call_sign, date1, hour

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists group25.WEATHER_main_fm15_impute;
# MAGIC 
# MAGIC create table group25.WEATHER_main_fm15_impute as
# MAGIC select a.report_type,
# MAGIC        a.call_sign,
# MAGIC        a.date1,
# MAGIC        a.hour,
# MAGIC        a.date_delta2h,
# MAGIC        a.hour_delta2h,
# MAGIC        a.date_delta4h,
# MAGIC        a.hour_delta4h,
# MAGIC        coalesce(a.AA1_depth_dimension_avg,0) as AA1_depth_dimension_avg,
# MAGIC        coalesce(a.AA1_period_quantity_avg,0) as AA1_period_quantity_avg,
# MAGIC         coalesce(b.AJ1_dimension_avg,0) as AJ1_dimension_avg,
# MAGIC         coalesce(c.GA1_coverage_code_avg,0) as GA1_coverage_code_avg,
# MAGIC         coalesce(c.GA1_base_height_dimension_avg,0) as GA1_base_height_dimension_avg,
# MAGIC         coalesce(d.KA1_min_period_quantity_avg,0) as  KA1_min_period_quantity_avg,
# MAGIC         d.KA1_min_air_temperature_avg,
# MAGIC         e.KA1_max_period_quantity_avg,
# MAGIC         case when d.KA1_min_air_temperature_avg is null then 0 else 1 end as KA1_min_air_temperature_avg_bool,
# MAGIC         case when e.KA1_max_air_temperature_avg is null then 0 else 1 end as KA1_max_air_temperature_avg_bool,       
# MAGIC         coalesce(e.KA1_max_air_temperature_avg,0) as KA1_max_air_temperature_avg,
# MAGIC         f.WND_direction_angle_avg_dir,
# MAGIC         coalesce(f.WND_speed_avg,0) as WND_speed_avg,
# MAGIC         coalesce(g.TMP_air_temperature_avg,0) as TMP_speed_avg,
# MAGIC             coalesce(h.MV_Thunderstorm,0) as MV_Thunderstorm,
# MAGIC             coalesce(h.MV_Showers,0) as MV_Showers,
# MAGIC             coalesce(h.MV_Sand_or_Dust, 0) as MV_Sand_or_dust,
# MAGIC             coalesce(h.MV_BlowingSnow,0) as MV_BlowingSnow,
# MAGIC              coalesce(h.MV_Fog,0) as MV_Fog
# MAGIC         from 
# MAGIC           group25.AA1_df a left outer join group25.AJ1_df b
# MAGIC               and a.call_sign = b.call_sign
# MAGIC               and a.date1 = b.date1
# MAGIC               and a.hour = b.hour
# MAGIC               left outer join group25.GA1_df c
# MAGIC                   and a.call_sign = c.call_sign
# MAGIC                   and a.date1 = c.date1
# MAGIC                   and a.hour = c.hour
# MAGIC               left outer join group25.KA1_min_df d
# MAGIC                   and a.call_sign = d.call_sign
# MAGIC                   and a.date1 = d.date1
# MAGIC                   and a.hour = d.hour
# MAGIC               left outer join group25.KA1_max_df e
# MAGIC                   and a.call_sign = e.call_sign
# MAGIC                   and a.date1 = e.date1
# MAGIC                   and a.hour = e.hour
# MAGIC               left outer join group25.WND_df f
# MAGIC                   and a.call_sign = f.call_sign
# MAGIC                   and a.date1 = f.date1
# MAGIC                   and a.hour = f.hour
# MAGIC               left outer join group25.TMP_df g
# MAGIC                   and a.call_sign = g.call_sign
# MAGIC                   and a.date1 = g.date1
# MAGIC                   and a.hour = g.hour
# MAGIC               left outer join group25.MV_df1 h
# MAGIC                   and a.call_sign = h.call_sign
# MAGIC                   and a.date1 = h.date1
# MAGIC                   and a.hour = h.hour
# MAGIC         and a.call_sign != "99999"
# MAGIC         order by call_sign, date1, hour

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from group25.weather_main
# MAGIC limit 100

# COMMAND ----------

# MAGIC %sql
# MAGIC select max(AA1_depth_dimension_avg) from group25.weather_main_fm15;

# COMMAND ----------

# MAGIC %md 
# MAGIC # EDA

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) as count, "AA1_count" as column_count from group25.aa1_df
# MAGIC union
# MAGIC select count(*) as count, "AJ1_count" as column_count from group25.aj1_df
# MAGIC union
# MAGIC select count(*) as count, "GA1_count" as column_count from group25.ga1_df
# MAGIC union
# MAGIC select count(*) as count, "KA1_max_count" as column_count from group25.ka1_max_df
# MAGIC union 
# MAGIC select count(*) as count, "KA1_min_count" as column_count from group25.ka1_min_df
# MAGIC union
# MAGIC select count(*) as count, "TMP_count" as column_count  from group25.tmp_df
# MAGIC union
# MAGIC select count(*) as count, "WND_count" as column_count  from group25.wnd_df
# MAGIC union
# MAGIC select count(*) as count, "MV_count" as column_count  from group25.MV_df1
# MAGIC union
# MAGIC select count(*) as count, "SLP_count" as column_count  from group25.SLP_df
# MAGIC order by column_count
# MAGIC ;

# COMMAND ----------

$aa1 = sqlContext.sql('SELECT * FROM group25.WEATHER_main_fm15').sample(False,0.01).toPandas()

#test = aa1.hist(column='AA1_depth_dimension_avg', bins=10)
#test2 = aa1.hist(column='AA1_period_quantity_avg', bins=10)

# COMMAND ----------

#print(aa1.shape)
#print(aa1.columns)
test = aa1.hist(column='AA1_depth_dimension_avg', bins=10)
plt.title('AA1_depth_dimension_avg Histogram')
plt.xlabel('Average depth dimension per hour for an airport')
plt.ylabel('Number of hourly observations across all airports')
test2 = aa1.hist(column='AA1_period_quantity_avg', bins=10)
plt.title('AA1_period_quantity_avg Histogram')
plt.xlabel('Average period quantity per hour for an airport')
plt.ylabel('Number of hourly observations across all airports')

test3 = aa1.hist(column='KA1_min_air_temperature_avg', bins=10)

test4 = aa1.hist(column='KA1_max_air_temperature_avg', bins=10)

# COMMAND ----------

for i in aa1.columns:
  try:
    aa1.hist(column = i, bins = 10)
  except:
    pass

# COMMAND ----------

for i in aa1.columns:
  print(i)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from group25.weather_main_fm15

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.dropdown("Weather Column For Plotting", weather_numerical_columns[0], [str(col) for col in weather_numerical_columns + numerical_columns])
dbutils.widgets.dropdown("OverSampling", "ClassWeighted", [str(col) for col in ['ClassWeighted', 'BootStrapping', 'NoOverSampling']])

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists test;
# MAGIC create temporary view test as 
# MAGIC select *
# MAGIC from group25.weather_base_table
# MAGIC where MV1 != ""

# COMMAND ----------

# MAGIC %sql
# MAGIC select MV1, call_sign, date, latitude, longitude
# MAGIC from test
# MAGIC where call_sign != '99999'
# MAGIC limit 100

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from group25.weather_base_table_2
# MAGIC where KB1 != ""
# MAGIC and call_sign == "KORD"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # null imputation test

# COMMAND ----------

df =  sqlContext.sql("""select *
                     from group25.weather_main_fm15
                     """)

# COMMAND ----------

# https://stackoverflow.com/questions/46845672/median-quantiles-within-pyspark-groupby

categorical_columns = [
 'WND_direction_angle_avg_dir', 
  'MV_Thunderstorm',
 'MV_Showers',
 'MV_Sand_or_dust',
 'MV_BlowingSnow',
 'MV_Fog']

continuous_columns = ['AA1_depth_dimension_avg',
 'AA1_period_quantity_avg',
 'AJ1_dimension_avg',
 'GA1_coverage_code_avg',
 'GA1_base_height_dimension_avg',
 'KA1_min_period_quantity_avg',
 'KA1_min_air_temperature_avg',
 'KA1_max_period_quantity_avg',
 'KA1_max_air_temperature_avg',
 'WND_speed_avg',
 'TMP_air_temperature_avg',
 'SLP_atmospheric_pressure_avg']


# COMMAND ----------

str = 'SLP_atmospheric_pressure_avg'

#function to generate median
magic_percentile = f.expr('percentile_approx(' + str + ', 0.5)')

#get median for each group
median_by_airport = df.groupBy('call_sign').agg(magic_percentile.alias('med_val'))

#get overall median, if there is no value for an airport
total_median = df.agg(magic_percentile.alias(str+'_med_val_total')).collect()[0][0]

# COMMAND ----------

print(total_median)

# COMMAND ----------

from pyspark.sql.functions import lit, coalesce
df2 = df.select('call_sign', 'SLP_atmospheric_pressure_avg')
df2 = df2.join(median_by_airport,['call_sign'])
df2 = df2.withColumn('total', lit(total_median))
df2 = df2.withColumn('SLP_atmospheric_pressure_avg_impute', coalesce(df2.SLP_atmospheric_pressure_avg, df2.med_val, df2.total))
display(df2)

# COMMAND ----------

display(df2.filter(df2['SLP_atmospheric_pressure_avg'].isNull() & ~df2.med_val.isNull()))

# COMMAND ----------

from pyspark.sql.functions import lit, coalesce


categorical_columns = [
 'WND_direction_angle_avg_dir', 
  'MV_Thunderstorm',
 'MV_Showers',
 'MV_Sand_or_dust',
 'MV_BlowingSnow',
 'MV_Fog']

continuous_columns = ['AA1_depth_dimension_avg',
 'AA1_period_quantity_avg',
 'AJ1_dimension_avg',
 'GA1_coverage_code_avg',
 'GA1_base_height_dimension_avg',
 'KA1_min_period_quantity_avg',
 'KA1_min_air_temperature_avg',
 'KA1_max_period_quantity_avg',
 'KA1_max_air_temperature_avg',
 'WND_speed_avg',
 'TMP_air_temperature_avg',
 'SLP_atmospheric_pressure_avg']


#function to generate median
magic_percentile = f.expr('percentile_approx(' + str + ', 0.5)')


def imputation_method1(categorical,
                      continuous,
                      base_dataframe):
  """
  Takes pyspark dataframe as input
  Function to impute null values using the median across each call sign
  if the median cannot be found for a call_sign, use median across the entire dataset
  categorical - a list of categorical columns
  continuous - a list of continuous columns
  base_dataframe - pyspark dataframe with null values
  """
  #df2 to to be joined to
  df2 = base_dataframe
  # get imputation for continuous columns first. 
  for i in continuous:

    #get median for each group. For now, use get medians across all call_signs
    median_by_airport = df.groupBy('call_sign').agg(magic_percentile.alias(i + '_med_val'))

    #get overall median across entire dataset, if there is no value for an airport
    total_median = df.agg(magic_percentile.alias(i+'_med_val_total')).collect()[0][0]

    # join median across each group and overall median
    df2 = df2.join(median_by_airport,['call_sign'])
    df2 = df2.withColumn(i+'_med_val_total', lit(total_median))

    # Perform imputation. 
    df2 = df2.withColumn( i + '_impute', coalesce(df2[i], df2[ i + "_med_val"], df2[i+'_med_val_total']))

    #drop excessive columns for impute
    df2 = df2.drop(i + "_med_val")
    df2 = df2.drop(i+"_med_val_total")

  #impute a 0 for null categorical fields
  for j in categorical:
    df2 = df2.withColumn(j + '_impute', coalesce(df2[j], lit(0)))
  
  #return imputed df
  return df2

imputed_dataframe = imputation_method1(categorical_columns, continuous_columns, df)

display(imputed_dataframe)
  
  

# COMMAND ----------

from sklearn.impute import KNNImputer

# COMMAND ----------

display(df2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Evalulate periodic data vs other observations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Episodic

# COMMAND ----------

# MAGIC %sql
# MAGIC create temporary view aa1_examination as
# MAGIC select report_type, call_sign, year(date) as date, count(*), avg(SPLIT(AA1, ',')[1]) as avg from group25.PHASE_1_PROCESSED_WEATHER_a
# MAGIC where SPLIT(AA1, ',')[1] != '9999'
# MAGIC group by report_type, call_sign, year(date)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from aa1_examination

# COMMAND ----------

# MAGIC %md
# MAGIC ## monthly aggregate

# COMMAND ----------

# MAGIC %sql
# MAGIC drop view if exists ab1_examination;
# MAGIC create temporary view ab1_examination as
# MAGIC select report_type, call_sign, year(date) as date, count(*), avg(SPLIT(AB1, ',')[0]) as avg from group25.PHASE_1_PROCESSED_WEATHER_a
# MAGIC where SPLIT(AB1, ',')[0] != '99999'
# MAGIC group by report_type, call_sign, year(date)

# COMMAND ----------

# MAGIC %sql
# MAGIC select call_sign, count(distinct(report_type)) as test
# MAGIC from aa1_examination
# MAGIC group by call_sign
# MAGIC order by count(distinct(report_type)) desc

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ab1_examination

# COMMAND ----------

# MAGIC %sql
# MAGIC select call_sign, count(distinct(report_type)) as test
# MAGIC from ab1_examination
# MAGIC group by call_sign
# MAGIC order by count(distinct(report_type)) desc

# COMMAND ----------

