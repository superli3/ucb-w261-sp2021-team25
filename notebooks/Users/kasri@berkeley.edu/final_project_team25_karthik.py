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

# MAGIC %md
# MAGIC ## SQL QUERIES FOR DATABASE

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS WEATHER;
# MAGIC 
# MAGIC CREATE TEMPORARY VIEW WEATHER 
# MAGIC USING parquet OPTIONS (path "dbfs:/mnt/mids-w261/datasets_final_project/weather_data/weather*.parquet");
# MAGIC 
# MAGIC SELECT * FROM WEATHER LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS AIRLINES_2015;
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
# MAGIC DROP TABLE IF EXISTS group25.PHASE_1_PROCESSED_AIRLINE;
# MAGIC CREATE TABLE group25.PHASE_1_PROCESSED_AIRLINE AS (
# MAGIC   SELECT *, CONCAT('K', ORIGIN, ' ') AS CALL_SIGN_A FROM AIRLINES_2015
# MAGIC   WHERE ORIGIN LIKE ('%ORD%') OR ORIGIN LIKE ('%ATL%') AND QUARTER=1
# MAGIC ) 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- need information on hourly basis to perform join
# MAGIC -- Airport codes are of 2 kinds, IATA AND ICAO - IATA is the common one (e.g. JFK, DFW)
# MAGIC -- ICAO is for contiguous united states, starts with "K", e.g. KJFK
# MAGIC -- Weather has the ICAO code, whereas airlines have the IATA code, so need to append K when doing the join
# MAGIC 
# MAGIC DROP TABLE IF EXISTS group25.PHASE_1_PROCESSED_WEATHER_AIRLINES;
# MAGIC CREATE TABLE group25.PHASE_1_PROCESSED_WEATHER_AIRLINES AS (
# MAGIC   SELECT * FROM group25.PHASE_1_PROCESSED_AIRLINE AS A
# MAGIC   LEFT JOIN (SELECT * FROM group25.PHASE_1_PROCESSED_WEATHER) AS W
# MAGIC   ON W.CALL_SIGN=A.CALL_SIGN_A
# MAGIC   AND CAST(W.DATE AS DATE)=CAST(A.FL_DATE AS DATE)
# MAGIC   ORDER BY W.CALL_SIGN,  A.OP_CARRIER_FL_NUM, W.DATE ASC
# MAGIC );
# MAGIC 
# MAGIC SELECT * FROM group25.PHASE_1_PROCESSED_WEATHER_AIRLINES LIMIT 10;

# COMMAND ----------

#Load joined table on to dataframe
df = sqlContext.table("group25.phase_1_processed_weather_airlines")
df.count()


# COMMAND ----------

# Downselect the columns of interest
columns = [
  'YEAR'
QUARTER:integer
MONTH:integer
DAY_OF_MONTH:integer
DAY_OF_WEEK:integer
FL_DATE:string
OP_UNIQUE_CARRIER:string
OP_CARRIER_AIRLINE_ID:integer
OP_CARRIER:string
TAIL_NUM:string
OP_CARRIER_FL_NUM:integer
ORIGIN_AIRPORT_ID:integer
ORIGIN_AIRPORT_SEQ_ID:integer
ORIGIN_CITY_MARKET_ID:integer
ORIGIN:string
ORIGIN_CITY_NAME:string
ORIGIN_STATE_ABR:string
ORIGIN_STATE_FIPS:integer
ORIGIN_STATE_NM:string
ORIGIN_WAC:integer
DEST_AIRPORT_ID:integer
DEST_AIRPORT_SEQ_ID:integer
DEST_CITY_MARKET_ID:integer
DEST:string
DEST_CITY_NAME:string
DEST_STATE_ABR:string
DEST_STATE_FIPS:integer
DEST_STATE_NM:string
DEST_WAC:integer
CRS_DEP_TIME:integer
DEP_TIME:integer
DEP_DELAY:double
DEP_DELAY_NEW:double
DEP_DEL15:double
DEP_DELAY_GROUP:integer
DEP_TIME_BLK:string
TAXI_OUT:double
WHEELS_OFF:integer
WHEELS_ON:integer
TAXI_IN:double
CRS_ARR_TIME:integer
ARR_TIME:integer
ARR_DELAY:double
ARR_DELAY_NEW:double
ARR_DEL15:double
ARR_DELAY_GROUP:integer
ARR_TIME_BLK:string
CANCELLED:double
CANCELLATION_CODE:string
DIVERTED:double
CRS_ELAPSED_TIME:double
ACTUAL_ELAPSED_TIME:double
AIR_TIME:double
FLIGHTS:double
DISTANCE:double
DISTANCE_GROUP:integer
CARRIER_DELAY:double
WEATHER_DELAY:double
NAS_DELAY:double
SECURITY_DELAY:double
LATE_AIRCRAFT_DELAY:double
FIRST_DEP_TIME:integer
TOTAL_ADD_GTIME:double
LONGEST_ADD_GTIME:double
DIV_AIRPORT_LANDINGS:integer
DIV_REACHED_DEST:double
DIV_ACTUAL_ELAPSED_TIME:double
DIV_ARR_DELAY:double
DIV_DISTANCE:double
DIV1_AIRPORT:string
DIV1_AIRPORT_ID:integer
DIV1_AIRPORT_SEQ_ID:integer
DIV1_WHEELS_ON:integer
DIV1_TOTAL_GTIME:double
DIV1_LONGEST_GTIME:double
DIV1_WHEELS_OFF:integer
DIV1_TAIL_NUM:string
DIV2_AIRPORT:string
DIV2_AIRPORT_ID:integer
DIV2_AIRPORT_SEQ_ID:integer
DIV2_WHEELS_ON:integer
DIV2_TOTAL_GTIME:double
DIV2_LONGEST_GTIME:double
DIV2_WHEELS_OFF:integer
DIV2_TAIL_NUM:string
DIV3_AIRPORT:string
DIV3_AIRPORT_ID:integer
DIV3_AIRPORT_SEQ_ID:integer
DIV3_WHEELS_ON:integer
DIV3_TOTAL_GTIME:double
DIV3_LONGEST_GTIME:double
DIV3_WHEELS_OFF:string
DIV3_TAIL_NUM:string
DIV4_AIRPORT:string
DIV4_AIRPORT_ID:string
DIV4_AIRPORT_SEQ_ID:string
DIV4_WHEELS_ON:string
DIV4_TOTAL_GTIME:string
DIV4_LONGEST_GTIME:string
DIV4_WHEELS_OFF:string
DIV4_TAIL_NUM:string
DIV5_AIRPORT:string
DIV5_AIRPORT_ID:string
DIV5_AIRPORT_SEQ_ID:string
DIV5_WHEELS_ON:string
DIV5_TOTAL_GTIME:string
DIV5_LONGEST_GTIME:string
DIV5_WHEELS_OFF:string
DIV5_TAIL_NUM:string
CALL_SIGN_A:string
STATION:string
DATE:timestamp
SOURCE:short
LATITUDE:double
LONGITUDE:double
ELEVATION:double
NAME:string
REPORT_TYPE:string
CALL_SIGN:string
QUALITY_CONTROL:string
WND:string
CIG:string
VIS:string
TMP:string
DEW:string
SLP:string
AW1:string
GA1:string
GA2:string
GA3:string
GA4:string
GE1:string
GF1:string
KA1:string
KA2:string
MA1:string
MD1:string
MW1:string
MW2:string
OC1:string
OD1:string
OD2:string
REM:string
EQD:string
AW2:string
AX4:string
GD1:string
AW5:string
GN1:string
AJ1:string
AW3:string
MK1:string
KA4:string
GG3:string
AN1:string
RH1:string
AU5:string
HL1:string
OB1:string
AT8:string
AW7:string
AZ1:string
CH1:string
RH3:string
GK1:string
IB1:string
AX1:string
CT1:string
AK1:string
CN2:string
OE1:string
MW5:string
AO1:string
KA3:string
AA3:string
CR1:string
CF2:string
KB2:string
GM1:string
AT5:string
AY2:string
MW6:string
MG1:string
AH6:string
AU2:string
GD2:string
AW4:string
MF1:string
AA1:string
AH2:string
AH3:string
OE3:string
AT6:string
AL2:string
AL3:string
AX5:string
IB2:string
AI3:string
CV3:string
WA1:string
GH1:string
KF1:string
CU2:string
CT3:string
SA1:string
AU1:string
KD2:string
AI5:string
GO1:string
GD3:string
CG3:string
AI1:string
AL1:string
AW6:string
MW4:string
AX6:string
CV1:string
ME1:string
KC2:string
CN1:string
UA1:string
GD5:string
UG2:string
AT3:string
AT4:string
GJ1:string
MV1:string
GA5:string
CT2:string
CG2:string
ED1:string
AE1:string
CO1:string
KE1:string
KB1:string
AI4:string
MW3:string
KG2:string
AA2:string
AX2:string
AY1:string
RH2:string
OE2:string
CU3:string
MH1:string
AM1:string
AU4:string
GA6:string
KG1:string
AU3:string
AT7:string
KD1:string
GL1:string
IA1:string
GG2:string
OD3:string
UG1:string
CB1:string
AI6:string
CI1:string
CV2:string
AZ2:string
AD1:string
AH1:string
WD1:string
AA4:string
KC1:string
IA2:string
CF3:string
AI2:string
AT1:string
GD4:string
AX3:string
AH4:string
KB3:string
CU1:string
CN4:string
AT2:string
CG1:string
CF1:string
GG1:string
MV2:string
CW1:string
GG4:string
AB1:string
AH5:string
CN3:string
Out[3]: Row(YEAR=2015, QUARTER=1, MON
]

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
# MAGIC 
# MAGIC ## Proposed Features
# MAGIC - Predicted variable (Y) - boolean variable, 0 if no delay, 1 if there is a delay of 15 minutes or more
# MAGIC 
# MAGIC ## Predicted Varaibles
# MAGIC - Temporal
# MAGIC   - Day of the Week
# MAGIC   - Weekday vs weekend
# MAGIC   - Calendar Holidays (might involve brining in another dataset)
# MAGIC - Planning / Traffic
# MAGIC   - How busy is an airport might be on a particular day (how many fights are in bound)
# MAGIC   - Traffic needs to be normalized against itself
# MAGIC   - Diverted flights
# MAGIC   
# MAGIC - Delay propegation
# MAGIC 
# MAGIC   
# MAGIC - Weather *
# MAGIC   - 
# MAGIC - Airline Characteristics?
# MAGIC   - Large vs small airline
# MAGIC   - Delay history (risk score per airport/airline)
# MAGIC   - Centrality of the airport (page rank) - delays at a large airport may propegate to a smaller airport, at source and destination
# MAGIC     - Look at Origin/Destination, get a distinct list of pairs of origin/destination, generate a graph, calculate centrality measure of airports
# MAGIC   - Look at inbound delays, the status of these inbound flights, and the chance leading to an outbound delays 
# MAGIC     
# MAGIC ## 
# MAGIC - Do the readings associated in git repo for flight delays
# MAGIC - 
# MAGIC 
# MAGIC 
# MAGIC ## 
# MAGIC - Plan for the week
# MAGIC   - Do the readings associated in git repo for flight delays 
# MAGIC   - 2 separate notebooks - one for a data processing, one for modeling
# MAGIC   - Start with simple model with easy to grab features - Karthik
# MAGIC   - Start constructing more complex features
# MAGIC     - Weather - Jeff
# MAGIC       - Weather Data
# MAGIC       - Hour variable
# MAGIC     - Planning / Traffic - 
# MAGIC       - How busy is an airport might be on a particular day (how many flights are inbound) - how to do the joi
# MAGIC       - Traffic needs to be normalized against itself
# MAGIC       - Diverted flights
# MAGIC     - Centrality of the Airport (may be extra)

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

