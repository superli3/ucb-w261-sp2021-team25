// Databricks notebook source
// MAGIC %md # w261 Final Project - Clickthrough Rate Prediction

// COMMAND ----------

// MAGIC %md 25   
// MAGIC Justin Trobec, Jeff Li, Sonya Chen, Karthik Srinivasan
// MAGIC Spring 2021, section 5, Team 25

// COMMAND ----------

// MAGIC %md ## Table of Contents
// MAGIC 
// MAGIC * __Section 1__ - Question Formulation
// MAGIC * __Section 2__ - Algorithm Explanation
// MAGIC * __Section 3__ - EDA & Challenges
// MAGIC * __Section 4__ - Algorithm Implementation
// MAGIC * __Section 5__ - Course Concepts

// COMMAND ----------

// MAGIC %md # __Section 1__ - Question Formulation

// COMMAND ----------

// MAGIC %md 

// COMMAND ----------

// MAGIC %md 

// COMMAND ----------

// MAGIC %md # __Section 2__ - Algorithm Explanation

// COMMAND ----------

// MAGIC %md 

// COMMAND ----------

// MAGIC %md 

// COMMAND ----------

// MAGIC %md # __Section 3__ - EDA & Challenges

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC !pip install -U altair

// COMMAND ----------

// MAGIC %python 
// MAGIC import altair as alt
// MAGIC from pyspark.sql import functions as f
// MAGIC from pyspark.sql.functions import col, sum, avg, max, count, countDistinct, weekofyear, to_timestamp, date_format
// MAGIC 
// MAGIC from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
// MAGIC from pyspark.sql import SQLContext
// MAGIC import pandas as pd
// MAGIC import numpy as np
// MAGIC import matplotlib.pyplot as plt
// MAGIC import seaborn as sns
// MAGIC import pyspark
// MAGIC 
// MAGIC from distutils.version import LooseVersion
// MAGIC from pyspark.ml import Pipeline
// MAGIC 
// MAGIC from pandas.tseries.holiday import USFederalHolidayCalendar
// MAGIC import datetime
// MAGIC 
// MAGIC sqlContext = SQLContext(sc)

// COMMAND ----------

// MAGIC 
// MAGIC %python 
// MAGIC 
// MAGIC df_airlines = sqlContext.table("group25.phase_1_processed_airline")

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC SELECT *
// MAGIC FROM group25.phase_1_processed_airline TABLESAMPLE(10000 ROWS)
// MAGIC WHERE DEP_DEL15 IS NOT NULL

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC sampledDf = sqlContext.sql("""SELECT *
// MAGIC                               FROM group25.phase_1_processed_airline TABLESAMPLE(40000 ROWS)
// MAGIC                               WHERE DEP_DEL15 IS NOT NULL""").toPandas()

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC alt.data_transformers.disable_max_rows()
// MAGIC 
// MAGIC def plot_cont_vs_delay(data, col_name):
// MAGIC   dist = data[["DEP_DELAY", "DEP_DEL15", col_name]]
// MAGIC   scatter = alt.Chart(dist).mark_point().encode(
// MAGIC     x="DEP_DELAY:Q",
// MAGIC     y=f'{col_name}:Q',
// MAGIC     tooltip=[col_name, 'DEP_DELAY', 'DEP_DEL15']
// MAGIC   ).properties(width=450).interactive()
// MAGIC 
// MAGIC   box = alt.Chart(dist).mark_boxplot().encode(
// MAGIC     x="DEP_DEL15:O",
// MAGIC     y=f'{col_name}:Q'
// MAGIC   ).properties(width=450)
// MAGIC   both = alt.hconcat(scatter, box)
// MAGIC   both.title = f'{col_name} vs. Delay'
// MAGIC   return both
// MAGIC 
// MAGIC def plot_cat_vs_delay(data, col_name):
// MAGIC   dist = data[["DEP_DELAY", "DEP_DEL15", col_name]]
// MAGIC   scatter = alt.Chart(dist).mark_boxplot().encode(
// MAGIC     y="DEP_DELAY:Q",
// MAGIC     x=f'{col_name}:N',
// MAGIC     tooltip=[col_name, 'DEP_DELAY', 'DEP_DEL15']
// MAGIC   ).properties(width=450).interactive()
// MAGIC 
// MAGIC   box = alt.Chart(dist).mark_bar().encode(
// MAGIC     x=f'{col_name}:N',
// MAGIC     y="count():Q",
// MAGIC     color="DEP_DEL15:O"
// MAGIC   ).properties(width=450).properties(width=300)
// MAGIC   both = alt.hconcat(scatter, box)
// MAGIC   both.title = f'{col_name} vs. Delay'
// MAGIC   return both
// MAGIC   

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC plot_cont_vs_delay(sampledDf, "DISTANCE")

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC plot_cat_vs_delay(sampledDf, "ORIGIN")

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC plot_cat_vs_delay(sampledDf, "DEST")

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC plot_cat_vs_delay(sampledDf, "DAY_OF_WEEK")

// COMMAND ----------

// MAGIC %md ### Load Spark GraphX libraries.

// COMMAND ----------

import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, concat, lit}

import org.apache.spark.graphx.GraphLoader
import org.apache.spark.graphx.GraphOps

// COMMAND ----------

spark

// COMMAND ----------

// MAGIC %fs ls dbfs:/mnt/mids-w261/datasets_final_project/

// COMMAND ----------

// MAGIC %fs ls dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data

// COMMAND ----------

// starting out with 2015 data only as a test set...
val airlines2015 = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data/2015.parquet")
display(airlines2015.sample(false, 0.00001))

// COMMAND ----------

println(s"Count of flights, 2015: ${airlines2015.count()}")

// COMMAND ----------

// starting out with 2015 data only as a test set...
val airlinesFull = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/datasets_final_project/parquet_airlines_data/201*.parquet")
display(airlinesFull.sample(false, 0.00001))

// COMMAND ----------

println(s"Count of flights, Full: ${airlinesFull.count()}")

// COMMAND ----------

 def code(str: String) = str.foldLeft(0L) { case (code, c) => 31*code + c }

// COMMAND ----------

def buildVerticies(data: DataFrame): RDD[(VertexId, String)] = {
  return data
    .select("ORIGIN")
    .union(airlines2015.select("DEST"))
    .distinct()
    .rdd
    .map(x => (code(x.getString(0)), x.getString(0)))
}

val airportVerticies2015 = buildVerticies(airlines2015)
airportVerticies.take(1)

// COMMAND ----------

val airportVerticiesFull = buildVerticies(airlinesFull)

// COMMAND ----------

println(s"Count of airports, 2015: ${airportVerticies2015.count()}")
println(s"Count of airports, full: ${airportVerticiesFull.count()}")

// COMMAND ----------

def buildEdges(data: DataFrame): RDD[Edge[Long]] = {
  return data 
    .select(col("ORIGIN"), col("DEST"))
    .rdd
    .map(row => Edge(code(row.getString(0)), code(row.getString(1)), 1))
}

val airportEdges2015 = buildEdges(airlines2015)
val airportEdgesFull = buildEdges(airlinesFull)

airportEdges2015.take(1)
airportEdgesFull.take(1)

// COMMAND ----------

def buildGraph(data: DataFrame): Graph[String, Long] = {
  return Graph(
    buildVerticies(data),
    buildEdges(data)
  )
}

val airportGraph2015 = buildGraph(airlines2015)
val airportGraphFull = buildGraph(airlinesFull)

airportGraphFull.cache()

// COMMAND ----------

def graphStats(g: Graph[String, Long]): Unit = println(s"Graph vertices: ${g.numVertices}, Graph Edges: ${g.numEdges}")
graphStats(airportGraphFull)

// COMMAND ----------

val ranks = airportGraphFull.pageRank(0.0001).vertices
ranks
  .join(airportVerticiesFull)
  .sortBy(_._2._1, ascending=false) // sort by the rank
  .take(10) // get the top 10
  .foreach(x => println(s"Airport: ${x._2._2}\tPageRank: ${x._2._1}"))

// COMMAND ----------

ranks
  .join(airportVerticiesFull)
  .map(port => (port._2._2, port._2._1))
  .

// COMMAND ----------

val airlineAirports = airlines2015.select(concat(col("ORIGIN"), lit(":"), col("OP_CARRIER")).as("ORIGIN"), concat(col("DEST"), lit(":"), col("OP_CARRIER")).as("DEST"))

val airlineAirportVertices: RDD[(VertexId, String)] = airlineAirports
  .select("ORIGIN")
  .union(airlineAirports.select("DEST"))
  .distinct()
  .rdd
  .map(x => (code(x.getString(0)), x.getString(0)))

val airlineAirportEdges:RDD[Edge[Long]] = airlineAirports
  .select(col("ORIGIN"), col("DEST"))
  .rdd
  .map(row => Edge(code(row.getString(0)), code(row.getString(1)), 1))

val airlineAirportsGraph = Graph(airlineAirportVertices, airlineAirportEdges)
graphStats(airlineAirportsGraph)

// COMMAND ----------

:55
df_lag = df_temp.withColumn('prev_delay', lag(df_temp['DEP_DELAY']).over(Window.partitionBy("TAIL_NUM").orderBy("FL_DATE", "DEP_TIME"))) \
                .withColumn('prev_dep_time', lag(df_temp['DEP_TIME']).over(Window.partitionBy("TAIL_NUM").orderBy("FL_DATE", "DEP_TIME"))) \
                .na.fill(value=0,subset=["prev_delay"])
val airlineAirportsPageRanks = airlineAirportsGraph.pageRank(0.0001).vertices
airlineAirportsPageRanks
  .join(airlineAirportVertices)
  .sortBy(_._2._1, ascending=false) // sort by the rank
  .take(10) // get the top 10
  .foreach(x => println(s"Airport: ${x._2._2}\tPageRank: ${x._2._1}"))

// COMMAND ----------

val airlineAirportsPageRanks = airlineAirportsGraph.pageRank(0.0001).vertices
val ranked = airlineAirportsPageRanks
  .join(airlineAirportVertices)
  .sortBy(_._2._1, ascending=false) 
ranked

// COMMAND ----------

def originDestPageRanks(df: DataFrame): 

// COMMAND ----------

// MAGIC %md # __Section 4__ - Algorithm Implementation

// COMMAND ----------

// MAGIC %md 

// COMMAND ----------

// MAGIC %md 

// COMMAND ----------

// MAGIC %md # __Section 5__ - Course Concepts

// COMMAND ----------

// MAGIC %md 

// COMMAND ----------

// MAGIC %md 

// COMMAND ----------

