// Databricks notebook source
// MAGIC %md 
// MAGIC 
// MAGIC W261 Final Project
// MAGIC ------------------
// MAGIC ## Graph Centrality for Airline Delay Prediction
// MAGIC   
// MAGIC #### Justin Trobec, Jeff Li, Sonya Chen, Karthik Srinivasan
// MAGIC #### Spring 2021, Section 5
// MAGIC #### Group 25 

// COMMAND ----------

// MAGIC %md ## Table of Contents
// MAGIC 
// MAGIC * __Section 1__ - Algorithm Explanation
// MAGIC * __Section 2__ - EDA & Challenges
// MAGIC * __Section 3__ - Algorithm Implementation
// MAGIC * __Section 4__ - Course Concepts

// COMMAND ----------

// MAGIC %md # __Section 1__ - Algorithm Explanation

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC Airline flight data forms a natural graph in some obvious ways. Airports are clearly connected by flights, so we can consider airports as verticies and flights between them as edges. We could make the vertices more complex by addtionally segmenting them by airline or time frame. We could also invert the graph, treat vertices as edges that connect an incoming flight to an outgoing flight. These approaches can be mixed in many ways, some of which may be computationally prohibitive.
// MAGIC 
// MAGIC In this notebook, we will experiment with different ways to approach using the PageRank centrality metric to improve flight delay predictions.

// COMMAND ----------

// MAGIC %md # __Section 2__ - EDA & Challenges

// COMMAND ----------

// MAGIC %md ### Load Spark, GraphX, and other libraries.

// COMMAND ----------

import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.graphx.lib._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, concat, lit}

import org.apache.spark.graphx.GraphLoader
import org.apache.spark.graphx.GraphOps

def graphStats(g: Graph[String, Long]): Unit = println(s"Graph vertices: ${g.numVertices}, Graph Edges: ${g.numEdges}")

/**
* code takes a string and converts it to a simple Long hashcode. Used to create IDs for vertices.
*/
def code(str: String) = str.foldLeft(0L) { case (code, c) => 31*code + c }

spark

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC !pip install -U altair vega_datasets

// COMMAND ----------

// MAGIC %python
// MAGIC import altair as alt
// MAGIC import pandas as pd
// MAGIC import numpy as np
// MAGIC import matplotlib.pyplot as plt
// MAGIC import seaborn as sns
// MAGIC import pyspark
// MAGIC 
// MAGIC from pyspark.sql import functions as f
// MAGIC from pyspark.sql.functions import col, sum, avg, max, count, countDistinct, weekofyear, to_timestamp, date_format
// MAGIC 
// MAGIC from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
// MAGIC from pyspark.sql import SQLContext
// MAGIC from vega_datasets import data
// MAGIC 
// MAGIC from distutils.version import LooseVersion
// MAGIC from pyspark.ml import Pipeline
// MAGIC 
// MAGIC from pandas.tseries.holiday import USFederalHolidayCalendar
// MAGIC import datetime
// MAGIC 
// MAGIC sqlContext = SQLContext(sc)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC In order to prevent any possibility of leakage, we are going to limit our training set to data through EOY 2017. Ideally, for every row we would compute relevant PRs two hours ahead of time. This would be computationally prohibitive and impractical for our project, but with unlimited time and resources, we could have investigated the possibility of doing streaming updates to PageRank (e.g. J. Riedy, "Updating PageRank for Streaming Graphs," 2016 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), Chicago, IL, USA, 2016, pp. 877-884, doi: 10.1109/IPDPSW.2016.22. https://ieeexplore.ieee.org/document/7529953).

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### Functions for computing PageRank

// COMMAND ----------

/**
* Given a dataframe with ORIGIN and DEST columns, build the set
* of vertices.
*
* Note that ORIGIN & DELAY are just column names, and there's no
* reason those columns need to be airports...later we will use
* a more complex key and just stuff it all into columns called
* ORIGIN and DEST.
*/
def buildVertices(data: DataFrame): RDD[(VertexId, String)] = {
  return data
    .select("ORIGIN")
    .union(data.select("DEST"))
    .distinct()
    .rdd
    .map(x => (code(x.getString(0)), x.getString(0)))
}
/**
* Given a dataframe with ORIGIN and DEST columns, build the set
* of edges.
*/
def buildEdges(data: DataFrame): RDD[Edge[Long]] = {
  return data 
    .select(col("ORIGIN"), col("DEST"))
    .rdd
    .map(row => Edge(code(row.getString(0)), code(row.getString(1)), 1))
}
/**
* Given a dataframe with ORIGIN and DEST columns, build the graph.
* Returns a tuple with (vertices, edges, graph)
*/
def buildGraph(data: DataFrame): (RDD[(VertexId, String)], RDD[Edge[Long]], Graph[String, Long]) = {
  val v = buildVertices(data)
  val e = buildEdges(data)
  val g = Graph(v, e)
  
  return (v, e, g)
}
/**
* Given a graph and its vertices, get the pagerank joined with the vertices.
*/
def getPageRanks(g: Graph[String, Long], v: RDD[(VertexId, String)], conv: Double = 0.0001) = {
  val ranks = g.pageRank(conv).vertices
  ranks
    .join(v)
    .sortBy(_._2._1, ascending=false) // sort by the rank
}

/**
* Given a dataset, create a graph of ORIGIN to DEST edges, and compute pagerank for each of them.
*/
def runPageRank(dataSet: DataFrame, conv: Double = 0.0001): RDD[((org.apache.spark.graphx.VertexId, (Double, String)))] = {
  val (v, e, g) = buildGraph(dataSet)
  g.cache()
  graphStats(g)

  val ranks = getPageRanks(g, v, conv)
  ranks
    .take(10) // get the top 10
    .foreach(x => println(s"Vertex: ${x._2._2}\tPageRank: ${x._2._1}"))
  
  return ranks
}


// COMMAND ----------



// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### Create Datasets For PR EDA
// MAGIC 
// MAGIC We want to see if any of these PageRanks appear to correlate strongly with delays at a given level of granularity.

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC DROP TABLE IF EXISTS group25.origin_delay_fraction;
// MAGIC CREATE TABLE group25.origin_delay_fraction AS 
// MAGIC (SELECT ORIGIN, count(*) as FLIGHT_COUNT, cast(sum(DEP_DEL15) as float) / count(*) as fraction_delayed
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2019
// MAGIC GROUP BY 1);
// MAGIC 
// MAGIC DROP TABLE IF EXISTS group25.origin_carrier_delay_fraction;
// MAGIC CREATE TABLE group25.origin_carrier_delay_fraction AS 
// MAGIC (SELECT ORIGIN, OP_CARRIER, count(*) as FLIGHT_COUNT, cast(sum(DEP_DEL15) as float) / count(*) as fraction_delayed
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2019
// MAGIC GROUP BY 1, 2);
// MAGIC 
// MAGIC DROP TABLE IF EXISTS group25.origin_carrier_dow_delay_fraction;
// MAGIC CREATE TABLE group25.origin_carrier_dow_delay_fraction AS 
// MAGIC (SELECT ORIGIN, OP_CARRIER, DAY_OF_WEEK, count(*) as FLIGHT_COUNT, cast(sum(DEP_DEL15) as float) / count(*) as fraction_delayed
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2019
// MAGIC GROUP BY 1, 2, 3);

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC DROP TABLE IF EXISTS group25.dest_delay_fraction;
// MAGIC CREATE TABLE group25.dest_delay_fraction AS 
// MAGIC (SELECT dest, count(*) as FLIGHT_COUNT, cast(sum(DEP_DEL15) as float) / count(*) as fraction_delayed
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2019
// MAGIC GROUP BY 1);
// MAGIC 
// MAGIC DROP TABLE IF EXISTS group25.dest_carrier_delay_fraction;
// MAGIC CREATE TABLE group25.dest_carrier_delay_fraction AS 
// MAGIC (SELECT dest, OP_CARRIER, count(*) as FLIGHT_COUNT, cast(sum(DEP_DEL15) as float) / count(*) as fraction_delayed
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2019
// MAGIC GROUP BY 1, 2);
// MAGIC 
// MAGIC DROP TABLE IF EXISTS group25.dest_carrier_dow_delay_fraction;
// MAGIC CREATE TABLE group25.dest_carrier_dow_delay_fraction AS 
// MAGIC (SELECT dest, OP_CARRIER, DAY_OF_WEEK, count(*) as FLIGHT_COUNT, cast(sum(DEP_DEL15) as float) / count(*) as fraction_delayed
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2019
// MAGIC GROUP BY 1, 2, 3);

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC DROP TABLE IF EXISTS group25.flight_delay_fraction;
// MAGIC CREATE TABLE group25.flight_delay_fraction AS 
// MAGIC (SELECT origin, dest, count(*) as FLIGHT_COUNT, cast(sum(DEP_DEL15) as float) / count(*) as fraction_delayed
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2019
// MAGIC GROUP BY 1, 2);
// MAGIC 
// MAGIC DROP TABLE IF EXISTS group25.flight_dow_delay_fraction;
// MAGIC CREATE TABLE group25.flight_dow_delay_fraction AS 
// MAGIC (SELECT origin, dest, day_of_week, count(*) as FLIGHT_COUNT, cast(sum(DEP_DEL15) as float) / count(*) as fraction_delayed
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2019
// MAGIC GROUP BY 1, 2, 3);
// MAGIC 
// MAGIC DROP TABLE IF EXISTS group25.flight_dow_hour_delay_fraction;
// MAGIC CREATE TABLE group25.flight_dow_hour_delay_fraction AS 
// MAGIC (SELECT origin, dest, day_of_week, dep_utc_hour, count(*) as FLIGHT_COUNT, cast(sum(DEP_DEL15) as float) / count(*) as fraction_delayed
// MAGIC FROM group25.airlines_utc_main 
// MAGIC WHERE YEAR < 2019
// MAGIC GROUP BY 1, 2, 3, 4);

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC origin_delay_fraction = spark.sql("""SELECT fraction_delayed
// MAGIC FROM group25.origin_delay_fraction""").toPandas()
// MAGIC 
// MAGIC origin_carrier_delay_fraction = spark.sql("""SELECT fraction_delayed
// MAGIC FROM group25.origin_carrier_delay_fraction""").toPandas()
// MAGIC 
// MAGIC origin_carrier_dow_delay_fraction = spark.sql("""SELECT fraction_delayed
// MAGIC FROM group25.origin_carrier_dow_delay_fraction""").toPandas()
// MAGIC 
// MAGIC dest_delay_fraction = spark.sql("""SELECT fraction_delayed
// MAGIC FROM group25.dest_delay_fraction""").toPandas()
// MAGIC 
// MAGIC dest_carrier_delay_fraction = spark.sql("""SELECT fraction_delayed
// MAGIC FROM group25.dest_carrier_delay_fraction""").toPandas()
// MAGIC 
// MAGIC dest_carrier_dow_delay_fraction = spark.sql("""SELECT fraction_delayed
// MAGIC FROM group25.dest_carrier_dow_delay_fraction""").toPandas()

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC all_delay_fraction = spark.sql("""
// MAGIC   SELECT 'Airport' as gran, 'Origin' as origin_dest, fraction_delayed FROM group25.origin_delay_fraction
// MAGIC   UNION
// MAGIC   SELECT 'Airport/Airline' as gran, 'Origin' as origin_dest, fraction_delayed FROM group25.origin_carrier_delay_fraction
// MAGIC   UNION
// MAGIC   SELECT 'Airport/Airline/DOW' as gran, 'Origin' as origin_dest, fraction_delayed FROM group25.origin_carrier_dow_delay_fraction
// MAGIC   UNION
// MAGIC   SELECT 'Airport' as gran, 'Destination' as origin_dest, fraction_delayed FROM group25.dest_delay_fraction
// MAGIC   UNION
// MAGIC   SELECT 'Airport/Airline' as gran, 'Destination' as origin_dest, fraction_delayed FROM group25.dest_carrier_delay_fraction
// MAGIC   UNION
// MAGIC   SELECT 'Airport/Airline/DOW' as gran, 'Destination' as origin_dest, fraction_delayed FROM group25.dest_carrier_dow_delay_fraction
// MAGIC """).toPandas()

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC fig, ax = plt.subplots(figsize=(20, 7))
// MAGIC sns.violinplot(x='gran', y='fraction_delayed', hue='origin_dest', split=True, palette='pastel', data=all_delay_fraction, scale_hue=False, order=['Airport', 'Airport/Airline', 'Airport/Airline/DOW'], axes=ax)
// MAGIC ax.set(title="Distribution of Fraction of Flights Delayed", xlabel='Granularity', ylabel="Fraction Delayed")

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC flight_delay_fraction = spark.sql("SELECT fraction_delayed FROM group25.flight_delay_fraction").toPandas()
// MAGIC flight_dow_delay_fraction = spark.sql("SELECT fraction_delayed FROM group25.flight_dow_delay_fraction").toPandas()
// MAGIC flight_dow_hour_delay_fraction = spark.sql("SELECT fraction_delayed FROM group25.flight_dow_hour_delay_fraction").toPandas()
// MAGIC 
// MAGIC fig, ax = plt.subplots(1, 3, figsize=(20,7))
// MAGIC sns.distplot(flight_delay_fraction, ax=ax[0])
// MAGIC sns.distplot(flight_dow_delay_fraction, ax=ax[1])
// MAGIC sns.distplot(flight_dow_hour_delay_fraction, ax=ax[2])

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC Our hope is that we can find some relationship between PR and the fraction of flights delayed at a given granularity, that would indicate the PR might be a good predictor for whether or not a flight is delayed. The above charts show how the distribution of fraction delayed spreads out with higher granularity, which is not surprising. They also show that the distributions are similar but, but slightly shifted between the airport being a origin or a destination.

// COMMAND ----------

// MAGIC %md # __Section 3__ - PageRank Variations

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### 1. Per Airport PageRank

// COMMAND ----------

val airlinesTrain = spark.sql("SELECT ORIGIN, DEST FROM group25.airlines_main WHERE YEAR < 2018")

// COMMAND ----------

val (airportTrainVertices, airportTrainEdges, airportTrainGraph) = buildGraph(airlinesTrain)
airportTrainGraph.cache()
graphStats(airportTrainGraph)

// COMMAND ----------

val ranks = getPageRanks(airportTrainGraph, airportTrainVertices)
ranks
  .take(10) // get the top 10
  .foreach(x => println(s"Airport: ${x._2._2}\tPageRank: ${x._2._1}"))

// COMMAND ----------

val airportPrTrain = ranks
  .map(port => (port._2._2, port._2._1))
val prTrainDf = spark.createDataFrame(airportPrTrain).toDF("airport", "pageRank")

// COMMAND ----------

prTrainDf
  .write
  .mode("overwrite")
  .saveAsTable("group25.AIRPORTS_PR_TRAIN")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Validate expected count of rows...

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC SELECT count(*)
// MAGIC FROM   group25.AIRPORTS_PR_TRAIN

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Plot PR versus Percentage of Delayed Flights Per Airport

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC origin_airports_pr = spark.sql("""
// MAGIC   SELECT pr.airport, pr.pageRank as pageRank, ap.fraction_delayed
// MAGIC   FROM group25.origin_delay_fraction ap left join group25.AIRPORTS_PR_TRAIN pr on ap.origin = pr.airport
// MAGIC   WHERE ap.fraction_delayed < 1.0
// MAGIC """).toPandas()
// MAGIC 
// MAGIC dest_airports_pr = spark.sql("""
// MAGIC   SELECT pr.airport, pr.pageRank as pageRank, ap.fraction_delayed
// MAGIC   FROM group25.dest_delay_fraction ap left join group25.AIRPORTS_PR_TRAIN pr on ap.dest = pr.airport
// MAGIC   WHERE ap.fraction_delayed < 1.0
// MAGIC """).toPandas()

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC def plotPrVsFd(orig_data, dest_data):
// MAGIC   fig, axes = plt.subplots(1, 2, figsize=(20, 7))
// MAGIC   sns.regplot(x='pageRank', y='fraction_delayed', data=orig_data, ax=axes[0])
// MAGIC   sns.regplot(x='pageRank', y='fraction_delayed', data=dest_data, ax=axes[1])
// MAGIC   plt.suptitle('PageRank V. Fraction Delayed (ORIGIN/DEST)')
// MAGIC   
// MAGIC plotPrVsFd(origin_airports_pr, dest_airports_pr)

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC od_pr = spark.sql("""
// MAGIC   SELECT pr.pageRank, ap.dep_del15, "Origin" as o_or_d
// MAGIC   FROM group25.airlines_main ap INNER JOIN group25.AIRPORTS_PR_TRAIN pr
// MAGIC     ON pr.airport = ap.ORIGIN
// MAGIC   UNION
// MAGIC   SELECT pr.pageRank, ap.dep_del15, "Destination" as o_or_d
// MAGIC   FROM group25.airlines_main ap INNER JOIN group25.AIRPORTS_PR_TRAIN pr
// MAGIC     ON pr.airport = ap.DEST
// MAGIC """).toPandas()
// MAGIC 
// MAGIC fig, ax = plt.subplots(figsize=(15, 7))
// MAGIC sns.boxplot(y='pageRank', x='o_or_d', hue='dep_del15', ax=ax, data=od_pr)
// MAGIC ax.set(title="PageRank vs Departure Delays Over 15 Minutes", ylabel="PageRank", xlabel="Origin/Destination")

// COMMAND ----------

// MAGIC %python 
// MAGIC 
// MAGIC pr_dat = spark.sql("SELECT * FROM group25.airports_dat_txt2 ap INNER JOIN group25.AIRPORTS_PR_TRAIN pr on pr.airport = ap.IATA").toPandas()
// MAGIC cn_dat = spark.sql("SELECT * FROM group25.airports_dat_txt2 ap INNER JOIN (SELECT ORIGIN, count(*) as fl_count FROM group25.airlines_main WHERE year < 2018 GROUP BY 1) cn on cn.ORIGIN = ap.IATA").toPandas()
// MAGIC 
// MAGIC def map_background():
// MAGIC   airports = data.airports.url
// MAGIC   states = alt.topo_feature(data.us_10m.url, feature='states')
// MAGIC   # US states background
// MAGIC   return alt.Chart(states).mark_geoshape(
// MAGIC         fill='lightgray',
// MAGIC         stroke='white'
// MAGIC     ).properties(
// MAGIC         width=1000,
// MAGIC         height=600
// MAGIC     ).project('albersUsa')
// MAGIC 
// MAGIC # airport positions on background
// MAGIC points_pr = alt.Chart(pr_dat).mark_circle().encode(
// MAGIC     longitude='Longitude:Q',
// MAGIC     latitude='Latitude:Q',
// MAGIC     size=alt.Size('pageRank:Q', title='Page Rank'),
// MAGIC     color=alt.value('steelblue'),
// MAGIC     tooltip=['Name:N','pageRank:Q']
// MAGIC ).properties(
// MAGIC     title='Page Rank of Airports in the US',
// MAGIC     width=1000,
// MAGIC     height=600
// MAGIC )
// MAGIC 
// MAGIC # airport positions on background
// MAGIC points_cn = alt.Chart(cn_dat).mark_circle().encode(
// MAGIC     longitude='Longitude:Q',
// MAGIC     latitude='Latitude:Q',
// MAGIC     size=alt.Size('fl_count:Q', title='Count of Flights'),
// MAGIC     color=alt.value('steelblue'),
// MAGIC     tooltip=['Name:N','fl_count:Q']
// MAGIC ).properties(
// MAGIC     title='Count of Flights for Airports in the US, 2015-2017',
// MAGIC     width=1000,
// MAGIC     height=600
// MAGIC )
// MAGIC 
// MAGIC pr = map_background() + points_pr
// MAGIC cn = map_background() + points_cn
// MAGIC 
// MAGIC pr

// COMMAND ----------

// MAGIC %python
// MAGIC cn

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC cn_pr_dat = spark.sql("SELECT * FROM group25.airports_dat_txt2 ap INNER JOIN group25.AIRPORTS_PR_TRAIN pr on pr.airport = ap.IATA INNER JOIN  (SELECT ORIGIN, count(*) as fl_count FROM group25.airlines_main WHERE year < 2018 GROUP BY 1) cn on cn.ORIGIN = pr.airport").toPandas()
// MAGIC 
// MAGIC fig, ax = plt.subplots(figsize=(15, 7))
// MAGIC sns.scatterplot(x='fl_count', y='pageRank', ax=ax, data=cn_pr_dat)
// MAGIC ax.set(title='Airport Level PageRank vs Count of Flights', xlabel="Flight Count", ylabel="PageRank")

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC cn_pr_dat_h = spark.sql("""
// MAGIC   SELECT * 
// MAGIC   FROM group25.airports_dat_txt2 ap 
// MAGIC     INNER JOIN group25.AIRPORTS_AIRLINE_DOW_PR_TRAIN pr on pr.airport = ap.IATA 
// MAGIC     INNER JOIN  (SELECT ORIGIN, OP_CARRIER, DAY_OF_WEEK, count(*) as fl_count FROM group25.airlines_main WHERE year < 2018 GROUP BY 1, 2, 3) cn on cn.ORIGIN = pr.airport AND cn.DAY_OF_WEEK = pr.day_of_week AND cn.OP_CARRIER = pr.airline
// MAGIC   """).toPandas()
// MAGIC 
// MAGIC weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
// MAGIC cn_pr_dat_h['weekday_name'] = cn_pr_dat_h.apply(lambda row: weekday_names[int(row.day_of_week)-1], axis=1)
// MAGIC 
// MAGIC fig, ax = plt.subplots(figsize=(15, 7))
// MAGIC sns.scatterplot(x='fl_count', y='pageRank', hue='weekday_name', hue_order=weekday_names, ax=ax, data=cn_pr_dat_h)
// MAGIC ax.legend(title = 'Day of Week')
// MAGIC ax.set(title='Airport/Airline/Day of Week Level PageRank vs Count of Flights', xlabel="Flight Count", ylabel="PageRank")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### 2. Per Airport/Airline PageRank

// COMMAND ----------

val airportAirlineTrain = spark.sql("SELECT CONCAT(ORIGIN, ':', OP_CARRIER) AS ORIGIN, CONCAT(DEST, ':', OP_CARRIER) AS DEST FROM group25.airlines_main WHERE YEAR < 2018")

// COMMAND ----------

spark.createDataFrame(
  runPageRank(airportAirlineTrain).map(port => {
    val aa = port._2._2.split(':')
    (aa(0), aa(1), port._2._1)
  }))
  .toDF("airport", "airline", "pageRank")
  .write.mode("overwrite")
  .saveAsTable("group25.AIRPORTS_AIRLINE_PR_TRAIN")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Validate expected count of rows...
// MAGIC 
// MAGIC The SQL below computes expected rows for this level of granularity. Check output above to confirm it matches.

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC SELECT COUNT(DISTINCT AIRPORT_AIRLINE)
// MAGIC FROM
// MAGIC (SELECT CONCAT(ORIGIN, ':', OP_CARRIER) AS AIRPORT_AIRLINE
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2018
// MAGIC UNION
// MAGIC SELECT CONCAT(DEST, ':', OP_CARRIER) as AIRPORT_AIRLINE
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2018)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Plot PR versus Percentage of Delayed Flights Per Airport/Airline

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC origin_airport_airline_pr = spark.sql("""
// MAGIC   SELECT pr.airport, pr.airline, pr.pageRank as pageRank, ap.fraction_delayed
// MAGIC   FROM group25.origin_carrier_delay_fraction ap left join group25.AIRPORTS_AIRLINE_PR_TRAIN pr on ap.origin = pr.airport AND ap.op_carrier = pr.airline
// MAGIC   WHERE ap.fraction_delayed < 1.0
// MAGIC """).toPandas()
// MAGIC 
// MAGIC dest_airport_airline_pr = spark.sql("""
// MAGIC   SELECT pr.airport, pr.airline, pr.pageRank as pageRank, ap.fraction_delayed
// MAGIC   FROM group25.dest_carrier_delay_fraction ap left join group25.AIRPORTS_AIRLINE_PR_TRAIN pr on ap.dest = pr.airport AND ap.op_carrier = pr.airline
// MAGIC   WHERE ap.fraction_delayed < 1.0
// MAGIC """).toPandas()

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC plotPrVsFd(origin_airport_airline_pr, dest_airport_airline_pr)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### 3. Per Airport/Airline/Day Of Week PageRank

// COMMAND ----------

val airportAirlineDowTrain = spark.sql("SELECT CONCAT(ORIGIN, ':', OP_CARRIER, ':', DAY_OF_WEEK) AS ORIGIN, CONCAT(DEST, ':', OP_CARRIER, ':', DAY_OF_WEEK) AS DEST FROM group25.airlines_main WHERE YEAR < 2018")

// COMMAND ----------

spark.createDataFrame(
  runPageRank(airportAirlineDowTrain).map(port => {
    val aa = port._2._2.split(':')
    (aa(0), aa(1), aa(2), port._2._1)
  }))
  .toDF("airport", "airline", "day_of_week", "pageRank")
  .write.mode("overwrite")
  .saveAsTable("group25.AIRPORTS_AIRLINE_DOW_PR_TRAIN")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Validate expected count of rows...

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC SELECT COUNT(DISTINCT AIRPORT_AIRLINE)
// MAGIC FROM
// MAGIC (SELECT CONCAT(ORIGIN, ':', OP_CARRIER, ':', DAY_OF_WEEK) AS AIRPORT_AIRLINE
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2018
// MAGIC UNION
// MAGIC SELECT CONCAT(DEST, ':', OP_CARRIER, ':', DAY_OF_WEEK) as AIRPORT_AIRLINE
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2018)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Plot PR versus Percentage of Delayed Flights Per Airport/Airline/Day of Week

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC origin_airport_airline_dow_pr = spark.sql("""
// MAGIC   SELECT pr.airport, pr.airline, pr.day_of_week, pr.pageRank as pageRank, ap.fraction_delayed
// MAGIC   FROM group25.origin_carrier_dow_delay_fraction ap left join group25.AIRPORTS_AIRLINE_DOW_PR_TRAIN pr on ap.origin = pr.airport AND ap.op_carrier = pr.airline AND ap.day_of_week = pr.day_of_week
// MAGIC   WHERE ap.fraction_delayed < 1.0
// MAGIC """).toPandas()
// MAGIC 
// MAGIC dest_airport_airline_dow_pr = spark.sql("""
// MAGIC   SELECT pr.airport, pr.airline, pr.day_of_week, pr.pageRank as pageRank, ap.fraction_delayed
// MAGIC   FROM group25.dest_carrier_dow_delay_fraction ap left join group25.AIRPORTS_AIRLINE_DOW_PR_TRAIN pr on ap.dest = pr.airport AND ap.op_carrier = pr.airline AND ap.day_of_week = pr.day_of_week
// MAGIC   WHERE ap.fraction_delayed < 1.0
// MAGIC """).toPandas()

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC plotPrVsFd(origin_airport_airline_dow_pr, dest_airport_airline_dow_pr)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### 4. Delays Only, Per Airport/Airline/Day of Week PageRank

// COMMAND ----------

val airportAirlineDowDelaysTrain = spark.sql("""SELECT CONCAT(ORIGIN, ':', OP_CARRIER, ':', DAY_OF_WEEK) AS ORIGIN, CONCAT(DEST, ':', OP_CARRIER, ':',  DAY_OF_WEEK) AS DEST 
  FROM group25.airlines_main WHERE YEAR < 2018 AND DEP_DEL15 = 1""")

// COMMAND ----------

spark.createDataFrame(
  runPageRank(airportAirlineDowDelaysTrain).map(port => {
    val aa = port._2._2.split(':')
    (aa(0), aa(1), aa(2), port._2._1)
  }))
  .toDF("airport", "airline", "day_of_week", "pageRank")
  .write.mode("overwrite")
  .saveAsTable("group25.AIRPORTS_AIRLINE_DOW_DELAYS_PR_TRAIN")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Validate expected count of rows...

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC SELECT COUNT(DISTINCT AIRPORT_AIRLINE)
// MAGIC FROM
// MAGIC (SELECT CONCAT(ORIGIN, ':', OP_CARRIER, ':', DAY_OF_WEEK) AS AIRPORT_AIRLINE
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2018 AND DEP_DEL15 = 1
// MAGIC UNION
// MAGIC SELECT CONCAT(DEST, ':', OP_CARRIER, ':', DAY_OF_WEEK) as AIRPORT_AIRLINE
// MAGIC FROM group25.airlines_main 
// MAGIC WHERE YEAR < 2018 AND DEP_DEL15 = 1)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Plot PR versus Percentage of Delayed Flights Per Airport/Airline/Day of Week, Delays Only

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC origin_airport_airline_dow_delays_pr = spark.sql("""
// MAGIC   SELECT pr.airport, pr.airline, pr.day_of_week, pr.pageRank as pageRank, ap.fraction_delayed
// MAGIC   FROM group25.origin_carrier_dow_delay_fraction ap left join group25.AIRPORTS_AIRLINE_DOW_DELAYS_PR_TRAIN pr on ap.origin = pr.airport AND ap.op_carrier = pr.airline AND ap.day_of_week = pr.day_of_week
// MAGIC   WHERE ap.fraction_delayed < 1.0
// MAGIC """).toPandas()
// MAGIC 
// MAGIC dest_airport_airline_dow_delays_pr = spark.sql("""
// MAGIC   SELECT pr.airport, pr.airline, pr.day_of_week, pr.pageRank as pageRank, ap.fraction_delayed
// MAGIC   FROM group25.dest_carrier_dow_delay_fraction ap left join group25.AIRPORTS_AIRLINE_DOW_DELAYS_PR_TRAIN pr on ap.dest = pr.airport AND ap.op_carrier = pr.airline AND ap.day_of_week = pr.day_of_week
// MAGIC   WHERE ap.fraction_delayed < 1.0
// MAGIC """).toPandas()

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC plotPrVsFd(origin_airport_airline_dow_delays_pr, dest_airport_airline_dow_delays_pr)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### 5. Origin/Dest Vertex, A/B -> B/C Edges PageRank

// COMMAND ----------

val flightsTrain = spark.sql("""
WITH flights as (
SELECT ORIGIN, DEST
FROM group25.airlines_main 
WHERE YEAR < 2018
GROUP BY 1, 2)

SELECT concat(f.ORIGIN, ':', f.DEST) as ORIGIN, concat(s.ORIGIN, ':', s.DEST) as DEST
FROM flights f INNER JOIN flights s
ON f.DEST = s.ORIGIN""")

// COMMAND ----------

spark.createDataFrame(
  runPageRank(flightsTrain, 0.000001).map(port => {
    val aa = port._2._2.split(':')
    (aa(0), aa(1), port._2._1)
  }))
  .toDF("origin", "dest", "pageRank")
  .write.mode("overwrite")
  .saveAsTable("group25.FLIGHTS_PR_TRAIN")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Validate expected count of rows...

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC SELECT count(*)
// MAGIC FROM (SELECT ORIGIN, DEST
// MAGIC       FROM group25.airlines_main 
// MAGIC       WHERE YEAR < 2018
// MAGIC       GROUP BY 1, 2)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Plot Flight PR versus Percentage of Delayed Flights

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC flights_delays_pr = spark.sql("""
// MAGIC   SELECT pr.origin, pr.dest, pr.pageRank as pageRank, ap.fraction_delayed
// MAGIC   FROM group25.flight_delay_fraction ap left join group25.FLIGHTS_PR_TRAIN pr on ap.origin = pr.origin AND ap.dest = pr.dest
// MAGIC   WHERE ap.fraction_delayed < 1 AND ap.fraction_delayed > 0
// MAGIC """).toPandas()

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC fig, ax = plt.subplots(1, 1, figsize = (15, 7))
// MAGIC sns.regplot(x='pageRank', y='fraction_delayed', data=flights_delays_pr)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### 6. Origin/Dest/DOW Vertex, A/B -> B/C Edges PageRank

// COMMAND ----------

val flightsDowTrain = spark.sql("""
WITH flights as (
SELECT ORIGIN, DEST, DAY_OF_WEEK
FROM group25.airlines_main 
WHERE YEAR < 2018
GROUP BY 1, 2, 3)

SELECT concat(f.ORIGIN, ':', f.DEST, ':', f.DAY_OF_WEEK) as ORIGIN, concat(s.ORIGIN, ':', s.DEST, ':', s.DAY_OF_WEEK) as DEST
FROM flights f INNER JOIN flights s
ON f.DEST = s.ORIGIN AND f.DAY_OF_WEEK = s.DAY_OF_WEEK""")

// COMMAND ----------

spark.createDataFrame(
  runPageRank(flightsDowTrain, 0.00001).map(port => {
    val aa = port._2._2.split(':')
    (aa(0), aa(1), aa(2), port._2._1)
  }))
  .toDF("origin", "dest", "day_of_week", "pageRank")
  .write.mode("overwrite")
  .saveAsTable("group25.FLIGHTS_DOW_PR_TRAIN")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Validate expected count of rows...

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC SELECT count(*)
// MAGIC FROM (SELECT ORIGIN, DEST, DAY_OF_WEEK
// MAGIC       FROM group25.airlines_main 
// MAGIC       WHERE YEAR < 2018
// MAGIC       GROUP BY 1, 2, 3)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #### Plot PR versus Percentage of Delayed Flights Per Airport/Airline/Day of Week, Delays Only

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC flights_dow_delays_pr = spark.sql("""
// MAGIC   SELECT pr.origin, pr.dest, pr.day_of_week, pr.pageRank as pageRank, ap.fraction_delayed
// MAGIC   FROM group25.flight_dow_delay_fraction ap left join group25.FLIGHTS_DOW_PR_TRAIN pr on ap.origin = pr.origin AND ap.dest = pr.dest AND ap.day_of_week = pr.day_of_week
// MAGIC   WHERE ap.fraction_delayed < 1 AND ap.fraction_delayed > 0
// MAGIC """).toPandas()

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC fig, ax = plt.subplots(1, 1, figsize = (15, 7))
// MAGIC sns.regplot(x='pageRank', y='fraction_delayed', data=flights_dow_delays_pr)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### 7. Origin/Dest/DOW/HOUR Vertex, A/B -> B/C Edges PageRank

// COMMAND ----------

val flightsDowHourTrain = spark.sql("""
WITH firsts as (
SELECT ORIGIN, DEST, DAY_OF_WEEK, ARR_UTC_HOUR as HOUR
FROM group25.airlines_utc_main 
WHERE YEAR < 2018
GROUP BY 1, 2, 3, 4),
seconds as (
SELECT ORIGIN, DEST, DAY_OF_WEEK, DEP_UTC_HOUR as HOUR
FROM group25.airlines_utc_main 
WHERE YEAR < 2018
GROUP BY 1, 2, 3, 4
)

SELECT concat(f.ORIGIN, ':', f.DEST, ':', f.DAY_OF_WEEK, ':', f.HOUR) as ORIGIN, concat(s.ORIGIN, ':', s.DEST, ':', s.DAY_OF_WEEK, ':', s.HOUR) as DEST
FROM firsts f INNER JOIN seconds s
ON f.DEST = s.ORIGIN AND f.DAY_OF_WEEK = s.DAY_OF_WEEK AND f.HOUR = s.HOUR""")

// COMMAND ----------

spark.createDataFrame(
  runPageRank(flightsDowHourTrain, 0.0001).map(port => {
    val aa = port._2._2.split(':')
    (aa(0), aa(1), aa(2), aa(3), port._2._1)
  }))
  .toDF("origin", "dest", "day_of_week", "hour", "pageRank")
  .write.mode("overwrite")
  .saveAsTable("group25.FLIGHTS_DOW_HOUR_PR_TRAIN")

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC SELECT count(*)
// MAGIC FROM (SELECT ORIGIN, DEST, DAY_OF_WEEK, DEP_UTC_HOUR AS HOUR
// MAGIC       FROM group25.airlines_utc_main 
// MAGIC       WHERE YEAR < 2018
// MAGIC       GROUP BY 1, 2, 3, 4
// MAGIC       UNION
// MAGIC       SELECT ORIGIN, DEST, DAY_OF_WEEK, ARR_UTC_HOUR AS HOUR
// MAGIC       FROM group25.airlines_utc_main 
// MAGIC       WHERE YEAR < 2018
// MAGIC       GROUP BY 1, 2, 3, 4
// MAGIC )

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC flights_dow_hour_delays_pr = spark.sql("""
// MAGIC   SELECT pr.origin, pr.dest, pr.day_of_week, pr.hour, pr.pageRank as pageRank, ap.fraction_delayed
// MAGIC   FROM group25.flight_dow_hour_delay_fraction ap left join group25.FLIGHTS_DOW_HOUR_PR_TRAIN pr on ap.origin = pr.origin AND ap.dest = pr.dest AND ap.day_of_week = pr.day_of_week
// MAGIC   WHERE ap.fraction_delayed < 1 AND ap.fraction_delayed > 0
// MAGIC """).toPandas()

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC fig, ax = plt.subplots(1, 1, figsize = (15, 7))
// MAGIC sns.regplot(x='pageRank', y='fraction_delayed', data=flights_dow_hour_delays_pr)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC # __Section 4__ - Topic Specific Page Rank

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC The first hurdle we faced in attmpeting to compute topic sensitive Page Rank was figuring out what implementation we could use. We could grab our HW5 implementation and adapt it relatively easily; the only modification we need to make is to only distribute the dangling node mass from teleportation to nodes within the 'topic' in question.

// COMMAND ----------

val airportAirlineTrain = spark.sql("SELECT CONCAT(ORIGIN, ':', OP_CARRIER) AS ORIGIN, CONCAT(DEST, ':', OP_CARRIER) AS DEST FROM group25.airlines_main WHERE YEAR < 2018")

// COMMAND ----------

val airlines: Array[String] = spark.sql("SELECT DISTINCT OP_CARRIER FROM group25.airlines_main WHERE year < 2018").collect().map(a => a.getString(0))
println(s"Distinct airlines: ${airlines.size}")

val aa: Array[String] = spark.sql("""(SELECT CONCAT(ORIGIN, ':', OP_CARRIER) as airportAirline FROM group25.airlines_main WHERE year < 2018 GROUP BY 1) 
  UNION 
  (SELECT CONCAT(DEST, ':', OP_CARRIER) as airportAirline FROM group25.airlines_main WHERE year < 2018 GROUP BY 1)""").collect().map(a => a.getString(0))
println(s"Distinct airport/airlines: ${aa.size}")

// COMMAND ----------

val (v, e, g) = buildGraph(airportAirlineTrain)

val airlineRanks = airlines.take(1).map { airline =>
  // get array of vertex IDs within this subject
  val sourceIds = aa.filter(_.endsWith(airline)).map(code)
  PageRank.runParallelPersonalizedPageRank(graph=g, numIter=10, sources=sourceIds)
}

airlineRanks(0).vertices

// COMMAND ----------

airlineRanks(0)
  .vertices
  .join(v)
  .sortBy(_._2._1, ascending=false)
  .take(10)
  .foreach(println)

// COMMAND ----------

# part c - job to initialize the graph (RUN THIS CELL AS IS)
def initGraph(dataRDD):
    """
    Spark job to read in the raw data and initialize an 
    adjacency list representation with a record for each
    node (including dangling nodes).
    
    Returns: 
        graphRDD -  a pair RDD of (node_id , (score, edges))
        
    NOTE: The score should be a float, but you may want to be 
    strategic about how format the edges... there are a few 
    options that can work. Make sure that whatever you choose
    is sufficient for Question 8 where you'll run PageRank.
    """
    ############## YOUR CODE HERE ###############

    # write any helper functions here
    def get_adj_list(line):
        node, edges = line.split('\t')
        edge_list = [(k,v) for k,v in ast.literal_eval(edges).items()]
        yield (node, edge_list)
        for edge_node in edge_list:
          yield(edge_node[0], [])
    
    # write your main Spark code here
    graphRDD = dataRDD.flatMap(lambda x: get_adj_list(x))\
                       .reduceByKey(lambda x,y: x + y).cache()
    N = graphRDD.count()
    graphRDD = graphRDD.map(lambda x: (x[0], (1.0/N, list(set(x[1])))))
    ############## (END) YOUR CODE ##############
    
    return graphRDD

# part d - job to run PageRank (RUN THIS CELL AS IS)
def runPageRank(graphInitRDD, alpha = 0.15, maxIter = 10, verbose = True):
    """
    Spark job to implement page rank
    Args: 
        graphInitRDD  - pair RDD of (node_id , (score, edges))
        alpha         - (float) teleportation factor
        maxIter       - (int) stopping criteria (number of iterations)
        verbose       - (bool) option to print logging info after each iteration
    Returns:
        steadyStateRDD - pair RDD of (node_id, pageRank)
    """
    # teleportation:
    a = sc.broadcast(alpha)
    
    # damping factor:
    d = sc.broadcast(1-a.value)
    
    # initialize accumulators for dangling mass & total mass
    mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    
    ############## YOUR CODE HERE ###############
    
    # write your helper functions here, 
    # please document the purpose of each clearly 
    # for reference, the master solution has 5 helper functions.

      
    def distribute_mass(line):
      """
      Function to emit a key-value pair for each record
      
      ###Input
      line is (node_id, (page_rank, adj_list))
      adj_list is comprised of a list of tuples, where each tuple is (outgoing node id, number of times it is linked in the node id page)
      
      ###Output
      For each line, emit 2 records
      1) Emit (node, (0.0, adj_list))
      
      2) Emit based on whether or not it is an dangling node
      If it is a dangling node, emit
        ('Dangling' (page_rank, []))
      
      Otherwise, emit 
        (outgoing node_id, (page_rank * weighted distribution of mass, []))
      
      """
      
      # Get node id
      node = line[0]
      
      #Get page rank
      page_rank = line[1][0]

      # Get adjacency list
      adj_list = line[1][1]

      # Get number of adjacent nodes
      num_adj = len(adj_list)

      # Emit default 0 page rank score and adjacency list for node 
      yield(node, (0.0, adj_list))
      
      total_weight = 0

      #Check if node is a dangling node
      if num_adj > 0:

        # Get total number of outgoing page links
        for item in adj_list:
          total_weight += item[1]

        for item in adj_list:
          yield(item[0], (page_rank*item[1]/total_weight,[]))
          
      else:
        
        # Emit mass of dangling node
        yield('##Dangling', (page_rank, []))
                
    # write your main Spark Job here (including the for loop to iterate)
    # for reference, the master solution is 21 lines including comments & whitespace

    if verbose:
      print('-------- FINISHED INITIALIZATION------')
    N_bc = sc.broadcast(graphInitRDD.count())
    
    # loop through each iteration
    for i in range(maxIter):
      mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
      totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
      
      graphInitRDD.foreach(lambda x: totAccum.add(x[1][0]))
  
      if verbose:
        print('Initial Dangling Mass for iter {}: {}'.format(i,mmAccum.value))

      # NORMAL MAP REDUCE FOR PAGE_RANK CALCULATION
      steadyStateRDD = graphInitRDD.flatMap(lambda x: distribute_mass(x)) \
                                   .reduceByKey(lambda x,y: (x[0] +  y[0], x[1] + y[1])) \
                                   .cache()
      # Add all dangling masses
      #print(steadyStateRDD.filter(lambda x: x[0]=='##Dangling').collect())
      mmAccum.add(steadyStateRDD.filter(lambda x: x[0]=='##Dangling').collect()[0][1][0])
      
      # Remove dangling nodes from RDD 
      steadyStateRDD = steadyStateRDD.filter(lambda x: x[0]!='##Dangling')

      mmAccum_bc = sc.broadcast(mmAccum.value)
      if verbose:
#         print('{}: {}'.format(i, mmAccum_bc.value))
        print('Dangling Mass for iter {}: {}'.format(i, mmAccum_bc.value))
        print('Total Mass for iter {}: {}'.format(i, totAccum.value))
        data = steadyStateRDD.collect()
        for item in data:
          print(item)
  
      # SECOND MAP REDUCE JOB
      steadyStateRDD = steadyStateRDD.mapValues(lambda x: (a.value/N_bc.value + d.value * (mmAccum_bc.value/N_bc.value + x[0]), x[1])).cache()
      
      # Reset graph initialization to result of last iteration
      graphInitRDD = steadyStateRDD
      
      if verbose:
        data = steadyStateRDD.take(10)
        print("Iter: {}".format(i))
        for item in data:
          print(item)
    
    # Reformat RDD
    steadyStateRDD = steadyStateRDD.map(lambda x: (x[0], x[1][0]))
    
    ############## (END) YOUR CODE ###############

    return steadyStateRDD