# Databricks notebook source
# MAGIC %md # w261 Final Project - Airline Delays Prediction

# COMMAND ----------

# MAGIC %md 25   
# MAGIC Justin Trobec, Jeff Li, Sonya Chen, Karthik Srinivasan
# MAGIC Spring 2021, section 5, Team 25

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## last meeting notes
# MAGIC 
# MAGIC * Examine different thresholds cut offs for models *  Karthik
# MAGIC * Gradient Boosted Trees *  Karthik
# MAGIC * Fix features where we had to impute a lot of values (mostly done?)
# MAGIC * Presentation for next week
# MAGIC     * Flow charts
# MAGIC     * EDA plots
# MAGIC     * Add code slides
# MAGIC         * UTC/Timezone formatting *  Jeff
# MAGIC         * Example of how Weather features were generated *  Jeff
# MAGIC         * Page Rank *  Justin
# MAGIC         * Inbound Outbound *  Sonya
# MAGIC         * Code for Modelling *  Karthik
# MAGIC 	* A few plots to show how the distribution works after the imputation
# MAGIC 	* Inbound/Outbound
# MAGIC 	* Page Rank
# MAGIC * Write up results
# MAGIC * Notebooks aspect
# MAGIC * Clean up individual notebooks
# MAGIC 	* Main
# MAGIC         *  Toy Model *  take hw4, change it to logistic regression, run a small dataset through (Jeff)
# MAGIC         *  Business Case/Question Formulation *  Jeff
# MAGIC             *  SOTA in the domain
# MAGIC             *  Discuss evaluation Metrics
# MAGIC         *  EDA and Discussion of Challenges *  Justin
# MAGIC             *  EDA plots *  Justin, but Sonya and Jeff need to provide some good features for plotting
# MAGIC                 * Show MV features *  show how sparse they are
# MAGIC                 * Correlation Matrix *  Justin
# MAGIC             *  Challenges *  Justin
# MAGIC                 *  Large amount of Weather Vars, multiople dimensions (Jeff)
# MAGIC         *  Feature Engineering
# MAGIC             *  Inbound Outbound, Delays, Divered Flights *  Sonya
# MAGIC             *  Time aspect *  Sonya
# MAGIC             *  Leakage *  Sonya
# MAGIC             *  Page Rank *  Justin
# MAGIC         *  Algorithm Exploration *  Karthik
# MAGIC         *  Algorithm Implementation *  Karthik
# MAGIC         *  Conclusions *  Karthik 
# MAGIC         *  Application of Course Concepts *  Karthik
# MAGIC             *  Normalization 
# MAGIC             *  Assumptions for Different Algorithms
# MAGIC             *  Page Rank *  Justin
# MAGIC             *  Regularization
# MAGIC             *  One Hot Encoding / vector embeddings / feature selection
# MAGIC 	* Page Rank
# MAGIC 	* Weather
# MAGIC 	* Inbound outbound
# MAGIC 	* Joining
# MAGIC 	* Model
# MAGIC     * Conclusion
# MAGIC done
# MAGIC  * How many records are imputed for our dataset (done)

# COMMAND ----------

# MAGIC %md ## Table of Contents
# MAGIC 
# MAGIC * __Section 1__ - Question Formulation
# MAGIC * __Section 2__ - EDA & Discussion of Challenges
# MAGIC * __Section 3__ - Feature Engineering
# MAGIC * __Section 4__ - Algorithm Explanation
# MAGIC * __Section 5__ - Algorithm Implementation
# MAGIC * __Section 6__ - Conclusions
# MAGIC * __Section 7__ - Application of Course Concepts
# MAGIC * __Section 8__ - Companion Notebooks
# MAGIC * __Section 9__ - Appendix

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Companion Notebooks
# MAGIC 
# MAGIC Our team constructed different Databricks notebooks which provide the underlying code for how each of the following features plots were constructed. Contents such as plots will be referenced in this notebook, but the underlying code can be found in the companion notebooks:
# MAGIC 
# MAGIC - Data Processing (Time Features)
# MAGIC     - [Databricks](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4480952025475049/command/528073926731710)
# MAGIC     - [Github](https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/notebooks/Users/sonya/team25_final_project_airlines_main_sonya.ipynb)
# MAGIC 
# MAGIC - Weather
# MAGIC     - [Databricks](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4377823981601721/)
# MAGIC     - [Github](https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/notebooks/Users/jeffli930%40berkeley.edu/final_project_team25_weather.py)
# MAGIC 
# MAGIC - PageRank
# MAGIC     - Databricks
# MAGIC     - Github
# MAGIC 
# MAGIC - Inbound/Outbound
# MAGIC     - Databricks
# MAGIC     - [Github](https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/notebooks/Users/sonya/airports_inbound_outbound_diverted_sonya_v7.py)
# MAGIC 
# MAGIC - Join/Imputation Notebook
# MAGIC     - [Databricks](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/439895120619713/command/439895120619714)
# MAGIC     - [Github](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/439895120619713)
# MAGIC     
# MAGIC     
# MAGIC - Modeling
# MAGIC     - [Databricks](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/439895120621294)
# MAGIC     - [Github](https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/notebooks/Users/kasri%40berkeley.edu/final_project_team25_model.py)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Slides
# MAGIC * [Slides](https://docs.google.com/presentation/d/1Md7yOAzOhwlWGLVXru7cAlbE2OaZ3W4z34lLEQ3ESxU/edit)

# COMMAND ----------

# MAGIC %md ## Imports

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

from IPython.display import Image 
from IPython.core.display import HTML 

# COMMAND ----------

# MAGIC %md # S1 - Question Formulation
# MAGIC 
# MAGIC You should refine the question formulation based on the general task description you’ve been given, ie, predicting flight delays. This should include some discussion of why this is an important task from a business perspective, who the stakeholders are, etc.. Some literature review will be helpful to figure out how this problem is being solved now, and the State Of The Art (SOTA) in this domain. Introduce the goal of your analysis. What questions will you seek to answer, why do people perform this kind of analysis on this kind of data? Preview what level of performance your model would need to achieve to be practically useful. Discuss evaluation metrics.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ##  Overview
# MAGIC 
# MAGIC This notebook is a summary notebook of our team's effort to predict flight delays. Predicting flight delays is no small feat. From a high level, [airline delays cost ~ $33B](https://www.faa.gov/data_research/aviation_data_statistics/media/cost_delay_estimates.pdf). Not only that, approximately 19% of flights were delayed in 2019 [and approximately 1.5% of the flights were cancelled](https://www.mit.edu/~hamsa/pubs/GopalakrishnanBalakrishnanATM2017.pdf). 
# MAGIC 
# MAGIC While this problem is one that can definitely be continuously dug into and improved upon, we hope that some of the ideas and approaches outlined in this notebook can be applied to improving prediction down the line. 
# MAGIC 
# MAGIC We have featured some papers here that have helped us inform our approach to this problem.
# MAGIC [Some comment on state of the art - ask Karthik] [review]
# MAGIC 
# MAGIC 
# MAGIC ### Evaluation metrics
# MAGIC 
# MAGIC A brief comment: a common approach to evaluating most ML based problems is to look at problems from a precision recall approach. Emphasizing precision will position our target towards emphasizing false positive whereas recall will position our target against false negatives. Accidentally predicting a flight may be delayed (false positive) cause someone to show up later than expected to a flight. Conversely, failing to predict a delayed flight may cause someone to show up earlier than needed to an airport.
# MAGIC 
# MAGIC There are many people in the economic system who are affected by the accuracy of flight predictions - flight passengers, airport and airline employees, as well as general shareholders of airline companies. Ultimately, it is incumbent on the customer (namely, airline companies) to accurately measure the dollar cost of delayed flights from a FP, FN perspective, and which one to subsequently optimize for. For the purposes of this notebook, we will stick to optimizing on our models based on a F1 score to have a good balance between precision & recall.
# MAGIC 
# MAGIC The formula for an F beta score is below:
# MAGIC 
# MAGIC $$F_{\beta} = (1 + \beta^{2}) \frac{precision \cdot recall}{(\beta^{2} \cdot precision) + recall} $$
# MAGIC 
# MAGIC For reference, utilizing a F0.5 score would produce an F score more geared towards precision while still factoring in recall. On the other hand, utilizing a F2 score would bias our results more towards recall as opposed to precision. 

# COMMAND ----------

# MAGIC %md 

# COMMAND ----------

# MAGIC %md # S2 - EDA & Discussion of Challenges
# MAGIC 
# MAGIC Determine a handful of relevant EDA tasks that will help you make decisions about how you implement the algorithm to be scalable. Discuss any challenges that you anticipate based on the EDA you perform.

# COMMAND ----------

# MAGIC %md 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Sources of Data
# MAGIC 
# MAGIC 
# MAGIC - Flight plans with departure delays and flight information for travel within US (between 2015-2019)
# MAGIC - Weather data aggregated across various US weather stations (between 2015-2019) [1]
# MAGIC - [Airport codes and timezones](https://openflights.org/data.html)
# MAGIC 
# MAGIC 
# MAGIC #### Variable distribution/count
# MAGIC 
# MAGIC #### Relation of variable to outcome
# MAGIC 
# MAGIC #### Engineered Features
# MAGIC 
# MAGIC (Each of us should come up with 2-3 columns that would be interesting to plot - we want to plot them here in 1 go so that the format is standardized)

# COMMAND ----------

# MAGIC %md # S3 - Feature Engineering
# MAGIC 
# MAGIC Apply relevant feature transformations, dimensionality reduction if needed, interaction terms, treatment of categorical variables, etc.. Justify your choices.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Pipeline Breakdown
# MAGIC A high level overfiew of our overall feature engineering process can be found below. We will cover some of the interesting feature engineering topics from this pipeline.

# COMMAND ----------

displayHTML("<img src = 'https://raw.githubusercontent.com/kasri-mids/ucb-w261-sp2021-team25/main/images/w261_flowchart_1.png'>")

# COMMAND ----------

# MAGIC %md some writeup here on the prediction pipeline
# MAGIC [review]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time
# MAGIC 
# MAGIC ### Motivation
# MAGIC 
# MAGIC A common trope for time is that it is a precious resource, and that is doubly true for the field of flight delay prediction. The concept of time was a topic we discussed heavily when tackling this problem - more specifically, the granularity. For instance, will knowing the information for the previous year help us make an informed decision for whether the next flight will be delayed? Possibly, we also wanted to have more granular data as well, since we believed that there would be much better granularity. 
# MAGIC 
# MAGIC On the other hand, will knowing the flight data in 5 minute increments and binning the data as such lead to better predictions? That is certainly possible, but may also lead to some additional processing for complicated pipelined features.
# MAGIC 
# MAGIC Ultimately, we settled on an hourly granularity. While this was partially based on consensus and intuition, it is a nice round unit of time that is widely understood and helps to make this analysis more adjustable. However, more granular binnings of time, such as by 5 minute increments, could possibly yield better results.
# MAGIC 
# MAGIC In terms of how far in advance we would have this information, we decided to try and gather a set of features for each airport 2 hours in advance. For instance, what was the weather 2 hours prior to the scheduled takeoff? What was the status of delays and divered flights at the outgoing airport?
# MAGIC 
# MAGIC The challenge with this approach was that not all of our data was formatted correctly. While the weather data was given in a standardized UTC format, the flight data was not. Ensuring that our dataset utilized consistent time formats across the board was important to predicting delays 2 hours out. The datetime format is critical for several niche problems with joining on data from 2 hours prior. For instance, what if you want flight data 2 hours prior to a 1 am flight? You cannot join simply on FL_Date and a hour granularity. Furthermore, once you start working with timezones and daylight savings, things get especially complicated. A good rule of thumb for sanity is to generate all of your datetime formats first as upstream as possible, before applying any interval functions. Any downstream code or engineered features benefit immensely from this approach.
# MAGIC 
# MAGIC In terms of grabbing the time zone data per airport, we utilized the OpenFlights data. In order to ensure proper matching with our dataset, we utilized timezone data from [OpenFlights](https://openflights.org/data.html).
# MAGIC 
# MAGIC Spark has some limited inbuilt capabilities when it comes to handling timezones, so we leveraged more commonly used datetime python libraries. The issue with this is that you have to run your data through a Spark UDF to leverage the datetime library at scale. While doing this is not the most performant Spark operation, it did work with the scale of data we had in a reasonable amount of time.
# MAGIC 
# MAGIC A flowchart illustrating the challenge with time as a mind bending engineering challenge in tackling this problem is below. 

# COMMAND ----------

displayHTML("<img src = 'https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/images/w261_flowchart3.png?raw=true'>")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weather
# MAGIC 
# MAGIC ### Data Overview
# MAGIC Our team utilized data from NOAA's National Centers for Environmental Information. We found this data to be extremely rich, with 630,904,436 records. Not only was this dataset extremely dense, it consisted of is comprised of multiple different report types. A table of the different report types and their descriptions is provided below.
# MAGIC 
# MAGIC | Report Type | Description | Number of Records
# MAGIC |-----------------|---------------|---------------|
# MAGIC |CRN05|Climate Reference Network report, with 5-minute reporting interval|80,363,686
# MAGIC |FM-12|SYNOP Report of surface observation form a fixed land station|171,069,693
# MAGIC |FM-13|SHIP Report of surface observation from a sea station|5,041,888
# MAGIC |FM-14|SYNOP MOBIL Report of surface observation from a mobile land station|10,988,064
# MAGIC |FM-15|METAR Aviation routine weather report|313,794,637
# MAGIC |FM-16|SPECI Aviation selected special weather report|23,381,469
# MAGIC |FM-18|BUOY Report of a buoy observation|109,418
# MAGIC |SAO|Airways report (includes record specials)|22,149,449
# MAGIC |SAOSP|Airways special report (excluding record specials)|3,683
# MAGIC |SHEF|Standard Hydrologic Exchange Format|422,398
# MAGIC |SOD|Summary of Day|3,268,119
# MAGIC |SOM|Summary of Month|64,346
# MAGIC |SURF|Surface Radiation Network report|235,641
# MAGIC |SY-MT|Synoptic and METAR merged report|11,945
# MAGIC 
# MAGIC Furthermore, there was 177 different columns present within the dataset. For the purpose of brevity, we will not highlight every column in the dataset, however, a more detailed description for each column can be found in the following document. https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
# MAGIC 
# MAGIC We will cover one of the main fields that we utilized in our modeling process below.
# MAGIC 
# MAGIC ### Example of how a field was parsed
# MAGIC 
# MAGIC We will walk through the field AA1 as an example for how features were mined in the NOAA dataset. All of the other columns we mined for used a similar format, so highlighting one instance should also inform how the other fields were parsed. The field AA1 is actually a comma delimited field. This applies to most, if not all columns in the ISD dataset. An observation for AA1 may look like the following.
# MAGIC 
# MAGIC __Sample Field:__ 04,9996,4,2
# MAGIC 
# MAGIC Based on the ISD format document provided by NOAA, the definition for the field can be interpreted as the following
# MAGIC 
# MAGIC * First delimited field: LIQUID-PRECIPITATION period quantity
# MAGIC * Second delimited field: LIQUID-PRECIPITATION depth dimension
# MAGIC * Third delimited field: LIQUID-PRECIPITATION condition code
# MAGIC * Fourth delimited field: LIQUID-PRECIPITATION quality code
# MAGIC 
# MAGIC Using the definitions present in the NOAA dataset, we filtered down the dataset on the 3rd and 4th delimited fields (condition code, quality code), which give information about the quality about the particular record. For instance, a quality field with the value "2" is denoted as being suspect according to the ISD document. 
# MAGIC 
# MAGIC Once the appropriate records were filtered out, we then aggregated the data on a callsign (airport), date, and hourly basis for the respective field (AA1), and then took the average at each aggregation point. 
# MAGIC 
# MAGIC 
# MAGIC #### Why AA1?
# MAGIC 
# MAGIC When examining the NOAA dataset for instances of precipitation
# MAGIC 
# MAGIC [review] - Jeff note - will finish this before Friday

# COMMAND ----------

# MAGIC 
# MAGIC %sql
# MAGIC select * from group25.weather_base_table limit 100

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Page Rank

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inbound/Outbound Flights

# COMMAND ----------

# MAGIC %md ### inbound & outbound related dataframs compute flowcharts

# COMMAND ----------

displayHTML("<img src = 'https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/images/airports_inbound_outbound_dataframes_flowchart.png?raw=true'> style='height:20%;width:20%' ")

# COMMAND ----------

# MAGIC %md ### Maximum Median Compute Flowchart

# COMMAND ----------

displayHTML("<img src = 'https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/images/compute_max_median_for_inbound_and_outbound.png?raw=true'> style='height:20%;width:20%' ")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Notebook for Inbound & Outbound Features: 
# MAGIC https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/439895120630397/command/439895120630399
# MAGIC 
# MAGIC ### Hypothesis & backgound
# MAGIC - When we first approach the flights delay problem, we think that inbound flights and outbound flights should we great features for predicting the flights delay or not. 
# MAGIC - Busier airports could be more likely to have flights delay due to capacity constraints, tigher resources, and other reasons all likely to cause a flight at that airport to delay.
# MAGIC 
# MAGIC ### 1st Iteration: 
# MAGIC - So we compute related inbounds and outbound features: 
# MAGIC   - For outbound features, we caluclated:
# MAGIC     - outbound_counts /airpot/flight date/departure local hour
# MAGIC     - diverted outbound_counts /airpot/flight date/departure local hour
# MAGIC     - delay outbound_counts /airpot/flight date/departure local hour
# MAGIC   - For inbound features, we calculated:
# MAGIC     - inbound_counts /airport/flight date/arr local hour
# MAGIC     - diverted inbound_counts /airport/flight date/arr local hour
# MAGIC     - delay inbound_counts /airport/flight date/arr local hour
# MAGIC 
# MAGIC #### Problem
# MAGIC However, we we plot out the feature, we noticed that there isn't much difference for these set of features between delayed flights and non-delayed flights. Thus, it's not quit useful at a first glance.
# MAGIC 
# MAGIC #### Analysis of Problem
# MAGIC Later, we hypothesized that it could be the problem with normalization. The underlying reason is that each airport has different capacities. Say 100 outbounds flights could be the maximum limit of aiprotA but it could be a median outbound number for airportB (which could have a much bigger capcity). 
# MAGIC 
# MAGIC Another problem with raw number of inbound and outbound count is that aiports might have been through renovation during the year between 2015-2020. So an obsolute raw number could not take into count of whether an airport is busy or not busy with an absolute number. An airport might have increaes its capcity by say 20% after renovation, such that 100 outbound/hour could means that aiport is busy before renovation, yet 100 outbound/hour could be be easy to handle for that airports. 
# MAGIC 
# MAGIC So we went to search for data that tells us about each airport's capacity at each year. Eventually, we didn't find ideal data.
# MAGIC 
# MAGIC #### Solution: Max Median
# MAGIC So finally, we decided to use max median inbound and outbound counts to normalized the raw inbound and outbound data.
# MAGIC 
# MAGIC - The idea is that we want to find a number that normalized the flights statistics for each aiport. 
# MAGIC   - Step1: we compute the median inbound/outbound number for each airport at each hour (Say median inbound for every 11am for aiportA). 
# MAGIC   - Step2: we find the maximum among all the medians for that particular aiports
# MAGIC 
# MAGIC For code of computing max_median_inbound & max_median_outbound, please refer to here: https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/439895120630397/command/439895120630403
# MAGIC 
# MAGIC #### Data Leakage: 
# MAGIC - For prevent data-leakge when calculate the median for inbound counts and outbound bounds, we only use data in 2015-2017 to calculate the median.
# MAGIC - The reason we use only training data (data in 2015-2017) to calulate median is because our model shouldn't know any data from the future. 
# MAGIC - So we calulate median based on data from 2015-2017, and used that to normalized all the data from 2015-2020. 
# MAGIC 
# MAGIC ### 2nd Iteration: 
# MAGIC - So in 2nd iteration, we normalized the sets of features in the 1st iteration. We calculate the normalized inbound/outbound related features by dividing the raw statistics by max_median for each aiports
# MAGIC - For outbound features, we caluclated:
# MAGIC   - normalized_outbound_counts = outbound_counts / max_median_outbound
# MAGIC   - normalized_diverted_outbound_counts = diverted_outbound_counts / max_median_outbound
# MAGIC   - normalized_delay_outbound_counts = delay_outbound_counts / max_median_outbound
# MAGIC - For inbound features, we calculated:
# MAGIC   - normalized_inbound_counts = inbound_counts / max_median_inbound
# MAGIC   - normalized_diverted inbound_counts = diverted_inbound_counts / max_median_inbound
# MAGIC   - normalized_delay inbound_counts = delay_inbound_counts / max_median_inbound
# MAGIC   
# MAGIC 
# MAGIC ### Time Series of Inbound & Outbound Features
# MAGIC - To account for the time series-aspect of the airlines data, we also calculate something we called as lag features. 
# MAGIC - These features ends like faeture_name_xH (for example: NORMALIZED_OUTBOUND_COUNT_2H)
# MAGIC - For example NORMALIZED_OUTBOUND_COUNT_2H for a flight departs at 11am at AirportA, that means that normalized outbound counts from 8am-9am (1 hour timeframe, 2 hours prior the depature time) at airpotA.
# MAGIC 
# MAGIC - Below are all the lag features for inbound & outbound features: 
# MAGIC     - OUTBOUND_COUNT_2H
# MAGIC     - OUTBOUND_COUNT_3H
# MAGIC     - OUTBOUND_COUNT_4H
# MAGIC     - OUTBOUND_COUNT_5H
# MAGIC     - OUTBOUND_COUNT_6H
# MAGIC     - NORMALIZED_OUTBOUND_COUNT_2H
# MAGIC     - NORMALIZED_OUTBOUND_COUNT_3H
# MAGIC     - NORMALIZED_OUTBOUND_COUNT_4H
# MAGIC     - NORMALIZED_OUTBOUND_COUNT_5H
# MAGIC     - NORMALIZED_OUTBOUND_COUNT_6H
# MAGIC 
# MAGIC     - INBOUND_COUNT_2H
# MAGIC     - INBOUND_COUNT_3H
# MAGIC     - INBOUND_COUNT_4H
# MAGIC     - INBOUND_COUNT_5H
# MAGIC     - INBOUND_COUNT_6H
# MAGIC     - NORMALIZED_INBOUND_COUNT_2H
# MAGIC     - NORMALIZED_INBOUND_COUNT_3H
# MAGIC     - NORMALIZED_INBOUND_COUNT_4H
# MAGIC     - NORMALIZED_INBOUND_COUNT_5H
# MAGIC     - NORMALIZED_INBOUND_COUNT_6H
# MAGIC 
# MAGIC     - DIVERTED_OUTBOUND_COUNT_2H
# MAGIC     - DIVERTED_OUTBOUND_COUNT_3H
# MAGIC     - DIVERTED_OUTBOUND_COUNT_4H
# MAGIC     - DIVERTED_OUTBOUND_COUNT_5H
# MAGIC     - DIVERTED_OUTBOUND_COUNT_6H
# MAGIC     - NORMALIZED_DIVERTED_OUTBOUND_COUNT_2H
# MAGIC     - NORMALIZED_DIVERTED_OUTBOUND_COUNT_3H
# MAGIC     - NORMALIZED_DIVERTED_OUTBOUND_COUNT_4H
# MAGIC     - NORMALIZED_DIVERTED_OUTBOUND_COUNT_5H
# MAGIC     - NORMALIZED_DIVERTED_OUTBOUND_COUNT_6H
# MAGIC 
# MAGIC     - DIVERTED_INBOUND_COUNT_2H
# MAGIC     - DIVERTED_INBOUND_COUNT_3H
# MAGIC     - DIVERTED_INBOUND_COUNT_4H
# MAGIC     - DIVERTED_INBOUND_COUNT_5H
# MAGIC     - DIVERTED_INBOUND_COUNT_6H
# MAGIC     - NORMALIZED_DIVERTED_INBOUND_COUNT_2H
# MAGIC     - NORMALIZED_DIVERTED_INBOUND_COUNT_3H
# MAGIC     - NORMALIZED_DIVERTED_INBOUND_COUNT_4H
# MAGIC     - NORMALIZED_DIVERTED_INBOUND_COUNT_5H
# MAGIC     - NORMALIZED_DIVERTED_INBOUND_COUNT_6H
# MAGIC     
# MAGIC     - DELAY_OUTBOUND_COUNT_2H
# MAGIC     - DELAY_OUTBOUND_COUNT_3H
# MAGIC     - DELAY_OUTBOUND_COUNT_4H
# MAGIC     - DELAY_OUTBOUND_COUNT_5H
# MAGIC     - DELAY_OUTBOUND_COUNT_6H
# MAGIC     - NORMALIZED_DELAY_OUTBOUND_COUNT_2H
# MAGIC     - NORMALIZED_DELAY_OUTBOUND_COUNT_3H
# MAGIC     - NORMALIZED_DELAY_OUTBOUND_COUNT_4H
# MAGIC     - NORMALIZED_DELAY_OUTBOUND_COUNT_5H
# MAGIC     - NORMALIZED_DELAY_OUTBOUND_COUNT_6H
# MAGIC 
# MAGIC     - DELAY_INBOUND_COUNT_2H
# MAGIC     - DELAY_INBOUND_COUNT_3H
# MAGIC     - DELAY_INBOUND_COUNT_4H
# MAGIC     - DELAY_INBOUND_COUNT_5H
# MAGIC     - DELAY_INBOUND_COUNT_6H
# MAGIC     - NORMALIZED_DELAY_INBOUND_COUNT_2H
# MAGIC     - NORMALIZED_DELAY_INBOUND_COUNT_3H
# MAGIC     - NORMALIZED_DELAY_INBOUND_COUNT_4H
# MAGIC     - NORMALIZED_DELAY_INBOUND_COUNT_5H
# MAGIC     - NORMALIZED_DELAY_INBOUND_COUNT_6H

# COMMAND ----------

# MAGIC %md ###Join Inbound and Outbound Data with Airline_UTC_Main
# MAGIC - We join the inbound and outbound related features with the airlines_utc_main table with these columns:
# MAGIC   - for outbound: 
# MAGIC     - call_sign_dep (the airport code)
# MAGIC     - fl_date (flight date)
# MAGIC     - dep_local_hour (the local hour at the departure airport)
# MAGIC   - for inbound:
# MAGIC     - call_sign_arr
# MAGIC     - fl_date
# MAGIC     - arr_local_hour

# COMMAND ----------

# MAGIC %md ## Diverted Inbound/Outbound Flights

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ###Hypothesis
# MAGIC - diverted flights will increase the traffic of airport unexpectedly, might potentially cause other flights at that airport to delay.
# MAGIC - For each airport / day / hour, we calculated the number of diverted flights at that airport
# MAGIC - When predict flight delay, take into account of that diverted flight numbers at that airport

# COMMAND ----------

# MAGIC %md ###Normalization
# MAGIC 
# MAGIC - Use like Inbounds & Outbound features, we normalized this feature with the max_median inbound/outbound counts of each airport. 

# COMMAND ----------

# MAGIC %md ### Rolling Windows
# MAGIC 
# MAGIC - Calculated Rolling Windowed Features: eg 2_hr_prior, 3_hr_prior, ...

# COMMAND ----------

# MAGIC %md ##Delay Inbound/Outbound Flights

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ###Hypothesis
# MAGIC - Prior Delay flights at airport could be an indicating sign for further future delay for other flights 
# MAGIC - For each airport / day / hour, we calculated the number of delayed flights at that airport
# MAGIC - When predict flight delay, take into account of that delayed flight numbers at that airport

# COMMAND ----------

# MAGIC %md ### Normalization
# MAGIC 
# MAGIC - Use like Inbounds & Outbound features, we normalized this feature with the max_median inbound/outbound counts of each airport. 

# COMMAND ----------

# MAGIC %md ### Rolling Windows
# MAGIC 
# MAGIC - Calculated Rolling Windowed Features: eg 2_hr_prior, 3_hr_prior, ...

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Sources of Data
# MAGIC 
# MAGIC 
# MAGIC - Flight plans with departure delays and flight information for travel within US (between 2015-2019)
# MAGIC - Weather data aggregated across various US weather stations (between 2015-2019) [1]
# MAGIC - [Airport codes and timezones](https://openflights.org/data.html)

# COMMAND ----------

# MAGIC %md ## Airline_Main dataset Handling
# MAGIC - We start with airline_main data.
# MAGIC - Then we convert the Fl_date and the departure local hour (dep_local_hour) into utc timestamp (dep_utc_timestamp). 
# MAGIC - Then we add the flight duration to the departure local hour (dep_local_hour) to get arrival utc timestamp (arr_utc_timestamp).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Null Value Imputations
# MAGIC 
# MAGIC Our team largely used Spark SQL to join data together from various tables. Unfortunately, Spark’s MLLib will not work if the features will null values. Because of this, addressing nulls is extremely important for effective model results. 
# MAGIC 
# MAGIC Our ML pipeline approach involved generating features, and creating separate tables, then joining them back to the main airlines table via left joins. This allowed our team to work effectively in parallel to use a divide and conquer approach to generate interesting features. However, when joining the tables back together, a key issue we faced with these joins is null values.
# MAGIC 
# MAGIC We decided to impute the values differently, based on whether or not the feature was categorical or continuous.
# MAGIC 
# MAGIC **Continuous**
# MAGIC 
# MAGIC For all features, loop through &calculate the median for each grouping below:
# MAGIC 
# MAGIC   1. Call_Sign, Month, Hour
# MAGIC   2. Call_Sign, Month
# MAGIC   3. Call_Sign
# MAGIC   4. Overall Median
# MAGIC 
# MAGIC For each row, if the feature does not exist, use group 1. If group 1 does not exist, use group 2. If group 2 does not exist, use group 3. If group 3 does not exist, use group 4.
# MAGIC 
# MAGIC **Categorical**
# MAGIC 
# MAGIC The approach for this is much simpler. If the value is null, impute a 0 for null features.
# MAGIC 
# MAGIC A diagram highlighting our approach can be found below. The underlying code for this can be found in the Join notebook linked in the Companion notebook section.

# COMMAND ----------

displayHTML("<img src = 'https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/images/null_value_imputation.png?raw=true'>")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC [review]
# MAGIC 
# MAGIC #### Describe how we prevent leakage
# MAGIC 
# MAGIC The data that is given is from 2015-2019. A key concern of ours was how to prevent leakage, or val or test datasets leaking into our test dataset. To set up our data pipeline to prevent leakage, we set up train, test, and validation training sets:
# MAGIC 
# MAGIC - Train - 01/01/2015 - 12/31/2017
# MAGIC - Validation - 01/01/2018 - 12/31/2018
# MAGIC - Test - 01/01/2019 - 12/31/2019
# MAGIC 
# MAGIC #### Describe train val and test
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC #### Describe intuition behind engineered features
# MAGIC #### Some reference publications

# COMMAND ----------

# MAGIC %md ## Data Leakage

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Innbound & Outbound Statistics
# MAGIC For the Inbound and Outbound related features, to prevent data leakage, so we connect the flights 
# MAGIC - For prevent data-leakge when calculate the median for inbound counts and outbound bounds, we only use data in 2015-2017 to calculate the median.
# MAGIC - The reason we use only training data (data in 2015-2017) to calulate median is because our model shouldn't know any data from the future. 
# MAGIC - So we calulate median based on data from 2015-2017, and used that to normalized all the data from 2015-2020. 
# MAGIC 
# MAGIC ### Time Series Aspect of Inbound & Outbound Features
# MAGIC - To account for the time series-aspect of the airlines data, we also calculate something we called as lag features. 
# MAGIC - These features ends like faeture_name_xH (for example: NORMALIZED_OUTBOUND_COUNT_2H)
# MAGIC - For example NORMALIZED_OUTBOUND_COUNT_2H for a flight departs at 11am at AirportA, that means that normalized outbound counts from 8am-9am (1 hour interval, 2 hours prior the depature time) at airpotA.
# MAGIC - Same theory applies for the inbound features. 
# MAGIC 
# MAGIC ### Weather Data
# MAGIC - To prevent data leakage, we only use weather data 2 hour prior the departure time of a particular flight. 
# MAGIC - For example, if a flight's departure date is 1/1/2016 10am, then we only use weather feature up until 1/1/2016 8am to predict whether its delay. 
# MAGIC 
# MAGIC ### Page Rank
# MAGIC - We only use data from 2015-2017 to compute pagerank features. 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##

# COMMAND ----------

# MAGIC %md # S4 - Algorithm Exploration
# MAGIC Apply 2 to 3 algorithms to the training set, and discuss expectations, trade-offs, and results. These will serve as your baselines - do not spend too much time fine tuning these. You will want to use this process to select a final algorithm which you will spend your efforts on fine tuning.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We define the target variable 'DEP_DEL15' as a boolean and use classification algorithms to predict it. We start with the majority class as baseline (null majority class model). We then experimented with 
# MAGIC * Logistic Regression (with and without regularization)
# MAGIC * Decision Trees
# MAGIC * Random Forests
# MAGIC * Gradient Boosted and eXtreme Gradient Boosted Trees
# MAGIC 
# MAGIC We split the data into train, validation and test sets. We used data between 2015-2017 for train, 2018 for validation and 2019 for test. For all models discussed here, we use 5-fold cross validation on the train set. We perform a parameter grid search on the train set and evaluate the candidate models on the validation set. Since our dataset is imbalanced, i.e., the number of delays is much smaller than the number of undelayed fights, \\( \approx 4 \\) times smaller, we use the sampling techniques discussed below to address the imbalance. All sampling experiments are only conducted on the train data.
# MAGIC * No over/under sampling: This is the baseline case with regards to sampling. No modifications are done to the train dataset.
# MAGIC * Class-Weighted Sampling: Here, we do not modify the underlying train dataset but scale the losses to accomodate the imbalance.
# MAGIC * Bootstrapping: We duplicate the minority class so that we are left with similar proportions of delay/no-delay records.
# MAGIC 
# MAGIC Other kinds of oversampling such as SMOTE were considered but not pursued due to time constraints.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Models
# MAGIC 
# MAGIC For all the models discussed below, we ran the training exercise on randomly selected 1 million records. We fine-tune the models on this smaller train set for faster down selection and eventually run the preferable candidate on a larger 'X' record dataset.
# MAGIC 
# MAGIC ##### Logistic Regression
# MAGIC 
# MAGIC We ran hyperparameter experiments on the regularization parameters and the nature of the regularization ( \\( L_1 \\) vs 
# MAGIC \\( L_2 \\) vs blend). Curiously, we found that the Lasso regularization ends up training a degenerate model where every record is classified as 'No Delay'.
# MAGIC 
# MAGIC Hyperparameters:
# MAGIC * Regularization Parameter: [0, 0.3]
# MAGIC * ElasticNet Parameter: [0.0, 0.5, 1.0] # 0: Ridge, 1: Lasso
# MAGIC 
# MAGIC ##### Decision Trees
# MAGIC 
# MAGIC ##### Random Forest
# MAGIC 
# MAGIC ##### Gradient Boosted Trees
# MAGIC 
# MAGIC ##### eXtreme Gradient Boosted Trees

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Model implementation was done in [this following notebook](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/606274953377686/command/4377823981614326).

# COMMAND ----------

from IPython.display import Image 
from IPython.core.display import HTML 
#Image(url="https://www.google.com/logos/doodles/2021/celebrating-johannes-gutenberg-6753651837109212-l.png") 

displayHTML("<img src = '/dbfs/FileStore/shared_uploads/jeffli930@berkeley.edu/w261_flowchart_1.png'>")

# COMMAND ----------

# MAGIC %md # S5 - Algorithm Implementation
# MAGIC 
# MAGIC Create your own toy example that matches the dataset provided and use this toy example to explain the math behind the algorithm that you will perform. Apply your algorithm to the training dataset and evaluate your results on the test set. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Toy Problem for Logistic Regression

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC In order to illustrate the inner workings of one of the algorithms we are working with, we will be implementing a toy example of logistic regression using gradient descent. Unlike linear regression, which uses a linear combination of terms, logistic regression applies the sigmoid function to the typical regression equation. In other words,
# MAGIC 
# MAGIC $$f(x) = \frac{1}{1 + e^{-\theta^{T}x}} $$
# MAGIC 
# MAGIC Similarly, unlike linear regression, which uses squared loss, logistic regression uses log-loss cost function. The Logloss function is below:
# MAGIC 
# MAGIC 
# MAGIC $$ J(\theta) = \frac{1}{m} \sum_{i=1}^{m} -y_{i} \times log(h_{\theta}(x_{i})) + (1 - y_{i}) \times log(1-h_{\theta}(x_{i}))$$
# MAGIC 
# MAGIC 
# MAGIC Below is the code to implement logistic regression for a very small subset of data for KORD (Chicago), for  the first quarter of 2015. I will use 3 columns - dep_del15 (dependent variable), with 2 sample scaled features.

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

## Query a small subset of data for the toy problem. Ensure that the variables will be scaled from 0-1 or else the log loss function will not work properly
df =  sqlContext.sql("""
select dep_del15, 
    fraction_delays_impute as sample_feature_scaled1, 
    outbound_count_impute/109 as sample_feature_scaled2
   from group25.data_train_main
    where call_sign_dep = "KORD"
    limit 1000
""")


## Convert the dataframe to RDD
test_rdd = df.rdd.map(list).cache()

# COMMAND ----------

test_rdd2 = test_rdd.map(lambda x: (x[0], x[1:])).cache()

# initialize set of coefficients
initialize_weights = [0.01, 0.01, 0.01]

def LogLoss(dataRDD,W):
    """
    Compute log loss error.
    
    Args:
        dataRDD - records are tuples of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    """
    #Augment the matrix
    augmentedData = dataRDD.map(lambda x: (np.append([1.0],x[1:]),x[0]))
    
    #Take the dot product, apply the sigmoid function, then run through the cost function.
    log_loss = augmentedData \
        .map(lambda x: (np.dot(x[0],W),x[1])) \
        .map(lambda x: (1 / (1 + np.exp(-x[0])),x[1])) \
        .map(lambda x: (-x[1]*np.log(x[0]) - (1-x[1])*np.log(1-x[0]))) \
          .mean()


    return log_loss

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have computed the loss function, let's proceed with iterative updates using gradient descent. The gradient can be calculated as 
# MAGIC 
# MAGIC $$ C' = x(s(z) - y) $$
# MAGIC 
# MAGIC where \\( C' \\) is the derivative of wrt weights, \\( x \\) is the feature vector, \\( s(z) \\) is the predicted value, and \\( y \\) is the class label.

# COMMAND ----------

def GDUpdate(dataRDD, W, learningRate = 0.1):
    """
    Perform one Log Loss gradient descent step/update.
    Args:
        dataRDD - records are tuples of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    Returns:
        new_model - (array) updated coefficients, bias at index 0
    """
    #Augment the matrix
    augmentedData = dataRDD.map(lambda x: (np.append([1.0],x[1:]),x[0])) \
    
    # Step 1: Take the dot product of the weights and the features, and the label
        #(dotproduct, augmented  features vector, label)
    # Step 2. Apply the sigmoid function, with the 
        # sigmoid, augmented features vector, label.
    # Step 3: Calculate the gradient.
    gradient = augmentedData.map(lambda x: (np.dot(x[0],W),x[0],x[1])) \
                  .map(lambda x: (1 / (1 + np.exp(-x[0])),x[1],x[2])) \
                  .map(lambda x: np.dot((x[0] - x[2]), x[1])).mean()

    # Multiply the learning rate against the gradient. Return new weights.
    update = W - np.multiply(gradient, initialize_weights)
    
    return update
    
GDUpdate(test_rdd2,  [0.01, 0.01, 0.01])

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let's run through 5 iterations

# COMMAND ----------

# Take a look at a few Gradient Descent steps

nSteps = 5
model = initialize_weights
print(f"BASELINE:  Loss = {LogLoss(test_rdd2,model)}")
for idx in range(nSteps):
    print("----------")
    print(f"STEP: {idx+1}")
    model = GDUpdate(test_rdd2, model)
    loss = LogLoss(test_rdd2, model)
    print(f"Loss: {loss}")
    print(f"Model: {[round(w,3) for w in model]}")

# COMMAND ----------

# MAGIC %md
# MAGIC The loss function appears to have 

# COMMAND ----------

# MAGIC %md # S6 - Conclusions
# MAGIC 
# MAGIC Report results and learnings for both the ML as well as the scalability

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # S7 - Application of Course Concepts
# MAGIC 
# MAGIC  Pick 3-5 key course concepts and discuss how your work on this assignment illustrates an understanding of these concepts.
# MAGIC  
# MAGIC  
# MAGIC   
# MAGIC ## Normalization 
# MAGIC 
# MAGIC ## Assumptions for Different Algorithms
# MAGIC 
# MAGIC ## Page Rank *  Justin
# MAGIC 
# MAGIC ## Regularization
# MAGIC 
# MAGIC ##  One Hot Encoding / vector embeddings / feature selection

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md # Appendix

# COMMAND ----------

# MAGIC %md
# MAGIC ## List of Features
# MAGIC 
# MAGIC |Column name|Description|Categorical/Continuous|
# MAGIC |-------------------|-------------------|-------------------|
# MAGIC |DEP_DEL15|Our dependent variable. Indicates whether or not the flight was delayed by 15 minutes|Categorical|
# MAGIC |NORMALIZED_OUTBOUND_COUNT_2H|Count of outbound flights 2 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_OUTBOUND_COUNT_3H|Count of outbound flights 3 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_OUTBOUND_COUNT_4H|Count of outbound flights 4 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_OUTBOUND_COUNT_5H|Count of outbound flights 5 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_OUTBOUND_COUNT_6H|Count of outbound flights 6 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_INBOUND_COUNT_2H|Count of inbound flights 2 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_INBOUND_COUNT_3H|Count of inbound flights 3 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_INBOUND_COUNT_4H|Count of inbound flights 4 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_INBOUND_COUNT_5H|Count of inbound flights 5 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_INBOUND_COUNT_6H|Count of inbound flights 6 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DIVERTED_OUTBOUND_COUNT_2H|Count of diverted outbound flights 2 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DIVERTED_OUTBOUND_COUNT_3H|Count of diverted outbound flights 3 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DIVERTED_OUTBOUND_COUNT_4H|Count of diverted outbound flights 4 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DIVERTED_OUTBOUND_COUNT_5H|Count of diverted outbound flights 5 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DIVERTED_OUTBOUND_COUNT_6H|Count of diverted outbound flights 6 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DIVERTED_INBOUND_COUNT_2H|Count of diverted inbound flights 2 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DIVERTED_INBOUND_COUNT_3H|Count of diverted inbound flights 3 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DIVERTED_INBOUND_COUNT_4H|Count of diverted inbound flights 4 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DIVERTED_INBOUND_COUNT_5H|Count of diverted inbound flights 5 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DIVERTED_INBOUND_COUNT_6H|Count of diverted inbound flights 6 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DELAY_OUTBOUND_COUNT_2H|Count of delayed outbound flights 2 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DELAY_OUTBOUND_COUNT_3H|Count of delayed outbound flights 3 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DELAY_OUTBOUND_COUNT_4H|Count of delayed outbound flights 4 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DELAY_OUTBOUND_COUNT_5H|Count of delayed outbound flights 5 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DELAY_OUTBOUND_COUNT_6H|Count of delayed outbound flights 6 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DELAY_INBOUND_COUNT_2H|Count of delayed inbound flights 2 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DELAY_INBOUND_COUNT_3H|Count of delayed inbound flights 3 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DELAY_INBOUND_COUNT_4H|Count of delayed inbound flights 4 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DELAY_INBOUND_COUNT_5H|Count of delayed inbound flights 5 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |NORMALIZED_DELAY_INBOUND_COUNT_6H|Count of delayed inbound flights 6 hours prior to the actual flight at that particular hour, normalized by the volume of traffic at an airport|Continuous|
# MAGIC |PREV_DEP_TIMEDELTA|How long was the previous flight (by tail num) delayed by, in minutes|Categorical|
# MAGIC |MONTH|Month|Categorical|
# MAGIC |DAY_OF_WEEK|Day of the week|Categorical|
# MAGIC |OP_UNIQUE_CARRIER|Unique Carrier, e.g. airline|Categorical|
# MAGIC |call_sign_dep|Call Sign, or airport of the departing airport|Categorical|
# MAGIC |DEP_LOCAL_HOUR|Departure hour|Categorical|
# MAGIC |WND_DIRECTION_ANGLE_AVG_DIR|Categorical var of the wind direction (NW, SW, SE, NE) 2 hours prior from WND, binned on airport, date and hour|Categorical|
# MAGIC |MV_THUNDERSTORM|Boolean of whether or not there was a Thunderstorm from the Weather Events 2 hours prior from MV1 or MV2, binned on airport, date and hour|Categorical|
# MAGIC |MV_SHOWERS|Boolean of whether or not there were showers from the Weather Events 2 hours prior from MV1 or MV2, binned on airport, date and hour|Categorical|
# MAGIC |MV_SAND_OR_DUST|Boolean of whether or not there was sand or dust from the Weather Events 2 hours prior from MV1 or MV2, binned on airport, date and hour|Categorical|
# MAGIC |MV_BLOWINGSNOW|Boolean of whether or not there was blowing snow from the Weather Events 2 hours prior from MV1 or MV2, binned on airport, date and hour|Categorical|
# MAGIC |MV_FOG|Boolean of whether or not there was fog from the Weather Events 2 hours prior from MV1 or MV2, binned on airport, date and hour|Categorical|
# MAGIC |AA1_DEPTH_DIMENSION_AVG|Average of the AA1 episodic rain depth dimension from the NOAA dataset 2 hours prior, binned on airport, date, and hourly |Continouous|
# MAGIC |AA1_PERIOD_QUANTITY_AVG|Average of the AA1 period quantity rain depth dimension from the NOAA dataset 2 hours prior, binned on airport, date, and hourly |Continouous|
# MAGIC |GA1_COVERAGE_CODE_AVG|Average of the GA1 cloud coverage from the NOAA dataset 2 hours prior, binned on airport, date, and hourly |Continouous|
# MAGIC |GA1_BASE_HEIGHT_DIMENSION_AVG|Average of the GA1 cloud height from the NOAA dataset 2 hours prior, binned on airport, date, and hourly |Continouous|
# MAGIC |WND_SPEED_AVG|Average of wind speed from the NOAA dataset 2 hours prior, binned on airport, date, and hourly |Continouous|
# MAGIC |TMP_AIR_TEMPERATURE_AVG|Average of temperature from the NOAA dataset 2 hours prior, binned on airport, date, and hourly |Continouous|
# MAGIC |SLP_ATMOSPHERIC_PRESSURE_AVG|Average of SLP atmospheric pressure from the NOAA dataset 2 hours prior, binned on airport, date, and hourly |Continouous|

# COMMAND ----------

