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
# MAGIC - EDA
# MAGIC     - [Databricks](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/37954262501070/command/1553213441173274)
# MAGIC     - [Github](https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/notebooks/Users/kasri@berkeley.edu/final_project_team25_eda.py)
# MAGIC 
# MAGIC - Weather
# MAGIC     - [Databricks](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4377823981601721/)
# MAGIC     - [Github](https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/notebooks/Users/jeffli930%40berkeley.edu/final_project_team25_weather.py)
# MAGIC 
# MAGIC - PageRank
# MAGIC     - [Databricks](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/4377823981609021/command/4464614029139162)
# MAGIC     - [Github](https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/Assignments/final_project_team25_centrality.py)
# MAGIC 
# MAGIC - Inbound/Outbound
# MAGIC     - [Databricks](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/439895120630397/command/439895120630400)
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
# MAGIC * [Slides](https://docs.google.com/presentation/d/12L61_t3-HJnCUst95Y8bLuk15Xo07VXh3N7LWhKK9Lo/edit?usp=sharing)

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
# MAGIC 
# MAGIC [review] - Karthik 

# COMMAND ----------

# MAGIC %md 

# COMMAND ----------

# MAGIC %md # S2 - EDA & Discussion of Challenges
# MAGIC 
# MAGIC Determine a handful of relevant EDA tasks that will help you make decisions about how you implement the algorithm to be scalable. Discuss any challenges that you anticipate based on the EDA you perform.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC As part of the project, we were given access to a dataset consisting of US flights from the years 2015-2019. We generally worked with data in SQL format, using the databricks tables as that made it relatively easy to checkpoint our datasets and seemed less cumbersome than working with pyspark. We loaded the flight data into a table and looked at some basic stats.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT 'All' as Year, count(*)
# MAGIC FROM group25.airlines_main
# MAGIC UNION
# MAGIC SELECT YEAR, count(*)
# MAGIC FROM group25.airlines_main
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1;

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The airlines data is comprised of approximately 31M flights, and the number of flights seems to be increasing year-over-year. We suspect if we had more recent data, we would have seen a dip starting in 2020 as lockdowns for the Covid pandemic began in the US. We realized up front that we would run into challenges with leackage, which we will discuss more in depth later. For now, we will exclude 2019 from EDA, as that will be our holdout set for test.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Schema & Meaning

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DESCRIBE group25.airlines_main

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Looking at the schema for the table gave us some initial sense of which data might be continuous (e.g. often, but not always, `int` fields) versus categorical (`string` fields).
# MAGIC We used [the provided codebook](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ) to better understand the fields in the table. One of the first challenges we faced was getting data into UTC format, since the codebook indicated that times in the original dataset were local times relative to the airport. In order to convert these into UTC, we first had to find additional data telling us what timezone the airport was in. For this, we found [Airport codes and timezones](https://openflights.org/data.html) from the OpenFlights project.
# MAGIC 
# MAGIC This data was joined to the flights data and combined with the data in the flights field to generate a new table with UTC timings. This was particularly necesary for joining with the weather datasets, which will be discussed in a section of its own later in this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploration Process

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Our primary goal for EDA was to understand which features might be useful as predictors for flight delays. We wrote some simple code to visualize either categorical or continuous variables against the DEP_DEL15 column, which indicates whether a flight was delayed more than 15 minutes. We created a couple of simple methods to plot categorical and continuous variables in a uniform way. We'll visualize a few interesting examples below, with a more complete set to be found in the EDA notebook.

# COMMAND ----------

def get_sampled_dataset(col_name, sample_size):
    return sqlContext.sql("""SELECT * FROM (SELECT COALESCE(DEP_DELAY, 0) AS DEP_DELAY, COALESCE(DEP_DEL15, 0) AS DEP_DEL15, {}
                             FROM group25.airlines_utc_main WHERE year<2019) TABLESAMPLE({} ROWS)""".format(col_name, sample_size)).toPandas()
  
def set_xticklabels(ax):
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
  
def plot_cont_vs_delay(col_name, sample_size=100_000, df=None):
  sns.set(style="darkgrid")
  sns.set_palette("colorblind")

  sampledDf = df if (df is not None) else get_sampled_dataset(col_name, sample_size)
  fig, axes = plt.subplots(1, 3, figsize=(20,5))
  sns.scatterplot(x='DEP_DELAY', y=col_name, data=sampledDf, ax=axes[0])
  axes[0].set(title=f'DEP_DELAY vs {col_name}')
  
  sns.boxplot(y=col_name, x='DEP_DEL15', data=sampledDf, ax=axes[1], width=0.4)
  axes[1].set(title=f'DEP_DEL15 vs {col_name}')
  
  sns.boxplot(y=col_name, x='DEP_DEL15', data=sampledDf, ax=axes[2], showfliers=False, width=0.4)
  axes[2].set(title=f'DEP_DEL15 vs {col_name} - No outliers')
  plt.suptitle(f'{col_name} vs Delay')


def plot_cat_vs_delay(col_name, sample_size=100_000, rename_cats=None, cat_order=None, df = None):
  sns.set(style="darkgrid")
  sns.set_palette("colorblind")

  sampledDf = df if (df is not None) else get_sampled_dataset(col_name, sample_size)
  fig, axes = plt.subplots(1, 4, figsize=(25,5))
  
  if rename_cats:
    sampledDf[col_name] = sampledDf[col_name].apply(rename_cats)
  
  sns.stripplot(y='DEP_DELAY', x=col_name, order=cat_order, data=sampledDf, ax=axes[0])
  axes[0].set(title=f'DEP_DELAY vs {col_name} - Distributions')
  
  sns.boxplot(y='DEP_DELAY', x=col_name, order=cat_order, showfliers=False, data=sampledDf, ax=axes[1])
  axes[1].set(title=f'DEP_DELAY vs {col_name} - Distributions w/o Outliers')
  
  sns.countplot(x=col_name, hue='DEP_DEL15', order=cat_order, data=sampledDf, ax=axes[2])
  axes[2].set(title=f'DEP_DEL15 vs {col_name} - Class Counts')
  
  normalized = sampledDf.groupby(col_name)['DEP_DEL15'].value_counts(normalize=True).unstack('DEP_DEL15').plot.bar(stacked=True, ax=axes[3])
  axes[3].set(title=f'DEP_DEL15 by {col_name} - Normalized Counts')
  
  for ax in axes:
    set_xticklabels(ax)
    
  plt.suptitle(f'{col_name} vs Delay')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We looked at correlations in the data to understand relationships between certain features. For example, this plot shows correlations between many of the continuous features in the flights data:

# COMMAND ----------

sns.set(style='white')
cont_features = [
  'DEP_DELAY', 
  'ARR_DELAY', 
  'TAXI_IN', 
  'TAXI_OUT', 
  'WHEELS_ON', 
  'WHEELS_OFF', 
  'CRS_ELAPSED_TIME'
  ,'ACTUAL_ELAPSED_TIME'
  ,'AIR_TIME'
  ,'DISTANCE'
  ,'CARRIER_DELAY'
  ,'WEATHER_DELAY'
  ,'NAS_DELAY'
  ,'SECURITY_DELAY'
  ,'LATE_AIRCRAFT_DELAY'
  ,'FIRST_DEP_TIME'
  ,'TOTAL_ADD_GTIME'
  ,'LONGEST_ADD_GTIME'
  ,'DIV_AIRPORT_LANDINGS']

# Generate a large random dataset
feat_data = sqlContext.sql("""SELECT {} FROM (SELECT * FROM group25.airlines_utc_main WHERE year<2019) TABLESAMPLE({} ROWS)""".format(','.join(cont_features), 1_000_000)).toPandas()

# Compute the correlation matrix
corr = feat_data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 18))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Looking at this chart helps us identify features that are highly-colinear and probably would not make sense to include together in our models. For example, `ACTUAL_ELAPSED_TIME` and `AIR_TIME` have a correlation coefficient of 0.99 and therfore probably do not both need to be included in set of model features we use.
# MAGIC 
# MAGIC Further, we see some interesting relationships to the outcome variable, for example that distance and delay times are not highly correlated, which might suggest that the length of a flight is not particularly helpful in understanding the chances it might be delayed. We can look more closely at that relationship using our plotting methods 

# COMMAND ----------

plot_cont_vs_delay('DISTANCE')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC On the otherhand, many of the features in the dataset are not actually useful for prediction because they would not be available at the time of prediciton. Obviously, we cannot include arrival time in predicting departure delays. So many of these features were ruled out on that basis.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Time Based Features

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We looked at the daily aggregate fraction of 15-minute delayed flights and noticed some very clear patterns. The winter holiday season has a clear spike, and there also is an increase in the summer months, corresponding to the US vacation season. One possibility this suggested to us was running a time series model (e.g. Seasonal ARIMA) and using the output as a feature. This fell to low in our priority list to address, so we leave that for further work.

# COMMAND ----------

daily_delay_frac= spark.sql("SELECT * FROM group25.daily_delay_fraction WHERE year(day) < 2019 ORDER BY day").toPandas()
fig, ax = plt.subplots(1, 1, figsize=(20,7))
sns.lineplot(y='fraction_delayed', x = 'day', data=daily_delay_frac)
ax.set(xlabel="Day", ylabel="Fraction Delayed")
plt.suptitle('Fraction of Flights Delayed per Day')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC There is an interesting pattern in the impact of local and UTC hours on delays. For the continuous version of delays, we end up with a clear gap in the distribution that shows up in the strip plot (on the left) below, and is consistent across hours. Additionally we see that a larger percentage of flights are delayed past 15 minutes in the local evening hours, between 7 and 10PM.

# COMMAND ----------

plot_cat_vs_delay('DEP_LOCAL_HOUR', sample_size = 100_000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Somewhat surprisingly, we did not observe as much signal coming from the day of week. It seems that time of day is more predictive of delays than day of week:

# COMMAND ----------

plot_cat_vs_delay('DAY_OF_WEEK', sample_size = 100_000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In addition to the flights data, we also had access to a dataset with weather reports. Working with this dataset was a complicated process involving its own EDA, and we will discuss that process more in the Feature Engineering - Weather section of this report.

# COMMAND ----------

# MAGIC %md # S3 - Feature Engineering
# MAGIC 
# MAGIC Apply relevant feature transformations, dimensionality reduction if needed, interaction terms, treatment of categorical variables, etc.. Justify your choices.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Pipeline Breakdown
# MAGIC A high level overfiew of our overall feature engineering process can be found below. We will cover some of the interesting feature engineering topics from this pipeline.

# COMMAND ----------

displayHTML("<img src = 'https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/images/w261%20flowchart%201.png?raw=true'>")

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
# MAGIC Ultimately, we settled on an hourly granularity. While this was partially based on consensus and intuition, it is a nice round unit of time that is widely understood and helps to make this analysis easier to interpret. However, more granular binnings of time, such as by 5 minute increments, could possibly yield better results.
# MAGIC 
# MAGIC In terms of how far in advance we would have this information, we decided to try and gather a set of features for each airport 2 hours in advance. For instance, what was the weather 2 hours prior to the scheduled takeoff? What was the status of delays and divered flights at the outgoing airport?
# MAGIC 
# MAGIC The challenge with this approach was that not all of our data was formatted correctly. While the weather data was given in a standardized UTC format, the flight data was not. Ensuring that our dataset utilized consistent time formats across the board was important to predicting delays 2 hours out. The datetime format is critical for several niche problems with joining on data from 2 hours prior. For instance, what if you want flight data 2 hours prior to a 1 am flight? You cannot join simply on FL_Date and a hour granularity. Furthermore, once you start working with timezones and daylight savings, things get especially complicated. A good rule of thumb for sanity is to generate all of your datetime formats first as upstream as possible, before applying any interval functions. Any downstream code or engineered features benefit immensely from this approach.
# MAGIC 
# MAGIC In terms of grabbing the time zone data per airport, we utilized the OpenFlights data. In order to ensure proper matching with our dataset, we utilized timezone data from [OpenFlights](https://openflights.org/data.html).
# MAGIC 
# MAGIC Spark has some limited inbuilt capabilities when it comes to handling timezones, so we leveraged more commonly used datetime python libraries. The issue with this is that you have to run your data through a Spark UDF to leverage the datetime library at scale. While doing this is not the most performant Spark operation, it did work with the scale of data we had in a reasonable amount of time.

# COMMAND ----------

displayHTML("<img src = 'https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/images/w261_flowchart2.png?raw=true'>")

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
# MAGIC 
# MAGIC #### Diagram
# MAGIC 
# MAGIC A diagram highlighting the flow of how the weather data was processed and subsequently joined to the data is provided below.

# COMMAND ----------

displayHTML("<img src = 'https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/images/Weather_flow_diagram.png?raw=true'>")

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Page Rank

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Airline flight data forms a natural graph in some obvious ways. Airports are clearly connected by flights, so we can consider airports as verticies and flights between them as edges. We could make the vertices more complex by addtionally segmenting them by airline or time frame. We could also invert the graph, treat vertices as edges that connect an incoming flight to an outgoing flight. These approaches can be mixed in many ways, some of which may be computationally prohibitive.
# MAGIC 
# MAGIC For any of these graphs, we might compute the centrality of its vertices using various scores. PageRank is one such score, and it tells us what proportion time a random surfer might spend at a given node if they were doing an infinite walk. In the airline cone

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inbound/Outbound Flights

# COMMAND ----------

# MAGIC %md ### Inbound & outbound related dataframes computation flowcharts
# MAGIC 
# MAGIC - This is the flowchart of how we computed the sets of inbound and outbound features. 
# MAGIC - We first computed 1st iteration and raw statistics of inbound and outbound counts. 
# MAGIC - Then we computed the 2nd iteration of normalized inbound and outbound counts with the max_median.
# MAGIC - Please refer to below diagram.

# COMMAND ----------

displayHTML("<img src = https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/images/Inbound_Outbound_dataframes_v2.png?raw=true' style='height:740px;width:1000px'>  ")

# COMMAND ----------

# MAGIC %md ### Maximum Median Compute Flowchart
# MAGIC 
# MAGIC How to compute Maximum Median which is used for normalization? 
# MAGIC - Step1: we comute the houlry outbound/inbound counts for each airporlt/flight_date/hour.
# MAGIC - Step2: we compute the median inbound/outbound number for each airport at each hour (Say median inbound for all 10am-11am periods for airportA). 
# MAGIC - Step3: we find the maximum among all the medians for that particular airport.

# COMMAND ----------

displayHTML("<img src = 'https://github.com/kasri-mids/ucb-w261-sp2021-team25/blob/main/images/compute_max_median_for_inbound_and_outbound.png?raw=true' style='height:740px;width:1000px'> '' ")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Notebook for Inbound & Outbound Features: 
# MAGIC For code, please refer to: https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/439895120630397/command/439895120630399
# MAGIC 
# MAGIC ### Hypothesis & background
# MAGIC - When we first approached the flight delay problem, we thought that inbound flights and outbound flights should be great features for predicting the flights delay or not. 
# MAGIC - The hypothesis behind that is busier airports could be more likely to have flights delay due to capacity constraints, tighter resources, and consequent congestion. All these reasons are likely to cause flights delay.
# MAGIC 
# MAGIC ### 1st Iteration: 
# MAGIC - So we compute related inbounds and outbound features: 
# MAGIC   - For outbound features, we calculated:
# MAGIC     - outbound_counts /airport/flight date/departure local hour
# MAGIC     - diverted outbound_counts /airport/flight date/departure local hour
# MAGIC     - delay outbound_counts /airport/flight date/departure local hour
# MAGIC   - For inbound features, we calculated:
# MAGIC     - inbound_counts /airport/flight date/arr local hour
# MAGIC     - diverted inbound_counts /airport/flight date/arr local hour
# MAGIC     - delay inbound_counts /airport/flight date/arr local hour
# MAGIC 
# MAGIC #### Problem
# MAGIC However, when we plotted out the features against delay vs non-delay, we noticed that there wasn’t much difference for these inbound and outbound related features between delayed flights and non-delayed flights. Thus, these features were not quite useful at a first glance.
# MAGIC 
# MAGIC #### Analysis of Problem
# MAGIC Later, we further hypothesized that it could be the problem with normalization. The underlying reason is that each airport has different capacities. For instance, 100 outbound flights could be close to the maximum capacity of airportA but the same number could just be close to half of the capacity of airportB (which could have a much bigger capacity than AirportA). 
# MAGIC 
# MAGIC Another problem with raw statistics of inbound and outbound count is that airports might have increased maximum capacity due to renovation during the years between 2015-2020. So an absolute raw number would not take into account whether an airport is busy or not. An airport might have increased its capacity by say 20% after a renovation, such that 100 outbound/hour could be an amount that could cause congestion and delay flights before renovation, yet the same number might be an easy task for the same airport. And we didn’t take into account of this with the raw statistics.
# MAGIC 
# MAGIC So we went to search around for data that could tell us about each airport's capacity each year. However, we didn't find an ideal dataset.
# MAGIC 
# MAGIC #### Solution: Max Median
# MAGIC So finally, we decided to use max median inbound and max median outbound to normalize the raw inbound and outbound data.
# MAGIC 
# MAGIC The idea and reason behind that is we want to find a number that is somewhat approximate maximum capacity under the condition we don’t have any information of renovation. Here are some reasons:
# MAGIC Maximum hourly outbound/inbound number might be squeezed any statistics that happened before the renovation to a very small number. Thus we might get lots of incorrectly scaled number. 
# MAGIC Median hourly outbound/inbound might not approximate maximum capacity. 
# MAGIC Thus, to balance, we picked the maximum median, which is the maximum among the medians outbound/inbound for each hour for each airport.
# MAGIC 
# MAGIC #### How to calculate Maximum Median
# MAGIC How to compute Maximum Median which is used for normalization? 
# MAGIC - Step1: we comute the houlry outbound/inbound counts for each airporlt/flight_date/hour.
# MAGIC - Step2: we compute the median inbound/outbound number for each airport at each hour (Say median inbound for all 10am-11am periods for airportA). 
# MAGIC - Step3: we find the maximum among all the medians for that particular airport.
# MAGIC 
# MAGIC #### Code:
# MAGIC code for computing max_median_inbound & max_median_outbound, please refer to here: https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/439895120630397/command/439895120630403
# MAGIC 
# MAGIC #### Data Leakage: 
# MAGIC - For preventing data-leakage, we calculate the median for inbound counts and outbound bounds with data ONLY in 2015-2017, which is the same period as our training dataset. 
# MAGIC - The reason we use only training data (data in 2015-2017) to calculate median is because our model shouldn't know any data from the future. 
# MAGIC - We calculate the maximum median based on data from 2015-2017, and we use that to normalize all the data from 2015-2020. 
# MAGIC 
# MAGIC ### 2nd Iteration: 
# MAGIC - So in the 2nd iteration, we normalized the sets of features in the 1st iteration. We calculate the normalized inbound/outbound related features by dividing the raw statistics by max_median for that corresponding airport
# MAGIC - For outbound features, we calculated:
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
# MAGIC - To account for the time series-aspect of the airline data, we also calculate something we called as lag features. 
# MAGIC - These features ends like feature_name_xH (for example: NORMALIZED_OUTBOUND_COUNT_2H)
# MAGIC - For instance: for a flight that departs at 11am at AirportA that has NORMALIZED_OUTBOUND_COUNT_2H = 20, means that normalized outbound counts from 8am-9am at airportA (1 hour timeframe, 2 hours prior the departure time) .
# MAGIC 
# MAGIC - Below are some example lag features for inbound & outbound features. We done this for all 3 sets of inbound/outbound features: inbound/outbound, diverted inbound/outobund, delay inbound/outbound. 
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
# MAGIC - diverted flights will increase the traffic of the airport unexpectedly, and thus might potentially cause other flights at that airport to delay.
# MAGIC - For each airport / day / hour, we calculated the number of diverted flights at that airport
# MAGIC - When predicting flight delay, we take into account of the impact of diverted flight numbers at that airport

# COMMAND ----------

# MAGIC %md ###Normalization
# MAGIC 
# MAGIC - Use like the normalized Inbounds & Outbound features above, we normalized this feature with the max_median inbound/outbound counts of each airport. 

# COMMAND ----------

# MAGIC %md ### Rolling Windows
# MAGIC 
# MAGIC - Just like the discussion above at the normalized inbound/outbound, we also alculated Rolling Windowed Features for diverted flights (diverted inbound/outbound): eg 2_hr_prior, 3_hr_prior, …

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

# MAGIC %md ###Normalization
# MAGIC 
# MAGIC - Use like the normalized Inbounds & Outbound features above, we normalized this feature with the max_median inbound/outbound counts of each airport. 

# COMMAND ----------

# MAGIC %md ### Rolling Windows
# MAGIC 
# MAGIC - Just like the discussion above at the normalized inbound/outbound, we also alculated Rolling Windowed Features for diverted flights (diverted inbound/outbound): eg 2_hr_prior, 3_hr_prior, …

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
# MAGIC In addition to creating this imputation approach, we carefully examined each one of our attributes prior to imputation. In some scenarios, where the values were highly populated with null values, we decided to either impute 0's instead, or drop the data outright. Furthermore, we performed EDA on the features with high nulls.

# COMMAND ----------

nulls_for_each_column_pandas = sqlContext.sql("""select * from group25.nulls_for_each_column_prior_to_impute""").toPandas().transpose()

nulls = nulls_for_each_column_pandas.reset_index().sort_values(by=0, ascending = False)

nulls['Percentage of Records Missing'] = nulls[0] / (nulls[nulls['index'] == 'Total_Records'][0].values[0])

nulls.rename(columns = {'index':'Feature', 0: 'Num_of_Nulls'}, inplace = True)

plt.figure(figsize=(30,12))
plt.xticks(rotation=90)
plt.xlabel('Feature', fontsize = 15)
plt.ylabel('Percentage of Records Missing', fontsize = 20)
plt.title("Percentage of Nulls In Our Joined Data for Each Attribute, Prior to Imputation", fontsize = 30)
g = sns.barplot(x='Feature', y="Percentage of Records Missing", data=nulls)
g.set(ylim=(None, None))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC [review]
# MAGIC 
# MAGIC #### [review] Describe how we prevent leakage
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

# COMMAND ----------

# MAGIC %md ## Data Leakage
# MAGIC 
# MAGIC ### Why Prevent Data Leakage
# MAGIC Data leakage is a critical part of ensuring our Machine Learning pipeline is as robust as possible. It is important to prevent data leakage when training the model because we want our model to predict well when it meets unseen data in a real world scenario. 
# MAGIC 
# MAGIC To ensure that our model doesn't predict delays/not-delay based on future information, we handle data with below approaches. 
# MAGIC 
# MAGIC ### Split Train/Test/Validation
# MAGIC The data that is given is from 2015-2019. A key concern of ours was how to prevent leakage, or val or test datasets leaking into our test dataset. To set up our data pipeline to prevent leakage, we set up train, test, and validation training sets:
# MAGIC 
# MAGIC - Train - 01/01/2015 - 12/31/2017
# MAGIC - Validation - 01/01/2018 - 12/31/2018
# MAGIC - Test - 01/01/2019 - 12/31/2019
# MAGIC 
# MAGIC ### Inbound & Outbound Statistics
# MAGIC - For the Inbound and Outbound related features, to prevent data leakage, so we connect the flights 
# MAGIC - To prevent data-leakage, when calculating the median for inbound counts and outbound bounds, we only use data in 2015-2017 to calculate the median.
# MAGIC - The reason we use only training data (data in 2015-2017) to calculate median is because our model shouldn't know any data from the future. 
# MAGIC - So we calulate median based on data from 2015-2017, and used that to normalized all the data from 2015-2020. 
# MAGIC 
# MAGIC ### Time Series Aspect of Inbound & Outbound Features
# MAGIC - To account for the time series-aspect of the airline data, we also calculate something we called as lag features. 
# MAGIC - These features ends like faeture_name_xH (for example: NORMALIZED_OUTBOUND_COUNT_2H)
# MAGIC - For example NORMALIZED_OUTBOUND_COUNT_2H for a flight departs at 11am at AirportA, that means that normalized outbound counts from 8am-9am (1 hour interval, 2 hours prior the departure time) at airpotA.
# MAGIC - Same theory applies for the inbound features. 
# MAGIC 
# MAGIC ### Weather Data
# MAGIC - To prevent data leakage, we only use weather data 2 hour prior the departure time of a particular flight
# MAGIC - For example, if a flight's departure date is 1/1/2016 10am, then we only use weather feature up until 1/1/2016 8am to predict whether its delay
# MAGIC 
# MAGIC ### Page Rank
# MAGIC - We only use data from 2015-2017 to compute pagerank features. 

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
# MAGIC We split the data into train, validation and test sets. We used data between 2015-2017 for train, 2018 for validation and 2019 for test. For all models discussed here, we use 5-fold cross validation on the train set. We perform a parameter grid search on the train set and evaluate the candidate models on the validation set. Since our dataset is imbalanced, i.e., the number of delays is much smaller than the number of undelayed fights, \\( \approx 4 \\) times smaller, we use the sampling techniques discussed below to address the imbalance. 
# MAGIC 
# MAGIC All sampling experiments are only conducted on the train data. We chose 1,000,000 train records for training on a wide class of ML algorithms.
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
# MAGIC Model implementation was done in [this following notebook](https://dbc-c4580dc0-018b.cloud.databricks.com/?o=8229810859276230#notebook/606274953377686/command/4377823981614326).
# MAGIC 
# MAGIC For all the models discussed below, we ran the training exercise on randomly selected 1,000,000 records. We fine-tune the models on this smaller train set for faster model down selection and eventually run the preferred candidates on a larger 5,000,000 record dataset. We compared various models using the \\( F_{0.5} \\) score. We take the viewpoint of the Airport/Airline that is looking to have more precision in the predictions. Any decisions made on the delay predictions are likely to have cascading consequences in the origin airport as well as the destination.
# MAGIC 
# MAGIC Summary:
# MAGIC * Our best model was found to be the eXtreme Gradient Boosted Model.
# MAGIC * We need to address the data imbalance. Without data balancing, the model performance was unstable. Class-weighting is the easiest option available.

# COMMAND ----------

df_plot = sqlContext.sql(""" select * from group25.experiment_results_new where Prefix = 'Val' order by pr_AUC desc, f0p5_score desc""").cache().toPandas()
df_plot.loc[:,"f0p5_score"] = df_plot.f0p5_score.astype("float")
df_plot.loc[:,"pr_AUC"] = df_plot.pr_AUC.astype("float")

fig, axes = plt.subplots(1,2, figsize=(15,7))
sns.set(font_scale = 2)
sns.boxplot(data=df_plot, x="Model", y="f0p5_score", ax=axes[0])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, horizontalalignment='right')
sns.boxplot(data=df_plot, x="Model", y="pr_AUC", ax=axes[1])
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, horizontalalignment='right')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Null Majority Class
# MAGIC 
# MAGIC We treat the majority class prediction as the null baseline model. The validation dataste has 100,000 records with \\( \approx 17.9\% \\) of them belonging to the majority 'No Delay' class. This results in
# MAGIC * Baseline PR-AUC of 0.179
# MAGIC * Baseline \\( F_{0.5} \\) of 0.179

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Logistic Regression
# MAGIC 
# MAGIC We ran hyperparameter experiments on the regularization parameters and the nature of the regularization ( \\( L_1 \\) vs 
# MAGIC \\( L_2 \\) vs blend). 
# MAGIC 
# MAGIC The regularization is defined through the loss function
# MAGIC 
# MAGIC $$ L(w | \alpha, \lambda) := \alpha \lambda || w ||_1 + (1 - \alpha)(\frac{\lambda}{2} || w ||_2^2), \; \alpha \in [0,1], \lambda \ge 0 $$
# MAGIC 
# MAGIC where \\( \alpha \\) and \\( \lambda \\) are regularization parameters and \\( w \\) are the weights.
# MAGIC 
# MAGIC Observations:
# MAGIC * We achieved a \\( F_{0.5} \\) score of 0.543 on the validation dataset (100,000 samples).
# MAGIC * To our surprise, the best Logistic Regression model, as evaluated on the validation set, turned out to be the one with no regularization and no oversampling/class-weighting.
# MAGIC 
# MAGIC Hyperparameters:
# MAGIC * Regularization Parameter: [0, 0.3]
# MAGIC * ElasticNet Parameter: [0.0, 0.5, 1.0] # 0: Ridge, 1: Lasso

# COMMAND ----------

display(sqlContext.sql(""" select * from group25.experiment_results_new where Model = 'LogisticRegression' and Prefix = 'Val' order by pr_AUC desc, f0p5_score desc""").head(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Decision Trees
# MAGIC In this work, we used gini impurity to split the nodes in the tree. We considered various depth of trees as the hyperparameter.
# MAGIC 
# MAGIC Observations:
# MAGIC * We achieved a \\( F_{0.5} \\) score of 0.611 on the validation dataset (100,000 samples).
# MAGIC * As expected the performance of the model improved as the depth of the tree increased. However, we stopped at a depth of 10 to avoid overfitting.
# MAGIC 
# MAGIC Hyperparameters:
# MAGIC * Max Depth of tree: [5, 10]

# COMMAND ----------

print('Precision-Recall AUC vs F-0.5 score at various hyperparameter runs')
display(sqlContext.sql(""" select * from group25.experiment_results_new where Model = 'DecisionTrees' and Prefix = 'Val' order by pr_AUC desc, f0p5_score desc""").head(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Random Forest
# MAGIC 
# MAGIC A random forest is an ensemble method that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
# MAGIC We further control overfitting by parametrizing the depth of the tree and the number of trees in the ensemble.
# MAGIC 
# MAGIC Observations:
# MAGIC * We achieved a \\( F_{0.5} \\) score of 0.548 on the validation dataset (100,000 samples).
# MAGIC * The performance of the models saturated as the results from using 500 trees did not yield different results from an ensemble of 1000 trees.
# MAGIC * We also observed that not using some measure of oversampling or class-weighting resulted in degenerate models.
# MAGIC * Feature importances reveal that status of the previous journey of the scheduled airplane (TAIL_NUM) is the highest contributor to the variance in the delay data.
# MAGIC 
# MAGIC Hyperparameters:
# MAGIC * Max depth of tree: [3, 4, 5]
# MAGIC * Number of trees: [100, 500, 1000]
# MAGIC 
# MAGIC |idx|Feature Name|Importance|Category|
# MAGIC |-----|-------------------|-------------------|-------------------|
# MAGIC |456|PREV_DEP_DELAY_scaled_0|0.1|Delay Propagation|
# MAGIC |402|PREV_DEP_DELAY_BOOLclassVec_Delayed|0.1|Delay Propagation|
# MAGIC |401|PREV_DEP_DELAY_BOOLclassVec_Not_Delayed|0.09|Delay Propagation|
# MAGIC |430|NORMALIZED_DELAY_OUTBOUND_COUNT_2H_IMPUTE_scaled_0|0.07|Real-time Airport Delays|
# MAGIC |431|NORMALIZED_DELAY_OUTBOUND_COUNT_3H_IMPUTE_scaled_0|0.06|Real-time Airport Delays|
# MAGIC |435|NORMALIZED_DELAY_INBOUND_COUNT_2H_IMPUTE_scaled_0|0.05|Real-time Airport Delays|
# MAGIC |436|NORMALIZED_DELAY_INBOUND_COUNT_3H_IMPUTE_scaled_0|0.05|Real-time Airport Delays|
# MAGIC |432|NORMALIZED_DELAY_OUTBOUND_COUNT_4H_IMPUTE_scaled_0|0.05|Real-time Airport Delays|
# MAGIC |437|NORMALIZED_DELAY_INBOUND_COUNT_4H_IMPUTE_scaled_0|0.04|Real-time Airport Delays|
# MAGIC |433|NORMALIZED_DELAY_OUTBOUND_COUNT_5H_IMPUTE_scaled_0|0.04|Real-time Airport Delays|
# MAGIC |434|NORMALIZED_DELAY_OUTBOUND_COUNT_6H_IMPUTE_scaled_0|0.03|Real-time Airport Delays|
# MAGIC |414|NORMALIZED_OUTBOUND_COUNT_6H_IMPUTE_scaled_0|0.03|Airport Traffic/Capacity|
# MAGIC |438|NORMALIZED_DELAY_INBOUND_COUNT_5H_IMPUTE_scaled_0|0.03|Real-time Airport Delays|
# MAGIC |439|NORMALIZED_DELAY_INBOUND_COUNT_6H_IMPUTE_scaled_0|0.03|Real-time Airport Delays|
# MAGIC |411|NORMALIZED_OUTBOUND_COUNT_3H_IMPUTE_scaled_0|0.03|Airport Traffic/Capacity|
# MAGIC |412|NORMALIZED_OUTBOUND_COUNT_4H_IMPUTE_scaled_0|0.03|Airport Traffic/Capacity|
# MAGIC |413|NORMALIZED_OUTBOUND_COUNT_5H_IMPUTE_scaled_0|0.02|Airport Traffic/Capacity|
# MAGIC |455|PR_FLDH_scaled_0|0.02|Airport Relative Importance|
# MAGIC |440|PREV_DEP_TIMEDELTA_IMPUTE_scaled_0|0.01|Delay Propagation|
# MAGIC |419|NORMALIZED_INBOUND_COUNT_6H_IMPUTE_scaled_0|0.01|Airport Traffic/Capacity|
# MAGIC |410|NORMALIZED_OUTBOUND_COUNT_2H_IMPUTE_scaled_0|0.01|Airport Traffic/Capacity|
# MAGIC |418|NORMALIZED_INBOUND_COUNT_5H_IMPUTE_scaled_0|0.01|Airport Traffic/Capacity|
# MAGIC |417|NORMALIZED_INBOUND_COUNT_4H_IMPUTE_scaled_0|0.01|Airport Traffic/Capacity|
# MAGIC |363|DEP_LOCAL_HOURclassVec_17|0.01|Peak Hour|

# COMMAND ----------

display(sqlContext.sql(""" select * from group25.experiment_results_new where Model = 'RandomForest' and Prefix = 'Val' order by pr_AUC desc, f0p5_score desc""").head(30))

# COMMAND ----------

display(sqlContext.sql(""" select * from group25.experiment_results_new where Model = 'RandomForest' and Prefix = 'Val' order by pr_AUC desc, f0p5_score desc""").head(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Gradient Boosted Trees
# MAGIC Hyperparameters:
# MAGIC * Max depth of tree: [3, 4, 5]
# MAGIC * Number of trees: [3, 7, 11]

# COMMAND ----------

display(sqlContext.sql(""" select * from group25.experiment_results_new where Model = 'GradientBoostedTrees' and Prefix = 'Val' order by pr_AUC desc, f0p5_score desc""").head(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### eXtreme Gradient Boosted Trees
# MAGIC Hyperparameters:
# MAGIC * Max depth of tree: [4, 8, 12]
# MAGIC * Learning Rate: [0.001, 0.01, 0.1]

# COMMAND ----------

display(sqlContext.sql(""" select * from group25.experiment_results_new where Model = 'eXtremeGradientBoostedTrees' and Prefix = 'Val' order by pr_AUC desc, f0p5_score desc""").head(30))

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
# MAGIC ## Optimal model
# MAGIC 
# MAGIC Apply your algorithm to the training dataset and evaluate your results on the test set.
# MAGIC 
# MAGIC We chose the eXtreme Gradient Boosted model algorithm to train on a larger train data set and test on the hold=out test set. The model was trained on 5,000,000 train records and tested on 500,000 hold-out samples.
# MAGIC The overall results are:

# COMMAND ----------

# MAGIC %md # S6 - Conclusions
# MAGIC 
# MAGIC Report results and learnings for both the ML as well as the scalability

# COMMAND ----------

# MAGIC %md
# MAGIC ## Challenges & Learnings
# MAGIC 
# MAGIC Challenges:
# MAGIC 
# MAGIC - 
# MAGIC 
# MAGIC ## Areas for Improvement
# MAGIC 
# MAGIC - Look for additional features
# MAGIC - Experiment with additional time lags

# COMMAND ----------

 [review]



[Model]

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

# MAGIC %md # Some reference publications & websites:
# MAGIC 1) 2017. A Review on Flight Delay Prediction. Alice Sternberg, Jorge de Abreu Soares, Diego Carvalho, Eduardo S. Ogasawara
# MAGIC 
# MAGIC 2) 2019. A Data Mining Approach to Flight Arrival Delay Prediction for American Airlines. Navoneel Chakrabarty
# MAGIC 
# MAGIC 3) 2019. Development of a predictive model for on-time arrival flight of airliner by discovering correlation between flight and weather data. Noriko Etani.
# MAGIC 
# MAGIC 4) https://stat-or.unc.edu/wp-content/uploads/sites/182/2018/09/Paper3_MSOM_2012_AirlineFlightDelays.pdf

# COMMAND ----------

