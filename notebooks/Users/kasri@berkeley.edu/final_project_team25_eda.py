# Databricks notebook source
# MAGIC %md # W261 Final Project - Flight Delay Prediction
# MAGIC 
# MAGIC ## Airlines Data

# COMMAND ----------

# MAGIC %md   
# MAGIC ##### Justin Trobec, Jeff Li, Sonya Chen, Karthik Srinivasan
# MAGIC ##### Spring 2021, section 5, Team 25

# COMMAND ----------

## imports
!pip install -U seaborn==0.10.1

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

print(sns.__version__)
sqlContext = SQLContext(sc)

# COMMAND ----------

# MAGIC %md
# MAGIC # __Section 1__: Intro

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
# MAGIC # __Section 2__: Schema and Meaning

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
# MAGIC 
# MAGIC # __Section 3__: Exploration Process

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Our primary goal for EDA was to understand which features might be useful as predictors for flight delays. We wrote some simple code to visualize either categorical or continuous variables against the DEP_DEL15 column, which indicates whether a flight was delayed more than 15 minutes.

# COMMAND ----------

sns.set(style="darkgrid")
sns.set_palette("colorblind")

def get_sampled_dataset(col_name, sample_size):
    return sqlContext.sql("""SELECT COALESCE(DEP_DELAY, 0) AS DEP_DELAY, COALESCE(DEP_DEL15, 0) AS DEP_DEL15, {}
                             FROM group25.airlines_utc_main TABLESAMPLE({} ROWS) WHERE year<2019""".format(col_name, sample_size)).toPandas()
  
def set_xticklabels(ax):
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
  
def plot_cont_vs_delay(col_name, sample_size=100_000):
  sampledDf = get_sampled_dataset(col_name, sample_size)
  fig, axes = plt.subplots(1, 2, figsize=(15,5))
  sns.scatterplot(x='DEP_DELAY', y=col_name, data=sampledDf, ax=axes[0])
  axes[0].set(title=f'DEP_DELAY vs {col_name}')
  
  sns.boxplot(y=col_name, x='DEP_DEL15', data=sampledDf, ax=axes[1])
  axes[1].set(title=f'DEP_DEL15 vs {col_name}')
  plt.suptitle(f'{col_name} vs Delay')

def plot_cat_vs_delay(col_name, sample_size=100_000, rename_cats=None, cat_order=None):
  sampledDf = get_sampled_dataset(col_name, sample_size)
  fig, axes = plt.subplots(1, 3, figsize=(20,5))
  
  if rename_cats:
    sampledDf[col_name] = sampledDf[col_name].apply(rename_cats)
  
  sns.stripplot(y='DEP_DELAY', x=col_name, order=cat_order, data=sampledDf, ax=axes[0])
  axes[0].set(title=f'DEP_DELAY vs {col_name}')
  
  sns.countplot(x=col_name, hue='DEP_DEL15', order=cat_order, data=sampledDf, ax=axes[1])
  axes[1].set(title=f'DEP_DEL15 vs {col_name}')
  
  normalized = sampledDf.groupby(col_name)['DEP_DEL15'].value_counts(normalize=True).unstack('DEP_DEL15').plot.bar(stacked=True, ax=axes[2])
  axes[2].set(title=f'DEP_DEL15 by {col_name} Normalized Counts')
  
  for ax in axes:
    set_xticklabels(ax)
    
  plt.suptitle(f'{col_name} vs Delay')

# COMMAND ----------

plot_cont_vs_delay('DISTANCE')

# COMMAND ----------

plot_cont_vs_delay('AIR_TIME')

# COMMAND ----------

plot_cat_vs_delay('OP_CARRIER', sample_size = 100_000)

# COMMAND ----------

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plot_cat_vs_delay('DAY_OF_WEEK', sample_size = 100_000, rename_cats = lambda x: days[x-1], cat_order=days)

# COMMAND ----------

plot_cat_vs_delay('DEP_UTC_HOUR', sample_size = 100_000)

# COMMAND ----------

plot_cat_vs_delay('DEP_LOCAL_HOUR', sample_size = 100_000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Correlation Matrix

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Time Series Exploration

# COMMAND ----------

# !pip install -U numpy>=1.15.0 scipy==1.1.0 patsy pandas statsmodels==0.11.0 altair
!pip install -U altair

# COMMAND ----------

pd.__version__
np.__version__

# COMMAND ----------

import scipy as sp
sp.__version__

# COMMAND ----------

import altair as alt
from pyspark.sql import functions as f
from pyspark.sql.functions import col, sum, avg, max, count, countDistinct, weekofyear, to_timestamp, date_format

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark
import statsmodels.api as sm

from distutils.version import LooseVersion
from pyspark.ml import Pipeline

from pandas.tseries.holiday import USFederalHolidayCalendar
import datetime

sqlContext = SQLContext(sc)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM group25.daily_delay_fraction
# MAGIC ORDER BY day

# COMMAND ----------

ts_delay_frac = sqlContext.sql('SELECT * FROM group25.daily_delay_fraction ORDER BY day')

# COMMAND ----------

ts_pd = ts_delay_frac.toPandas()

# COMMAND ----------

ts_pd.interpolate(inplace=True)

# COMMAND ----------

ts_pd = ts_pd.set_index('day')

# COMMAND ----------

ts_pd

# COMMAND ----------

indexed = ts_pd.set_index(pd.DatetimeIndex(ts_pd['day'], freq='D')).drop('day',axis=1)

# COMMAND ----------

res = sm.tsa.seasonal_decompose(indexed)
resplot = res.plot()

# COMMAND ----------

stl_dec = sm.tsa.STL(indexed).fit()
stl_dec.plot()

# COMMAND ----------

df_airlines = sqlContext.table("group25.phase_1_processed_airline")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM group25.phase_1_processed_airline TABLESAMPLE(10000 ROWS)
# MAGIC WHERE DEP_DEL15 IS NOT NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM group25.airlines_main TABLESAMPLE(50000 ROWS)
# MAGIC WHERE (DEP_DEL15 IS NOT NULL) 
# MAGIC   AND (DEP_DELAY IS NOT NULL)

# COMMAND ----------

sampledDf = sqlContext.sql("""SELECT *
FROM group25.airlines_main TABLESAMPLE(50000 ROWS)
WHERE (DEP_DEL15 IS NOT NULL) 
  AND (DEP_DELAY IS NOT NULL)""").toPandas()

# COMMAND ----------

alt.data_transformers.disable_max_rows()

def plot_cont_vs_delay(data, col_name):
  data = data[["DEP_DELAY", "DEP_DEL15", col_name]]
  sns.scatterplot(x='DEP_DELAY', data=data)
  scatter = alt.Chart(data).mark_point().encode(
    x="DEP_DELAY:Q",
    y=f'{col_name}:Q',
    tooltip=[col_name, 'DEP_DELAY', 'DEP_DEL15']
  ).properties(width=450).interactive()

  box = alt.Chart(data).mark_boxplot().encode(
    x="DEP_DEL15:O",
    y=f'{col_name}:Q'
  ).properties(width=450)
  both = alt.hconcat(scatter, box)
  both.title = f'{col_name} vs. Delay'
  return both

def plot_cat_vs_delay(data, col_name):
  dist = data[["DEP_DELAY", "DEP_DEL15", col_name]]
  scatter = alt.Chart(dist).mark_boxplot().encode(
    y="DEP_DELAY:Q",
    x=f'{col_name}:N',
    tooltip=[col_name, 'DEP_DELAY', 'DEP_DEL15']
  ).properties(width=450).interactive()

  box = alt.Chart(dist).mark_bar().encode(
    x=f'{col_name}:N',
    y="count():Q",
    color="DEP_DEL15:O"
  ).properties(width=450).properties(width=300)
  both = alt.hconcat(scatter, box)
  both.title = f'{col_name} vs. Delay'
  return both
  

# COMMAND ----------

plot_cont_vs_delay(sampledDf, "DISTANCE")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ORIGIN, count(*)
# MAGIC FROM group25.airlines_main
# MAGIC GROUP BY 1
# MAGIC ORDER BY 2 desc

# COMMAND ----------

plot_cat_vs_delay(sampledDf, "ORIGIN")

# COMMAND ----------



# COMMAND ----------

plot_cat_vs_delay(sampledDf, "DEST")

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC plot_cat_vs_delay(sampledDf, "DAY_OF_WEEK")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT count(*)
# MAGIC FROM (SELECT ORIGIN, DEST, DAY_OF_WEEK, DEP_UTC_HOUR
# MAGIC       FROM group25.airlines_utc_main 
# MAGIC       WHERE YEAR < 2018
# MAGIC       GROUP BY 1, 2, 3, 4)