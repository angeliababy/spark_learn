#!/usr/bin/env python
# encoding: utf-8

from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession,Row
from pyspark.sql.types import *
conf = SparkConf().setMaster("local").setAppName("First_App")
sc = SparkContext(conf=conf)

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .getOrCreate()


flume_data = sc.textFile("")

xdr_data = flume_data.map(lambda line: line.split(','))

xdr_data = xdr_data.map(lambda p: Row(name=int(p[75]), salary=int(p[80])))

# Apply the schema to the RDD and Create DataFrame
swimmers = spark.createDataFrame(xdr_data)

# Creates a temporary view using the DataFrame
swimmers.createOrReplaceTempView("xdrdata")

spark.sql("select * from xdrdata limit 10").show()
# swimmers.select("id", "age").filter("age = 22").show()

spark.sql("select case when Response_Time>0 and Rcode=0 then Response_Time else 0 end from xdrdata limit 100").show()


