#!/usr/bin/env python
# encoding: utf-8

from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession
from pyspark.sql.types import *
conf = SparkConf().setMaster("local").setAppName("First_App")
sc = SparkContext(conf=conf)

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .getOrCreate()

# Generate our own CSV data
#   This way we don't have to access the file system yet.
# stringCSVRDD = sc.parallelize([(123, 'Katie', 19, 'brown'), (234, 'Michael', 22, 'green'), (345, 'Simone', 23, 'blue')])
user_data = sc.textFile(r'D:\projects\sparklearn_data\ml-100k\u.user')

user_fields = user_data.map(lambda line: line.split('|'))
# The schema is encoded in a string, using StructType we define the schema using various pyspark.sql.types
schemaString = "id name age eyeColor"
schema = StructType([
    StructField("id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("age", StringType(), True),
    StructField("eyeColor", StringType(), True),
    StructField("Color", StringType(), True)
])

# Apply the schema to the RDD and Create DataFrame
swimmers = spark.createDataFrame(user_fields, schema)

# Creates a temporary view using the DataFrame
swimmers.createOrReplaceTempView("swimmers")

spark.sql("select * from swimmers").show()
# swimmers.select("id", "age").filter("age = 22").show()

spark.sql("select name, eyeColor from swimmers where eyeColor like 'b%'").show()

