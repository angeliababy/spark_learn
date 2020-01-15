# # Spark-SQL from PySpark
#
# This example shows how to send SQL queries to Spark.

from __future__ import print_function
import os
import sys
from pyspark.sql import SparkSession

from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .getOrCreate()

# A list of Rows. Infer schema from the first row, create a DataFrame and print the schema
rows = [Row(name="John", age=19), Row(name="Smith", age=23), Row(name="Sarah", age=18)]
some_df = spark.createDataFrame(rows)
some_df.printSchema()

# A list of tuples
tuples = [("John", 19), ("Smith", 23), ("Sarah", 18)]

# Schema with two fields - person_name and person_age
schema = StructType([StructField("person_name", StringType(), False),
                    StructField("person_age", IntegerType(), False)])

# Create a DataFrame by applying the schema to the RDD and print the schema
another_df = spark.createDataFrame(tuples, schema)
another_df.printSchema()

# Add the data file to hdfs.
#!hdfs dfs -put resources/people.json /tmp

# A JSON dataset is pointed to by path.
# The path can be either a single text file or a directory storing text files.
if len(sys.argv) < 2:
    path = "hdfs://" + \
        os.path.join(os.environ['SPARK_HOME'], "/tmp/people.json")
else:
    path = sys.argv[1]
# Create a DataFrame from the file(s) pointed to by path
people = spark.read.json(path)

# The inferred schema can be visualized using the printSchema() method.
people.printSchema()

# Creates a temporary view using the DataFrame.
people.createOrReplaceTempView("people")

# SQL statements can be run by using the sql methods provided by `spark`
teenagers = spark.sql("SELECT name FROM people WHERE age >= 13 AND age <= 19")

for each in teenagers.collect():
    print(each[0])

spark.stop()



