#!/usr/bin/env python
# encoding: utf-8

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

import os
os.environ['PYSPARK_PYTHON']='/usr/local/python36/bin/python3.5'
os.environ['HADOOP_CONF_DIR']='/etc/hadoop/conf'
os.environ['YARN_CONF_DIR']='/etc/spark2/conf.cloudera.spark2_on_yarn'

# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext("local[2]", "NetworkWordCount")
sc.setLogLevel('WARN')
ssc = StreamingContext(sc, 10)

# Create a DStream that will connect to hostname:port, like localhost:9999
lines = ssc.textFileStream("/test/")

# Split each line into words
words = lines.flatMap(lambda line: line.split("|"))

# Count each word in each batch
pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)

# Print the first ten elements of each RDD generated in this DStream to the console
wordCounts.pprint()

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate
