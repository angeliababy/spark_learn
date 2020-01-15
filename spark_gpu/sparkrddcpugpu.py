# #!/usr/bin/env python
# # encoding: utf-8

#

from pyspark import SparkConf, SparkContext
import numpy as np
# from numba import cuda

import os
os.environ ['JAVA_HOME'] = 'C:\Program Files\Java\jdk1.8.0_171'
os.environ ['SPARK_HOME'] = 'C:\spark-2.2.0-bin-hadoop2.7'
os.environ ['HADOOP_HOME'] = 'C:\hadoop.dll-and-winutils.exe-for-hadoop2.7.3-on-windows_X64-master'
# os.environ['PYSPARK_PYTHON']='/usr/local/python36/bin/python3.5'
# os.environ['HADOOP_CONF_DIR']='/etc/hadoop/conf'
# os.environ['YARN_CONF_DIR']='/etc/spark2/conf.cloudera.spark2_on_yarn'

path=os.getcwd()+"/u.user"

from pyspark.sql import SparkSession,Row
from pyspark.sql.types import *
conf = SparkConf().setMaster("local").setAppName("First_App")
sc = SparkContext(conf=conf)

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .getOrCreate()

# @cuda.jit("(float32[:],float32[:])")
# def foo(Response_Time, Rcode, out):
#     i = cuda.grid(1)
#     if i < out.size:
#         if Response_Time[i]>0 and Rcode[i]==0:
#             out[i] = 1
#         else:
#             out[i] = 0

# def gpu_work(xs):
#     inp = np.asarray(list(xs),dtype=np.float32)
#     out = np.zeros_like(inp)
#     block_size = 32*4
#     grid_size = (inp.size+block_size-1)//block_size
#
#     foo[grid_size, block_size](inp,out)
#     return out

def fun(x):
    if x[0]>0 and x[1]==24:
        return x[0]
    else:
        return 0

flume_data = sc.textFile(path)

xdr_data = flume_data.map(lambda line: line.split('|'))
print(xdr_data.first())
xdr_data = xdr_data.map(lambda p: (int(p[0]), int(p[1])))

res_data = xdr_data.map(lambda x: fun(x))

# rdd = rdd.map(gpu_work)
# print(res_data.first())

sc.stop()
