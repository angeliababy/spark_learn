#!/usr/bin/env python
# encoding: utf-8


from pyspark import SparkConf, SparkContext
import numpy as np
from numba import cuda
import time
import os
#os.environ['PYSPARK_PYTHON']='/usr/local/python36/bin/python3.5'
os.environ['HADOOP_CONF_DIR']='/etc/hadoop/conf'
os.environ['YARN_CONF_DIR']='/etc/spark2/conf.cloudera.spark2_on_yarn'

conf = SparkConf().setMaster("local").setAppName("First_App")
sc = SparkContext(conf=conf)

#def gpu_work3(xs):
 #   inp = np.asarray(list(xs))
  #  print(inp)
   # out = np.zeros_like(inp)
    #block_size = 32*4
    #grid_size = (inp.size+block_size-1)//block_size
    #print(grid_size,block_size)
    #foo3[grid_size, block_size](inp,out)
    #print("C")
    #return out

#@cuda.jit("(float32[:],int32[:])")
#def foo3(inp,out):
 #   i,j = cuda.grid(2)
  #  if i < out.shape[0] and j < out.shape[1]:
   #     #print(i,j)
    #    if inp[i][0] > 0 and inp[i][1] == 0:
     #       out[i] = inp[i][0]
      #  else:
       #     out[i] = 0

def gpu_work3(xs):
    print(xs)
    inp = np.asarray(list(xs),dtype=np.int32)
    print(inp)
    out = np.zeros(len(inp),dtype=np.int32)
    block_size = 32*4
    grid_size = (inp.size+block_size-1)//block_size
    foo3[grid_size,block_size](inp,out)
    return out

@cuda.jit
def foo3(inp,out):
    i= cuda.grid(1)
    if i < out.size:
        if inp[i][0] > 0 and inp[i][1] == 0:
            out[i] = inp[i][0]
        else:
            out[i] = 0            

flume_data = sc.textFile("file:///home/users/chenzhuo/program/cudapython/u.user")

a = time.time()
xdr_data = flume_data.map(lambda line: line.split('|'))
#print(xdr_data.first())
#xdr_data = xdr_data.map(lambda p: Row(Response_Time=int(p[0]), Rcode=int(p[1]))
xdr_data = xdr_data.map(lambda p: [int(p[0]), int(p[1])])
print(xdr_data.take(10))

res_data = xdr_data.mapPartitions(gpu_work3)

#rdd = rdd.map(gpu_work)
#print(res_data.take(10))

print(time.time()-a)
sc.stop()
