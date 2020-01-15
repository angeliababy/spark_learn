#!/usr/bin/env python
# encoding: utf-8
'''
@author: 陈卓
@license: (C) Copyright 2018-2019, 广州丰石科技有限公司. 
@contact: chenzhuo@richstonedt.com
@software: garner
@file: sparkgpu.py
@time: 2018/10/9 15:44
@desc:
'''

from pyspark import SparkConf, SparkContext
import numpy as np
from numba import cuda
import os
#os.environ['PYSPARK_PYTHON']='/usr/local/python36/bin/python3.5'
#os.environ['HADOOP_CONF_DIR']='/etc/hadoop/conf'
#os.environ['YARN_CONF_DIR']='/etc/spark2/conf.cloudera.spark2_on_yarn'

conf = SparkConf().setMaster("local").setAppName("First_App")
sc = SparkContext(conf=conf)

@cuda.jit("(float32[:],float32[:])")
def foo(inp, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = inp[i] ** 2

def gpu_work(xs):
    inp = np.asarray(list(xs),dtype=np.float32)
    out = np.zeros_like(inp)
    block_size = 32*4
    grid_size = (inp.size+block_size-1)//block_size
   
    foo[grid_size, block_size](inp,out)
    return out

rdd = sc.parallelize(list(range(100)))
rdd.getNumPartitions()
print(rdd.collect())
#print(rdd.getNumPartitions())

rdd = rdd.mapPartitions(gpu_work)
print(rdd.collect())

sc.stop()
