#!/usr/bin/env python
# encoding: utf-8

from pyspark import SparkConf, SparkContext
import numpy as np
from numba import cuda
import time
import os
os.environ['PYSPARK_PYTHON']='/usr/local/python36/bin/python3.5'
os.environ['HADOOP_CONF_DIR']='/etc/hadoop/conf'
os.environ['YARN_CONF_DIR']='/etc/spark2/conf.cloudera.spark2_on_yarn'
 
conf = SparkConf().setMaster("local").setAppName("First_App")
sc = SparkContext(conf=conf)


import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from timeit import default_timer as timer

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void func(float *a, float *b, size_t N)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
  {
    return;
  }
  //float temp_a = a[i];
  //float temp_b = b[i];
  //a[i] = (temp_a * 10 + 2 ) * ((temp_b + 2) * 10 - 5 ) * 5;
  a[i] = b[i];
}
""")

func = mod.get_function("func")


def parseVector(line):
    return np.array([float(x) for x in line.split('|')])

from pycuda import gpuarray
def main():
    flume_data = sc.textFile("file:///home/users/chenzhuo/program/cudapython/u1.user")

    a = time.time()
    xdr_data = flume_data.map(lambda line: line.split('|'))
    #print(xdr_data.first())
    #xdr_data = xdr_data.map(lambda p: Row(Response_Time=int(p[0]), Rcode=int(p[1]))
    
    #xdr_data = xdr_data.map(lambda p: p[0])
    #print(xdr_data.first())
    #xdr_data = flume_data.map(parseVector).cache()
    xdr_data = xdr_data.map(lambda p: p[0])
    inp = np.asarray(xdr_data.collect(),dtype=np.float32)
    N = len(inp)
    # print("len:",N)
    out = np.zeros(N,dtype=np.float32)
    # out = np.empty(N, gpuarray.vec.float1)

    N = np.int32(N)
    print(inp,out)
    # GPU run
    nTheads = 256*4
    nBlocks = int( ( N + nTheads - 1 ) / nTheads )
    start = timer()
    func(
            drv.Out(out), drv.In(inp), N,
            block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
    run_time = timer() - start
    print("gpu run time %f seconds " % run_time)

    print(out)

if __name__ == '__main__':
    main()

