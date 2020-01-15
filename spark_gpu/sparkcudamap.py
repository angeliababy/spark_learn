#!/usr/bin/env python
# encoding: utf-8

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession,Row
from pyspark.sql.types import *
import numpy as np
#from numba import cuda
import time
import os
#os.environ['PYSPARK_PYTHON']='/root/anaconda2/bin/python'
os.environ['HADOOP_CONF_DIR']='/etc/hadoop/conf'
os.environ['YARN_CONF_DIR']='/etc/spark2/conf.cloudera.spark2_on_yarn'
 
conf = SparkConf().setMaster("yarn-client").setAppName("First_App")
conf.set("spark.executor.memory","10280m")
conf.set("spark.executor.cores","4")
conf.set("spark.driver.memory","10280m")
conf.set("spark.executor.memoryOverhead","4096m")

sc = SparkContext(conf=conf)


import pycuda.autoinit
import pycuda.driver as drv
from timeit import default_timer as timer

from pycuda.compiler import SourceModule


def gpufunc(xdr_data):
    a=time.time()
    xdr_data = iter(xdr_data)
    inp = np.asarray(list(xdr_data),dtype=np.float32)
    N = len(inp)
    # print("len:",N)
    out = np.zeros((len(inp),7),dtype=np.float32)
    # out = np.empty(N, gpuarray.vec.float1)

    N = np.int32(N)
    #print(inp,out)
    # GPU run
    nTheads = 256*4
    nBlocks = int( ( N + nTheads - 1 ) / nTheads )
    drv.init()
    dev = drv.Device(0)
    contx = dev.make_context()
    mod = SourceModule("""
__global__ void func(float *out, float *inp, size_t N)
{
  //const int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned idxInLeftBlocks = blockIdx.x * (blockDim.x * blockDim.y);
  unsigned idxInCurrBlock  = threadIdx.y * blockDim.x + threadIdx.x;
  unsigned i = idxInLeftBlocks + idxInCurrBlock;
  if (i >= N-1)
  {
    return;
  }
  out[i*7+0] = inp[i*6+5];
  out[i*7+1]=inp[i*6+0];
  if(inp[i*6+1] > 0) 
    out[i*7+2] = 1;
  else 
    out[i*7+2] = 0;
  if(inp[i*6+1]>0 and inp[i*6+3]!=NULL) 
    out[i*7+3] = 1;
  else 
    out[i*7+3] = 0;
  if(inp[i*6+4]>0 and inp[i*6+3] == 0) 
     out[i*7+4] = 1;
  else 
     out[i*7+4] = 0;
  if(inp[i*6+4]>0) 
     out[i*7+5] = inp[i*6+4];
  else 
    out[i*7+5] = 0;
  if(inp[i*6+4]>0 and inp[i*6+3]==0) 
     out[i*7+6] = inp[i*6+4];
  else 
    out[i*7+6] = 0;
}
""")

    func = mod.get_function("func")

    start = timer()
    func(
            drv.Out(out), drv.In(inp), N,
            block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
    out1 = [np.asarray(x) for x in out]
    print("len",len(out1))
    contx.pop()
    del contx
    del inp
    del out
    run_time = timer() - start
    print("gpu run time %f seconds " % run_time)
    return iter(out1)

from pycuda import gpuarray
def main():
    flume_data = sc.textFile("/test1/datas/xdr0.csv.COMPLETED",50)
    #flume_data = sc.textFile("file:///home/users/chenzhuo/program/sparktest/xaa")
    a = time.time()
    lines = flume_data.map(lambda line: line.split(','))
    #print(xdr_data.first())
    #xdr_data = xdr_data.map(lambda p: Row(Response_Time=int(p[0]), Rcode=int(p[1]))
    xdr_data = lines.map(lambda p: \
            Row(Procedure_End_Time=p[0],\
            UL_Data=p[2],\
            DL_Data=p[3],\
            Rcode=p[4],\
            DNSReq_Num=p[5],\
            Response_Time=p[6]))
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .getOrCreate()
    
    #print(xdr_data.first())
    print(time.time()-a)
    data = xdr_data.mapPartitions(gpufunc)
    print(time.time()-a)
    #print(data.take(10))
    data = data.take(10)
   
    print(time.time()-a)
    # The schema is encoded in a string.
    schemaString = "JIAKUAN_DNS_001|JIAKUAN_DNS_002|JIAKUAN_DNS_003|JIAKUAN_DNS_004|JIAKUAN_DNS_005|JIAKUAN_DNS_006|JIAKUAN_DNS_007"

    fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split("|")]
    schema = StructType(fields)

    dataX=[map(str,list(data[i])) for i in range(len(data))]
    #print(dataX)
    df = spark.createDataFrame(sc.parallelize(dataX),schema)
    df.createOrReplaceTempView("xdr_table")

    # SQL can be run over DataFrames that have been registered as a table.
    results = spark.sql("SELECT * FROM xdr_table limit 10")
    results.show()
    print(time.time()-a)

if __name__ == '__main__':
    main()

