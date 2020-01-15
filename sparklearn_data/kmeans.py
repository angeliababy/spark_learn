#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
The K-means algorithm written from scratch against PySpark. In practice,
one may prefer to use the KMeans algorithm in ML, as shown in
examples/src/main/python/ml/kmeans_example.py.

This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function

import sys

import numpy as np
from pyspark.sql import SparkSession

# 将读入数据换成float型矩阵
def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

# 求取该点分到哪个点集合中，返回序号
def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex


if __name__ == "__main__":

    # kmeans_data.csv 4 100
    if len(sys.argv) != 4:
        print("Usage: kmeans <file> <k> <convergeDist>", file=sys.stderr)
        exit(-1)

    print("""WARN: This is a naive implementation of KMeans Clustering and is given
       as an example! Please refer to examples/src/main/python/ml/kmeans_example.py for an
       example on how to use ML's KMeans implementation.""", file=sys.stderr)

    spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()

    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    a = lines.first()
    # 此处调用了rdd的map函数把所有的数据都转换为float类型
    data = lines.map(parseVector).cache()
    b = data.first()
    # 这里的K就是设置的中心点数
    K = int(sys.argv[2])
    # 设置的阈值，如果两次之间的距离小于该阈值的话则停止迭代
    convergeDist = float(sys.argv[3])
    # 从点集中用采样的方式来抽取K个值
    kPoints = data.takeSample(False, K, 1)
    # 中心点调整后的距离差
    tempDist = 1.0

    while tempDist > convergeDist:
        # 对所有数据执行map过程，最终生成的是(index, (point, 1))的rdd
        closest = data.map(
            lambda p: (closestPoint(p, kPoints), (p, 1)))
        b = closest.first()
        # 执行reduce过程，该过程的目的是重新求取中心点，生成的也是rdd
        pointStats = closest.reduceByKey(
            lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        b = closest.first()
        # 生成新的中心点
        newPoints = pointStats.map(
            lambda st: (st[0], st[1][0] / st[1][1])).collect()
        b = closest.first()
        # 计算一下新旧中心点的距离差
        tempDist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)

        # 设置新的中心点
        for (iK, p) in newPoints:
            kPoints[iK] = p

    print("Final centers: " + str(kPoints))

    spark.stop()
