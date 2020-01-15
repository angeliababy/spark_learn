#!/usr/bin/env python
# encoding: utf-8

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils, TopicAndPartition
# import MySQLdb

def start():
    sconf = SparkConf()
    sconf.set('spark.cores.max', 3)
    sc = SparkContext(appName='5MINUTES', conf=sconf)
    ssc = StreamingContext(sc, 5)

    sc = SparkContext("local[2]", "NetworkWordCount")
    ssc = StreamingContext(sc, 10)

    brokers = "slave-qqt06.gdyd.com:6667,slave-qqt07.gdyd.com:6667,slave-qqt08.gdyd.com:6667,slave-qqt10.gdyd.com:6667"
    topic = '5MINUTES'
    start = 70000
    partition = 0
    user_data = KafkaUtils.createDirectStream(ssc, [topic], kafkaParams={"metadata.broker.list": brokers})
    # fromOffsets 设置从起始偏移量消费
    # user_data = KafkaUtils.createDirectStream(ssc,[topic],kafkaParams={"metadata.broker.list":brokers},fromOffsets={TopicAndPartition(topic,partition):long(start)})
    user_fields = user_data.map(lambda line: line[1].split('|'))
    gender_users = user_fields.map(lambda fields: fields[3]).map(lambda gender: (gender, 1)).reduceByKey(
        lambda a, b: a + b)
    user_data.foreachRDD(offset)  # 存储offset信息
    gender_users.pprint()
    # gender_users.foreachRDD(lambda rdd: rdd.foreach(echo))  # 返回元组
    ssc.start()
    ssc.awaitTermination()


offsetRanges = []


def offset(rdd):
    global offsetRanges
    offsetRanges = rdd.offsetRanges()


# def echo(rdd):
#     zhiye = rdd[0]
#     num = rdd[1]
#     for o in offsetRanges:
#         topic = o.topic
#         partition = o.partition
#         fromoffset = o.fromOffset
#         untiloffset = o.untilOffset
#         # 结果插入MySQL
#     conn = MySQLdb.connect(user="root", passwd="******", host="192.168.26.245", db="test", charset="utf8")
#     cursor = conn.cursor()
#     sql = "insert into zhiye(id,zhiye,num,topic,partitions,fromoffset,untiloffset) \
#     values (NULL,'%s','%d','%s','%d','%d','%d')" % (zhiye, num, topic, partition, fromoffset, untiloffset)


# cursor.execute(sql)
# conn.commit()
# conn.close()

if __name__ == '__main__':
    start()