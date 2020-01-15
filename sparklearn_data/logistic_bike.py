# -*- coding:utf-8 -*-

from pyspark import SparkConf, SparkContext
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
import numpy as np
import os
# 配置环境
os.environ ['JAVA_HOME'] = 'C:\Java\jdk1.8.0_171'
os.environ ['SPARK_HOME'] = 'C:\spark-2.2.0-bin-hadoop2.7'
os.environ ['HADOOP_HOME'] = 'C:\hadoop.dll-and-winutils.exe-for-hadoop2.7.3-on-windows_X64-master'

# 格式化数据
conf = SparkConf().setMaster("local[*]").setAppName("First_App")
sc = SparkContext(conf=conf)

path = r'C:\Users\user\Desktop\Bike-Sharing-Dataset\hour_noheader.csv'
raw_data = sc.textFile(path)
num_data = raw_data.count()
records =raw_data.map(lambda x: x.split(','))
first = records.first()
print ('数据的第一行:',first)
print ('数据样本数:',num_data)

# 缓存
records.cache()

# 不考虑instant和dteday、casual 和registered
# 对其中8个类型变量，我们使用之前提到的二元编码，剩下4个实数变量不做处理
# 将类型特征表示成二维特征
def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()
print('第三个特征的类别编码： %s '%get_mapping(records,2))

# 对是类型变量的列（第2~9列）应用该函数
mappings = [get_mapping(records, i) for i in range(2,10)]   #对类型变量的列（第2~9列）应用映射函数
print ('类别特征编码字典:',mappings)
cat_len = sum(map(len,[i for i in mappings]))        #类别特征的个数

num_len = len(records.first()[11:15])                      #数值特征的个数
total_len = num_len+cat_len                                  #所有特征的个数
print( '类别特征的个数： %d'% cat_len)
print ('数值特征的个数： %d'% num_len)
print( '所有特征的个数:：%d' % total_len)

from pyspark.mllib.regression import LabeledPoint
import numpy as np

# 根据已经创建的映射对每个特征进行二元编码
def extract_features(record):
    cat_vec = np.zeros(cat_len)
    step = 0
    for i,raw_feature in enumerate(record[2:9]):
        dict_code = mappings[i]
        index = dict_code[raw_feature]
        cat_vec[index+step] = 1
        step = step+len(dict_code)
    num_vec = np.array([float(raw_feature) for raw_feature in record[10:14]])
    return np.concatenate((cat_vec, num_vec))

# 将数据中的最后一列cnt的数据转换成浮点数
def extract_label(record):
    return float(record[-1])

# 特征提取
data = records.map(lambda point: LabeledPoint(extract_label(point),extract_features(point)))
first_point = data.first()

print ('原始特征向量:' +str(first[2:]))
print ('标签:' + str(first_point.label))
print ('对类别特征进行独热编码之后的特征向量: \n' + str(first_point.features))
print ('对类别特征进行独热编码之后的特征向量长度:' + str(len(first_point.features)))


def extract_features_dt(record):
    return np.array(map(float, record[2:14]))
data_dt = records.map(lambda point: LabeledPoint(extract_label(point), extract_features_dt(point)))
first_point_dt = data_dt.first()
print '决策树特征向量: '+str(first_point_dt.features)
print '决策树特征向量长度: '+str(len(first_point_dt.features))


from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
help(LinearRegressionWithSGD.train)


linear_model = LinearRegressionWithSGD.train(data, iterations=10, step=0.1, intercept =False)
true_vs_predicted = data.map(lambda point:(point.label,linear_model.predict(point.features)))
print '线性回归模型对前5个样本的预测值: '+ str(true_vs_predicted.take(5))


dt_model = DecisionTree.trainRegressor(data_dt,{})
preds = dt_model.predict(data_dt.map(lambda p: p.features))
actual = data.map(lambda p:p.label)
true_vs_predicted_dt = actual.zip(preds)
print '决策树回归模型对前5个样本的预测值: '+str(true_vs_predicted_dt.take(5))
print '决策树模型的深度: ' + str(dt_model.depth())
print '决策树模型的叶子节点个数: '+str(dt_model.numNodes())


def squared_error(actual, pred):
    return (pred-actual)**2

def abs_error(actual, pred):
    return np.abs(pred-actual)

def squared_log_error(pred, actual):
    return (np.log(pred+1)-np.log(actual+1))**2

mse = true_vs_predicted.map(lambda (t, p): squared_error(t, p)).mean()
mae = true_vs_predicted.map(lambda (t, p): abs_error(t, p)).mean()
rmsle = np.sqrt(true_vs_predicted.map(lambda (t, p): squared_log_error(t, p)).mean())
print 'Linear Model - Mean Squared Error: %2.4f' % mse
print 'Linear Model - Mean Absolute Error: %2.4f' % mae
print 'Linear Model - Root Mean Squared Log Error: %2.4f' % rmsle


