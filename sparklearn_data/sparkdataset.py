# /opt/cloudera/parcels/CDH-5.12.0-1.cdh5.12.0.p0.29/lib/spark/bin/spark-submit
# /opt/cloudera/parcels/SPARK2-2.2.0.cloudera1-1.cdh5.12.0.p0.142354/lib/spark2/bin/spark-submit    \
# --master yarn-client    --executor-memory 8G    --num-executors 5    --executor-cores 2    \
# --driver-memory 8G    --queue spark   keliucellstep2.py     env=dev     log.level=info
#
#  /opt/cloudera/parcels/SPARK2-2.2.0.cloudera1-1.cdh5.12.0.p0.142354/lib/spark2/bin/spark-submit --jars /opt/cloudera/parcels/CDH-5.13.1-1.cdh5.13.1.p0.2/jars/spark-streaming-kafka_2.10-1.6.0-cdh5.13.1.jar streamingxdrcpu.py
# #
# # sc.setLogLevel('WARN')
#


# -*- coding:utf-8 -*-

# https://blog.csdn.net/u013719780/article/details/51768720

from pyspark import SparkConf, SparkContext
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import hist
import numpy as np
import os
# 配置环境
os.environ ['JAVA_HOME'] = 'C:\Program Files\Java\jdk1.8.0_171'
os.environ ['SPARK_HOME'] = 'C:\spark-2.2.0-bin-hadoop2.7'
os.environ ['HADOOP_HOME'] = 'C:\hadoop.dll-and-winutils.exe-for-hadoop2.7.3-on-windows_X64-master'

# 格式化数据
conf = SparkConf().setMaster("local[*]").setAppName("First_App")
sc = SparkContext(conf=conf)


# 路径需变
# 1.探索用户数据
user_data = sc.textFile(r'D:\projects\sparklearn_data\ml-100k\u.user')
print(user_data.first())  # 1|24|M|technician|85711首行用户、年龄、性别、职业、邮编
print(user_data.count())
# tip1:把map()理解为要对每一行做这个事情,对每个元素做动作
# tip2:lambda x:f(x) x就是那个object，f(x)是要对object做的事
# 各类算子
# 1、map()：对每行，用map()中的函数作用
# 2、filter():对每一个元素，括号里给出筛选条件，进行过滤
# 1、count():计数、加和
# 2、distinct():取所有不同的元素，类似于做set()操作，去重
# 3、collect():把分散的内容整理成一个数组，例如，形成一个由每行中的“年龄”组成的数组
user_fields = user_data.map(lambda line: line.split('|'))  # |分割
num_users = user_fields.map(lambda fields: fields[0]).count()   # 统计用户数,fields可改
num_genders = user_fields.map(lambda fields : fields[2]).distinct().count()   # 统计性别个数
num_occupations = user_fields.map(lambda fields: fields[3]).distinct().count()   # 统计职业个数
num_zipcodes = user_fields.map(lambda fields: fields[4]).distinct().count()   # 统计邮编个数
print(num_users,num_genders,num_occupations,num_zipcodes)

# 计算年龄分布人数
ages = user_fields.map(lambda x: int(x[1])).collect()  # 分组,数组
print(ages)
# hist(ages, bins=20, color='lightblue',normed=True) # bins指直方图总个数，调整区间
# fig = plt.gcf()  #通过plt.gcf函数获取当前的绘图对象
# fig.set_size_inches(12,6) # 尺寸
# plt.show()
#
# 画出用户的职业的分布图：
# 前面(fields[3],1)创建(各职业,1)，后面reduceByKey形成key-value，相同职业数目相加
count_by_occupation = user_fields.map(lambda fields: (fields[3],1)).reduceByKey(lambda x,y:x+y).collect()
print(count_by_occupation)
x_axis1 = np.array([c[0] for c in count_by_occupation])
y_axis1 = np.array([c[1] for c in count_by_occupation])
x_axis = x_axis1[np.argsort(y_axis1)]
y_axis = y_axis1[np.argsort(y_axis1)]
pos = np.arange(len(x_axis))
width = 1.0
# ax = plt.axes()
# ax.set_xticks(pos+(width)/2)
# print(pos+(width)/2)
# ax.set_xticklabels(x_axis)
#
# plt.bar(pos, y_axis, width, color='lightblue') # 柱状图
# plt.xticks(rotation=30)
# fig = plt.gcf()
# fig.set_size_inches(12,6)
# plt.show()
#
# countByValue与上作用相同，key_value
count_by_occupation2 = user_fields.map(lambda fields: fields[3]).countByValue()
print("Map-reduce approach:")
print(dict(count_by_occupation2))
print ("========================" )
print ("countByValue approach:")
print (dict(count_by_occupation))


# # 2.探索电影数据
movie_data = sc.textFile(r'D:\projects\sparklearn_data\ml-100k\u.item')
num_movies = movie_data.count()
print(movie_data.first()) #1|Toy Story (1995)|01-Jan-1995||http:
print(movie_data.count())

#画出电影的age分布图：
def convert_year(x):
    try:
        return int(x[-4:])
    except:
        return 1900

movie_fields = movie_data.map(lambda lines:lines.split('|'))
# 自建函数
years = movie_fields.map(lambda fields: fields[2]).map(lambda x: convert_year(x))
years_filtered = years.filter(lambda x: x!=1900)
print(years_filtered.count())
movie_ages = years_filtered.map(lambda yr:1998-yr).countByValue()
values = movie_ages.values()
print(values)
bins = len(movie_ages.keys())
print(bins)
#
# hist(values, bins=bins, color='lightblue',normed=True)
# fig = plt.gcf()
# fig.set_size_inches(12,6)
# plt.show()


# 3.探索评分数据

#查看数据记录数量：
rating_data = sc.textFile(r'D:\projects\sparklearn_data\ml-100k\u.data')
print(rating_data.first())
print(rating_data)
#对数据进行一些基本的统计：
num_ratings = rating_data.count()
rating_data = rating_data.map(lambda line: line.split('\t'))
print(type(rating_data))
ratings = rating_data.map(lambda fields: int(fields[2]))
# reduce依次对数据集中的数据进行操作
max_rating = ratings.reduce(lambda x,y:max(x,y))
min_rating = ratings.reduce(lambda x,y:min(x,y))
mean_rating = ratings.reduce(lambda x,y:x+y)/num_ratings
median_rating = np.median(ratings.collect())
ratings_per_user = num_ratings/num_users
ratings_per_movie = num_ratings/ num_movies
print ('Min rating: %d' %min_rating)
print ('max rating: %d' % max_rating)
print ('Average rating: %2.2f' %mean_rating)
print ('Median rating: %d '%median_rating)
print ('Average # of ratings per user: %2.2f'%ratings_per_user)
print ('Average # of ratings per movie: %2.2f' % ratings_per_movie)
# 数学综合统计，作用与上面相似
print(ratings.stats())

count_by_rating = ratings.countByValue()
x_axis = np.array(list(count_by_rating.keys()))  # 必须转化成list才有后面len操作
y_axis = np.array([float(c) for c in count_by_rating.values()])
y_axis_normed = y_axis/y_axis.sum()
pos = np.arange(len(x_axis))
width = 1.0
# ax = plt.axes()
# ax.set_xticks(pos+(width/2))
# ax.set_xticklabels(x_axis)

# plt.bar(pos, y_axis_normed, width, color='lightblue')
# plt.xticks(rotation=30)
# fig = plt.gcf()
# fig.set_size_inches(12,6)
# plt.show()

# 计算每个用户和其对应的评价次数：
# groupByKey与上面reduceByKey类似
user_rating_byuser = rating_data.map(lambda fields:(int(fields[0]),int(fields[2]))).groupByKey().map(lambda x_y:(x_y[0],len(x_y[1])))
print(user_rating_byuser.take(5)) #前五个数

user_ratings_byuser_local = user_rating_byuser.map(lambda k_v:k_v[1]).collect()
# hist(user_ratings_byuser_local, bins=200, color = 'lightblue',normed = True)
# fig = plt.gcf()
# fig.set_size_inches(12,6)
# plt.show()


# 4. 数据处理与转换
# 缺失值处理
#用指定值替换bad values和missing values
years_pre_processed = movie_fields.map(lambda fields: fields[2]).map(lambda x: convert_year(x)).collect()
years_pre_processed_array = np.array(years_pre_processed)
mean_year = np.mean(years_pre_processed_array[years_pre_processed_array!=1900])
median_year = np.median(years_pre_processed_array[years_pre_processed_array!=1900])
# 原型numpy.where(condition[, x, y])1、这里x,y是可选参数，condition是条件，这三个输入参数都是array_like的形式；而且三者的维度相同2、当conditon的某个位置的为true时，输出x的对应位置的元素，否则选择y对应位置的元素；3、如果只有参数condition，则函数返回为true的元素的坐标位置信息；
index_bad_data = np.where(years_pre_processed_array==1900)
print(index_bad_data)
years_pre_processed_array[index_bad_data] = median_year #中位数代替缺失值
print('Mean year of release: %d' % mean_year)
print('Median year of release: %d ' % median_year)
print("Index of '1900' after assigning median: %s"% np.where(years_pre_processed_array==1900)[0])
#
#  提取特征
# 4.1.类别特征编码
# 第一步，编码成数值型1,2,···
all_occupations = user_fields.map(lambda fields:fields[3]).distinct().collect()
print(type(all_occupations))
all_occupations.sort()
idx = 0
all_occupations_dict = {} #创建空字典
for o in all_occupations:
    all_occupations_dict[o] = idx
    idx +=1
print ("Encoding of 'doctor': %d" %all_occupations_dict['doctor']) #doctor的编码号
print ("Encoding of 'programmer': %d" % all_occupations_dict['programmer'])
#第二步，dummies处理[0,1,0,0,0]
K=len(all_occupations_dict)
binary_x = np.zeros(K)
k_programmer = all_occupations_dict['programmer']
binary_x[k_programmer] = 1
print ('Binary feature vector: %s'%binary_x) #programmer 0-1编码
print ('Length of binray vector: %d' %K)

# 4.2. 派生特征(新增特征)
# 时间函数：时间戳转化为datetime类型
def extract_datetime(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts)
timestamps = rating_data.map(lambda fields:int(fields[3]))
hour_of_day = timestamps.map(lambda ts: extract_datetime(ts).hour) #取出小时
# 按时间段划分morning,lunch, afternoon, evening, night
def assign_tod(hr):
    times_of_day = {
        'morning':range(7,12),
        'lunch': range(12,14),
        'afternoon':range(14,18),
        'evening':range(18,23),
        'night': [23,24,1,2,3,4,5,6]
        }
    for k,v in times_of_day.items():
        if hr in v:
            return k
time_of_day = hour_of_day.map(lambda hr: assign_tod(hr))
print(time_of_day.take(5))
# 类别编码，同上
time_of_day_unique = time_of_day.map(lambda fields:fields).distinct().collect()
time_of_day_unique = np.array(time_of_day_unique)
index_bad_data = np.where(time_of_day_unique==None)
# 缺失值处理
time_of_day_unique[index_bad_data] = 'lunch'
print(time_of_day_unique)
time_of_day_unique.sort()
idx = 0
time_of_day_unique_dict = {}
for o in time_of_day_unique:
    time_of_day_unique_dict[o] = idx
    idx +=1
print("Encoding of 'afternoon': %d" %time_of_day_unique_dict['afternoon'])
print ("Encoding of 'morning': %d" % time_of_day_unique_dict['morning'])
print ("Encoding of 'lunch': %d" % time_of_day_unique_dict['lunch'])
# dummies处理，同上
K=len(time_of_day_unique_dict)
binary_x = np.zeros(K)
k_time = time_of_day_unique_dict['afternoon']
binary_x[k_time] = 1
print ('Binary feature vector: %s'%binary_x)
print ('Length of binray vector: %d' %K)
#
# 4.3 文本特征
# 提取出titles
def extract_title(raw):
    import re
    grps = re.search("\((\w+)\)",raw)
    if grps:
        return raw[:grps.start()].strip()
    else:
        return raw
raw_titles = movie_fields.map(lambda fields: fields[1])
for raw_title in raw_titles.take(5):
    print (extract_title(raw_title))

# 分词处理
movie_titles = raw_titles.map(lambda m: extract_title(m))
title_terms = movie_titles.map(lambda m:m.split(' '))
print (title_terms.take(5))

# 所有titles出现的word去重，求word的list
all_terms = title_terms.flatMap(lambda x: x).distinct().collect()
idx = 0
all_terms_dict = {}
for term in all_terms:
    all_terms_dict[term] = idx
    idx += 1
print("Total number of terms: %d" % len(all_terms_dict))
print("Index of term 'Dead': %d" % all_terms_dict['Dead'])
print("Index of term 'Rooms': %d" % all_terms_dict['Rooms'])

# Spark内置的zipWithIndex，作用同上
all_terms_dict2 = title_terms.flatMap(lambda x:x).distinct().zipWithIndex().collectAsMap()
print ("Index of term 'Dead %d" % all_terms_dict['Dead'])
print ("Index of term 'Rooms': %d" % all_terms_dict['Rooms'])

# 压缩稀疏(csc_matrix)的存储数据
def create_vector(terms, term_dict):
    from scipy import sparse as sp
    num_terms = len(term_dict)
    x = sp.csc_matrix((1,num_terms))
    for t in terms:
        if t in term_dict:
            idx = term_dict[t]
            x[0,idx] = 1
    return x
all_terms_bcast = sc.broadcast(all_terms_dict)
term_vectors = title_terms.map(lambda terms: create_vector(terms,all_terms_bcast.value))
print(term_vectors.take(5))
#
#规范数据
np.random.seed(42)
# 从标准正态分布中返回一个或多个样本值
x = np.random.randn(10)
norm_x_2 = np.linalg.norm(x)
normalized_x = x / norm_x_2
print ("x:\n%s" % x)
print ("2-Norm of x: %2.4f" % norm_x_2)
print ("Normalized x:\n%s" % normalized_x)
print ("2-Norm of normalized_x: %2.4f" %np.linalg.norm(normalized_x))

from pyspark.mllib.feature import Normalizer
normlizer = Normalizer()
vector = sc.parallelize([x])
normalized_x_mllib = normlizer.transform(vector).first().toArray()
print ("x:\n%s" % x)
print ("2-Norm of x: %2.4f" % norm_x_2)
print ("Normalized x:\n%s" % normalized_x_mllib)
print ("2-Norm of normalized_x: %2.4f" %np.linalg.norm(normalized_x_mllib))
