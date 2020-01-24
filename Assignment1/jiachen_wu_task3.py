import sys
import json
import time
from pyspark import SparkContext

f_in1, f_out1 = sys.argv[1], sys.argv[3]
f_in2, f_out2 = sys.argv[2], sys.argv[4]

#start = time.time()
sc = SparkContext(master="local[*]", appName='inf553_hw1_task3')

avg_rdd = sc.textFile(f_in1) \
	.map(lambda x: (json.loads(x)['business_id'], json.loads(x)['stars'])) \
	.partitionBy(8) \
	.join(sc.textFile(f_in2).map(lambda x: (json.loads(x)['business_id'], json.loads(x)['city']))) \
	.map(lambda x: (x[1][1], (x[1][0], 1))) \
	.reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
	.mapValues(lambda x: x[0]/x[1]) \
	.sortBy(lambda x: (-x[1], x[0]), ascending=True)
avg_rdd.persist()
with open(f_out1, 'w') as output:
	for city in avg_rdd.collect():
		output.write('{},{}\n'.format(city[0], city[1]))

result = dict()
m1_start = time.time()
print(avg_rdd.collect()[0:10])
result['m1'] = time.time() - m1_start
m2_start = time.time()
print(avg_rdd.take(10))
result['m2'] = time.time() - m2_start
result['explanation'] = 'Method1 first copies all elements in RDD to a python list ' \
	+ 'while Method2 only copies ten elements of RDD. Method2 is faster because it saves ' \
	+ 'the time of copying rest of the elements from RDD.'
with open(f_out2, 'w') as output:
	json.dump(result, output, indent=4)

#print(time.time()-start)