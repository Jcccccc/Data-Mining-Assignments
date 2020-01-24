import sys
import json
import time
from pyspark import SparkContext


def custom_partitioner(business_id):
    return hash(business_id)


#start = time.time()
f_in, f_out = sys.argv[1], sys.argv[2]
num_partition = int(sys.argv[3])

sc = SparkContext(master="local[*]", appName='inf553_hw1_task2')

reviews_rdd = sc.textFile(f_in)
result = dict()
result['default'] = dict()
result['customized'] = dict()
result['explanation'] = 'By using the customized partition function, ' \
	+ 'elements with the same key (in this case, business_id) will be partitioned ' \
	+ 'in same partition of RDD, which will reduce the overhead of shuffling.'
result['default']['n_partition'] = reviews_rdd.getNumPartitions()
result['customized']['n_partition'] = num_partition

# default
default_start = time.time()
result['default']['n_items'] = reviews_rdd.glom().map(len).collect()
ans = reviews_rdd.map(lambda x: (json.loads(x)['business_id'], 1)) \
	.reduceByKey(lambda a, b: a+b) \
	.sortBy(lambda x: x[1], ascending=False).take(10)
result['default']['exe_time'] = time.time() - default_start

# customized
custom_start = time.time()
custom_rdd = reviews_rdd.map(lambda x: (json.loads(x)['business_id'], 1)).partitionBy(num_partition, custom_partitioner)
result['customized']['n_items'] = custom_rdd.glom().map(len).collect()
ans = custom_rdd.reduceByKey(lambda a, b: a+b) \
	.sortBy(lambda x: x[1], ascending=False).take(10)
result['customized']['exe_time'] = time.time() - custom_start
with open(f_out, 'w') as output:
	json.dump(result, output, indent=4)
#print(time.time() - start)
