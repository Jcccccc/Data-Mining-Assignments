import sys
import json
import time
from pyspark import SparkContext

#start = time.time()
f_in, f_out = sys.argv[1], sys.argv[2]
sc = SparkContext(master='local[*]', appName='inf553_hw1_task1')
reviews_rdd = sc.textFile(f_in).coalesce(16)

result = dict()
result['n_review'] = reviews_rdd.count()
result['n_review_2018'] = reviews_rdd.filter(lambda x: json.loads(x)['date'].split('-')[0] == '2018').count()
user_rdd = reviews_rdd.map(lambda x: (json.loads(x)['user_id'], 1)).reduceByKey(lambda a, b: a+b)
result['n_user'] = user_rdd.count()
result['top10_user'] = user_rdd.sortBy(lambda x: x[1], ascending=False).take(10)
bus_rdd = reviews_rdd.map(lambda x: (json.loads(x)['business_id'], 1)).reduceByKey(lambda a, b: a+b)
result['n_business'] = bus_rdd.count()
result['top10_business'] = bus_rdd.sortBy(lambda x: x[1], ascending=False).take(10)
with open(f_out, 'w') as output:
	json.dump(result, output, indent=4)
#print(time.time() - start)