import csv
import sys
import random
from pyspark import SparkContext

m = 65537

def getRandomCoeffs(num):
    coeffs = list()
    for i in range(num):
        x = random.randint(1, m)
        while x in coeffs:
            x = random.randint(1, m)
        coeffs.append(x)
    return coeffs

input_file, output_file = sys.argv[1], sys.argv[2]
b, r = 50, 3
hash_num = b * r
A = getRandomCoeffs(hash_num)
B = getRandomCoeffs(hash_num)
user_list = list()

def getSigns(business_with_rated_users):
    business_id, rated_users = business_with_rated_users[0], business_with_rated_users[1]
    sign = [m for i in range(hash_num)]
    for index, user in enumerate(user_list):
        if user in rated_users:
            for i in range(hash_num):
                hash_value = (index*A[i]+B[i]) % m
                if hash_value < sign[i]:
                    sign[i] = hash_value
    ret = list()
    for i in range(b):
        ret.append((tuple(sign[(i*r):(i*r+r)])+(i,), [business_id]))
    return ret

def getPairs(ids):
    l = len(ids)
    ret = list()
    for i in range(l-1):
        for j in range(i+1, l):
            ret.append((ids[i], ids[j]))
    return ret

sc = SparkContext(master='local[*]', appName='inf553_hw3_task1')
sc.setLogLevel("WARN")
ratings = sc.textFile(input_file, 8).filter(lambda x: x != 'user_id, business_id, stars')
businesses = ratings.map(lambda x: (x.split(',')[1], [x.split(',')[0]])) \
    .reduceByKey(lambda x,y: x+y).mapValues(set)
user_list = ratings.map(lambda x: x.split(',')[0]).distinct().collect()
signs = businesses.flatMap(getSigns).reduceByKey(lambda x,y: x+y).filter(lambda x: len(x[1])>1) \
    .values().flatMap(getPairs).distinct()
candidates_single = signs.flatMap(lambda x: [x[0], x[1]]).distinct().collect()
rate_set = businesses.filter(lambda x: x[0] in candidates_single).collectAsMap()

ans = list()
for pair in signs.collect():
    set1 = rate_set[pair[0]]
    set2 = rate_set[pair[1]]
    jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
    if jaccard >= 0.5:
        if pair[0] < pair[1]:
            ans.append([pair[0], pair[1], jaccard])
        else:
            ans.append([pair[1], pair[0], jaccard])
ans.sort()

fout = open(output_file, mode='w')
fwriter = csv.writer(fout, delimiter=',', quoting=csv.QUOTE_MINIMAL)
fwriter.writerow(['business_id_1', ' business_id_2', ' similarity'])
for pair in ans:
    fwriter.writerow([str(pair[0]), str(pair[1]), pair[2]])
fout.close()

