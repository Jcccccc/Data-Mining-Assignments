import sys
import time
import itertools
from pyspark import SparkContext


case, support = int(sys.argv[1]), int(sys.argv[2])
input_file, output_file = sys.argv[3], sys.argv[4]
num_partition = 0


def aPriori(iterator):
    part_support = support / num_partition
    baskets_local, counts, frequent_sets_local = list(), dict(), list()
    # count single items
    for basket in iterator:
        baskets_local.append(basket)
        for item in basket:
            if item in counts:
                counts[item] += 1
            else:
                counts[item] = 1
    # generate frewuent items
    prev_frequent_sets = list()
    for item, count in counts.items():
        if count >= part_support:
            frequent_sets_local.append(frozenset([item]))
            prev_frequent_sets.append(frozenset([item]))
    # generate larger frequent item sets
    size = 2
    while len(prev_frequent_sets) > 0:
        prev_frequent_sets = getNextFrequentSets(baskets_local, prev_frequent_sets, size)
        frequent_sets_local += prev_frequent_sets
        size += 1
    return frequent_sets_local


def getNextFrequentSets(baskets, prev_frequent_sets, size):
    candidates = set()
    for comb in itertools.combinations(prev_frequent_sets, 2):
        union = comb[0].union(comb[1])
        if len(union) == size:
            candidates.add(frozenset(union))
    counts = dict()
    for basket in baskets:
        for candi in candidates:
            if candi.issubset(basket):
                if candi in counts:
                    counts[candi] += 1
                else:
                    counts[candi] = 1
    part_support = support / num_partition
    new_frequent_sets = list()
    for item, count in counts.items():
        if count >= part_support:
            new_frequent_sets.append(item)
    return new_frequent_sets


def tupleKey(items):
    return (len(items),) + items


def output(tuple_list, fout, label):
    fout.write(label+':\n')
    curr_size = 1
    line = ''
    for item_set in tuple_list:
        if len(item_set) == curr_size:
            if curr_size == 1:
                line += ('(\''+item_set[0]+'\'),')
            else:
                line += (str(item_set)+',')
        else:
            fout.write(line[0:-1]+'\n\n')
            line = (str(item_set)+',')
            curr_size += 1
    if line:
        fout.write(line[0:-1]+'\n\n')


def countCandidates(basket):
    candi_in_basket = list()
    for candi in candidates.value:
        if set(candi).issubset(basket):
            candi_in_basket.append((candi, 1))
    return candi_in_basket


start = time.time()
sc = SparkContext(master='local[*]', appName='inf553_hw2_task1')
min_partition_num = max(support//4, 1)
reviews = sc.textFile(input_file, min_partition_num)
num_partition = reviews.getNumPartitions()
reviews = reviews.filter(lambda x: x != 'user_id,business_id')

if case == 1:
    baskets = reviews.distinct().map(lambda x: [str(x.split(',')[0]), str(x.split(',')[1])]) \
        .groupByKey().mapValues(set).values().persist()
else:
    baskets = reviews.distinct().map(lambda x: [str(x.split(',')[1]), str(x.split(',')[0])]) \
        .groupByKey().mapValues(set).values().persist()

candidates_list = baskets.mapPartitions(aPriori).map(lambda x: (tuple(sorted(list(x))), 1)) \
    .reduceByKey(lambda a, b: 1).keys().collect()
candidates_list.sort(key=tupleKey)
fout = open(output_file, 'w')
output(tuple_list=candidates_list, fout=fout, label='Candidates')
candidates = sc.broadcast(candidates_list)
frequent_sets = baskets.flatMap(countCandidates).reduceByKey(lambda a, b: a+b) \
    .filter(lambda x: x[1]>=support).keys().collect()
frequent_sets.sort(key=tupleKey)
output(tuple_list=frequent_sets, fout=fout, label='Frequent Itemsets')
fout.close()

print('Duration: ' + str(time.time()-start))
