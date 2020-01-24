# Author: Jiachen Wu, USC-ID: 8656902544
import sys
import time
import itertools
from pyspark import SparkContext


input_file, output_file = sys.argv[1], sys.argv[2]
adj_list, betweenness = dict(), dict()


class NodeInfo():
    def __init__(self, depth, parent, value, paths_num):
        self.depth = depth
        self.parent = parent
        self.value = value
        self.paths_num = paths_num


def GirvanNewman(root, betweenness):
    nodes = dict()
    for node in adj_list.keys():
        nodes[node] = NodeInfo(-1, [], 1.0, 0.0)
    nodes[root].depth = 0
    nodes[root].paths_num = 1.0
    bfs_queue = [root]
    back_stack = []
    while bfs_queue:
        u = bfs_queue.pop(0)
        if u != root:
            back_stack.append(u)
        for v in adj_list[u]:
            if nodes[v].depth == -1:
                nodes[v].depth = nodes[u].depth + 1
                nodes[v].parent.append(u)
                nodes[v].paths_num += nodes[u].paths_num
                bfs_queue.append(v)
            elif nodes[v].depth == nodes[u].depth+1:
                nodes[v].parent.append(u)
                nodes[v].paths_num += nodes[u].paths_num
    while back_stack:
        u = back_stack.pop()
        for v in nodes[u].parent:
            contrib = nodes[u].value * (nodes[v].paths_num/nodes[u].paths_num)
            if u < v:
                betweenness[(u, v)] += contrib
            else:
                betweenness[(v, u)] += contrib
            nodes[v].value += contrib


#start = time.time()
sc = SparkContext(master='local[*]', appName='inf553_hw4_task1')
sc.setLogLevel('WARN')
data = sc.textFile(input_file, 8)
header = data.first()
rating_dict = data.filter(lambda x: x != header).map(lambda x: (x.split(',')[0], [x.split(',')[1]])) \
    .reduceByKey(lambda x,y: x+y).mapValues(set).collectAsMap()

for pair in itertools.combinations(rating_dict.keys(), 2):
    set1, set2 = rating_dict[pair[0]], rating_dict[pair[1]]
    if len(set1.intersection(set2)) >= 7:
        if pair[0] in adj_list:
            adj_list[pair[0]].append(pair[1])
        else:
            adj_list[pair[0]] = [pair[1]]
        if pair[1] in adj_list:
            adj_list[pair[1]].append(pair[0])
        else:
            adj_list[pair[1]] = [pair[0]]
        if pair[0] < pair[1]:
            if not pair in betweenness:
                betweenness[pair] = 0
        elif not (pair[1], pair[0]) in betweenness:
            betweenness[(pair[1], pair[0])] = 0
for root in adj_list.keys():
    GirvanNewman(root, betweenness)
betweenness_list = list()
for k, v in betweenness.items():
    betweenness_list.append((k, v/2))
betweenness_list.sort(key=lambda x: (-x[1], x[0]))

fout = open(output_file, 'w')
for edge in betweenness_list:
    line = str(edge[0]) + ', ' + str(edge[1]) + '\n'
    fout.write(line)
fout.close()
