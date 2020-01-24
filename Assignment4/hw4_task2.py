# Author: Jiachen Wu, USC-ID: 8656902544
import sys
import copy
import time
import itertools
from pyspark import SparkContext


class NodeInfo():
    def __init__(self, depth, parent, value, paths_num):
        self.depth = depth
        self.parent = parent
        self.value = value
        self.paths_num = paths_num


def GirvanNewman(root, adj_list, betweenness):
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
                if (u, v) in betweenness:
                    betweenness[(u, v)] += contrib
                else:
                    betweenness[(u, v)] = contrib
            elif (v, u) in betweenness:
                betweenness[(v, u)] += contrib
            else:
                betweenness[(v, u)] = contrib
            nodes[v].value += contrib


def GetBetweeness(adj_list):
    betweenness = dict()
    for root in adj_list.keys():
        GirvanNewman(root, adj_list, betweenness)
    return betweenness


def GetCommunities(adj_list):
    communities, added = list(), dict()
    for u in adj_list.keys():
        if not u in added:
            communities.append([])
            to_add = [u]
            added[u] = True
            while to_add:
                v = to_add.pop()
                communities[-1].append(v)
                for w in adj_list[v]:
                    if not w in added:
                        added[w] = True
                        to_add.append(w)
    return communities


def GetModularity(communities, edges, degree):
    m = len(edges)
    modularity = 0.0
    for group in communities:
        for u in group:
            for v in group:
                if u != v:
                    a_uv = int((u, v) in edges or (v, u) in edges)
                    modularity += (a_uv - degree[u]*degree[v]/2.0/m)
    return modularity/2.0/m


#start = time.time()
input_file, output_file = sys.argv[1], sys.argv[2]
sc = SparkContext(master='local[*]', appName='inf553_hw4_task2')
sc.setLogLevel('WARN')
data = sc.textFile(input_file, 8)
header = data.first()
rating_dict = data.filter(lambda x: x != header).map(lambda x: (x.split(',')[0], [x.split(',')[1]])) \
    .reduceByKey(lambda x,y: x+y).mapValues(set).collectAsMap()

adj_list, edges_origin, degree = dict(), set(), dict()
for pair in itertools.combinations(rating_dict.keys(), 2):
    set1, set2 = rating_dict[pair[0]], rating_dict[pair[1]]
    if len(set1.intersection(set2)) >= 7:
        edges_origin.add(pair)
        if pair[0] in adj_list:
            adj_list[pair[0]].append(pair[1])
            degree[pair[0]] += 1
        else:
            adj_list[pair[0]] = [pair[1]]
            degree[pair[0]] = 1
        if pair[1] in adj_list:
            adj_list[pair[1]].append(pair[0])
            degree[pair[1]] += 1
        else:
            adj_list[pair[1]] = [pair[0]]
            degree[pair[1]] = 1

betweenness = GetBetweeness(adj_list)
communities = GetCommunities(adj_list)
modularity = GetModularity(communities, edges_origin, degree)
max_modularity = -1
epsilon = 1e-10
removed = 0
while removed < len(edges_origin):
    if modularity > max_modularity:
        max_modularity = modularity
        best_communities = copy.deepcopy(communities)
    max_betweenness = max(betweenness.values())
    for pair, value in betweenness.items():
        if abs(value-max_betweenness) < epsilon:
            removed += 1
            adj_list[pair[0]].remove(pair[1])
            adj_list[pair[1]].remove(pair[0])
    betweenness = GetBetweeness(adj_list)
    communities = GetCommunities(adj_list)
    modularity = GetModularity(communities, edges_origin, degree)

for c in best_communities:
    c.sort()
best_communities.sort(key=lambda x: (len(x), x[0]))
fout = open(output_file, 'w')
for c in best_communities:
    line = str(c)[1:-1] + '\n'
    fout.write(line)
fout.close()