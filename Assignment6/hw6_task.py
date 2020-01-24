# Author: Jiachen Wu, USC-ID: 8656902544
import sys
import copy
import time
import random
import itertools
import numpy as np
from pyspark import SparkContext
from sklearn.cluster import KMeans


class DiscardSet():

    def __init__(self, features):
        self.N = len(features)
        self.SUM = np.sum(features, axis=0)
        self.SUMSQ = np.sum(features**2, axis=0)
        self.SIGMA = np.sqrt((self.SUMSQ / self.N) - (self.SUM / self.N)**2)

    def add_point(self, point):
        self.N += 1
        self.SUM += point
        self.SUMSQ += point**2
        self.SIGMA = np.sqrt((self.SUMSQ / self.N) - (self.SUM / self.N)**2)

    def merge(self, cset):
        self.N += cset.N
        self.SUM += cset.SUM
        self.SUMSQ += cset.SUMSQ
        self.SIGMA = np.sqrt((self.SUMSQ / self.N) - (self.SUM / self.N)**2)

    def mahalanobis_distance(self, point):
        return np.sqrt(np.sum(((self.SUM/self.N-point)/self.SIGMA)**2))

    def mahalanobis_distance_cs(self, cset):
        point = cset.get_centroid()
        return np.sqrt(np.sum(((self.SUM/self.N-point)/self.SIGMA)**2))


class CompressionSet():

    def __init__(self, features, pids):
        self.pids = pids
        self.N = len(features)
        self.SUM = np.sum(features, axis=0)
        self.SUMSQ = np.sum(features**2, axis=0)
        self.SIGMA = np.sqrt((self.SUMSQ / self.N) - (self.SUM / self.N)**2)

    def get_centroid(self):
        return self.SUM / self.N

    def add_point(self, point, pid):
        self.pids.append(pid)
        self.N += 1
        self.SUM += point
        self.SUMSQ += point**2
        self.SIGMA = np.sqrt((self.SUMSQ / self.N) - (self.SUM / self.N)**2)

    def merge(self, cset):
        self.pids += cset.pids
        self.N += cset.N
        self.SUM += cset.SUM
        self.SUMSQ += cset.SUMSQ
        self.SIGMA = np.sqrt((self.SUMSQ / self.N) - (self.SUM / self.N)**2)

    def mahalanobis_distance(self, point):
        return np.sqrt(np.sum(((self.SUM/self.N-point)/self.SIGMA)**2))

    def mahalanobis_distance_cs(self, cset):
        point = cset.get_centroid()
        return np.sqrt(np.sum(((self.SUM/self.N-point)/self.SIGMA)**2))


class BradleyFayyadReina():

    def __init__(self, input_file, output_file, num_cluster):
        self.input_file = input_file
        self.fout = open(output_file, 'w')
        self.fout.write('The intermediate results:\n')
        self.num_cluster = num_cluster
        self.large_cluster = num_cluster * 10
        sc = SparkContext(master='local[*]', appName='inf553_hw6')
        #sc.setLogLevel('WARN')
        data = sc.textFile(input_file, 8).collect()
        self.ground_truth, self.features, self.pid_dict, self.predicts = {}, [], {}, {}
        for line in data:
            line_split = line.split(',')
            pid, p_cluster, feature = int(line_split[0]), int(line_split[1]), line_split[2:]
            self.ground_truth[pid] = p_cluster
            self.features.append(feature)
            self.pid_dict[tuple(feature)] = pid
            self.predicts[pid] = -1
        random.shuffle(self.features)
        self.features = np.array(self.features)
        self.dimension = self.features.shape[1]
        self.num_points = len(self.ground_truth)
        self.starts = [self.num_points//5, self.num_points//5*2, self.num_points//5*3, self.num_points//5*4]
        self.DS, self.CS, self.RS = [], [], []
        self.DS_points = 0
        self.CS_points, self.CS_clusters = 0, 0
        self.RS_points = 0

    def init_clusters(self):
        features = self.features[0:self.starts[0]]
        labels = KMeans(n_clusters=self.large_cluster, random_state=0) \
            .fit_predict(np.array(features, dtype=np.float64))
        for label in range(self.large_cluster):
            id_with_label = np.argwhere(labels==label)
            """ RS with <10-point clusters """
            if len(id_with_label) <= 10:
                id_with_label = id_with_label.reshape(id_with_label.shape[0])
                self.RS += list(features[id_with_label])
                labels = np.delete(labels, id_with_label)
                features = np.delete(features, id_with_label, 0)
        labels = KMeans(n_clusters=self.num_cluster, random_state=0) \
            .fit_predict(np.array(features, dtype=np.float64))
        for label in range(self.num_cluster):
            id_with_label = np.argwhere(labels==label)
            id_with_label = id_with_label.reshape(id_with_label.shape[0])
            feature_with_label = features[id_with_label]
            for feature in feature_with_label:
                self.predicts[self.pid_dict[tuple(feature)]] = label
                self.DS_points += 1
            self.DS.append(DiscardSet(np.array(feature_with_label, dtype=np.float64)))
        if len(self.RS) > self.large_cluster:            
            rs_labels = KMeans(n_clusters=self.large_cluster//2, random_state=0) \
                .fit_predict(np.array(self.RS, dtype=np.float64))
            self.RS = np.array(self.RS)
            for label in range(self.large_cluster):
                id_with_label = np.argwhere(rs_labels==label)
                if len(id_with_label) > 1:
                    id_with_label = id_with_label.reshape(-1)
                    feature_with_label = self.RS[id_with_label]
                    pids_with_label = []
                    for point in feature_with_label:
                        pids_with_label.append(self.pid_dict[tuple(point)])
                    self.CS.append(CompressionSet(np.array(feature_with_label, dtype=np.float64), pids_with_label))
                    self.CS_clusters += 1
                    self.CS_points += len(id_with_label)
                    self.RS = np.delete(self.RS, id_with_label, 0)
                    rs_labels = np.delete(rs_labels, id_with_label)
        else:
            self.RS = np.array(self.RS)
        self.print_state(1)

    def clustering(self):
        self.init_clusters()
        for round_num in range(2, 6):
            if round_num < 5:
                features = self.features[self.starts[round_num-2]:self.starts[round_num-1]]
            else:
                features = self.features[self.starts[round_num-2]:]
            for j in range(features.shape[0]):
                point = np.array(features[j], dtype=np.float64)
                min_distance, min_label = np.sqrt(self.dimension)*1.5, -1
                for i, cluster in enumerate(self.DS):
                    if cluster.mahalanobis_distance(point) < min_distance:
                        min_distance = cluster.mahalanobis_distance(point)
                        min_label = i
                if min_label != -1:
                    # assign to DS
                    self.DS[min_label].add_point(point)
                    self.predicts[self.pid_dict[tuple(features[j])]] = min_label
                    self.DS_points += 1
                else:
                    min_distance, min_label = np.sqrt(self.dimension)*1.5, -1
                    for i, cluster in enumerate(self.CS):
                        if cluster.mahalanobis_distance(point) < min_distance:
                            min_distance = cluster.mahalanobis_distance(point)
                            min_label = i
                    if min_label != -1:
                        # assign to CS
                        self.CS[min_label].add_point(point, self.pid_dict[tuple(features[j])])
                        self.CS_points += 1
                    else:
                        # assign to RS
                        self.RS = np.append(self.RS, [features[j]], axis=0)
            
            if self.RS.shape[0] > self.large_cluster:
                rs_labels = KMeans(n_clusters=self.large_cluster//2, random_state=0) \
                    .fit_predict(np.array(self.RS, dtype=np.float64))
                for label in range(self.large_cluster):
                    id_with_label = np.argwhere(rs_labels==label)
                    if len(id_with_label) > 1:
                        id_with_label = id_with_label.reshape(id_with_label.shape[0])
                        feature_with_label = self.RS[id_with_label]
                        pids_with_label = []
                        for point in feature_with_label:
                            pids_with_label.append(self.pid_dict[tuple(point)])
                        self.CS.append(CompressionSet(np.array(feature_with_label, dtype=np.float64), pids_with_label))
                        self.CS_clusters += 1
                        self.CS_points += len(id_with_label)
                        self.RS = np.delete(self.RS, id_with_label, 0)
                        rs_labels = np.delete(rs_labels, id_with_label)
            
            i = 0
            while i < self.CS_clusters-1:
                j = i+1
                while j < self.CS_clusters:
                    if self.CS[i].mahalanobis_distance_cs(self.CS[j]) < np.sqrt(self.dimension)*2:
                        self.CS[i].merge(self.CS[j])
                        self.CS.pop(j)
                        self.CS_clusters -= 1
                    else:
                        j += 1
                i += 1

            if round_num == 5:
                i = 0
                while i < self.CS_clusters:
                    min_distance, min_label = np.sqrt(self.dimension)*2, -1
                    for j in range(self.num_cluster):
                        if self.DS[j].mahalanobis_distance_cs(self.CS[i]) < min_distance:
                            min_distance = self.DS[j].mahalanobis_distance_cs(self.CS[i])
                            min_label = j
                    if min_label != -1:
                        for pid in self.CS[i].pids:
                            self.predicts[pid] = min_label
                        self.DS[min_label].merge(self.CS[i])
                        self.DS_points += self.CS[i].N
                        self.CS_clusters -= 1
                        self.CS_points -= self.CS[i].N
                        self.CS.pop(i)
                    else:
                        i += 1
            self.print_state(round_num)


    def print_state(self, round_num):
        line = 'Round {}: {},{},{},{}\n'.format(round_num, self.DS_points, self.CS_clusters, 
                                             self.CS_points, self.RS.shape[0])
        self.fout.write(line)
        #for cluster in self.DS:
        #    print(cluster.N, cluster.SIGMA, cluster.SUM)

    def get_features(self, partition):
        str_list = self.data[partition]
        features = []
        for line in str_list:
            line_split = line.split(',')
            pid, p_cluster = int(line_split[0]), int(line_split[1])
            features.append(line_split[2:])
        return np.array(features)

    def print_predicts(self):
        self.fout.write('\nThe clustering results:\n')
        for k in sorted(self.predicts.keys()):
            line = '{},{}\n'.format(k, self.predicts[k])
            self.fout.write(line)
        self.fout.close()


#start = time.time()
bfr = BradleyFayyadReina(sys.argv[1], sys.argv[3], int(sys.argv[2]))
bfr.clustering()
bfr.print_predicts()
#print(time.time()-start)

"""
cluster_map = {}
outlier = 0
for pid in bfr.predicts.keys():
    if not bfr.ground_truth[pid] in cluster_map:
        cluster_map[bfr.ground_truth[pid]] = bfr.predicts[pid]
    if bfr.ground_truth[pid] == -1:
        outlier += 1
    if not cluster_map[bfr.ground_truth[pid]] == bfr.predicts[pid]:
        print(cluster_map[bfr.ground_truth[pid]], bfr.ground_truth[pid])
print(cluster_map)
print(outlier)
"""


