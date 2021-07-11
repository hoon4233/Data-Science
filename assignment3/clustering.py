import sys
import math
from collections import deque
from heapq import heapify, heappop
input_file = sys.argv[1]
N = int(sys.argv[2])
EPS = int(sys.argv[3])
MIN_PTS = int(sys.argv[4])
DIM_OBJ = 2
OUT_LIER = -1

def find_objs_neighbors(data_set, obj):
    global EPS, DIM_OBJ
    ret = deque()
    for other_obj in data_set :
        distance = 0
        for dim in range(1,DIM_OBJ+1,1):
            distance += math.pow(obj[dim]-other_obj[dim],2)
        distance = math.sqrt(distance)
        if distance <= EPS :
            ret.append(other_obj)
    return ret


def DBSCAN(data_set, obj_cluster):
    global EPS, MIN_PTS, OUT_LIER, cluster_id

    for obj_idx, obj in enumerate(obj_cluster) :
        if obj != None:
            continue
        neighbors = find_objs_neighbors(data_set, data_set[obj_idx])

        if len(neighbors) < MIN_PTS :
            obj_cluster[obj_idx] = OUT_LIER
        else :
            obj_cluster[obj_idx] = cluster_id
            while neighbors :
                now_obj = neighbors.popleft()
                now_neighbors = find_objs_neighbors(data_set, now_obj)
                if len(now_neighbors) >= MIN_PTS :
                    for now_nei in now_neighbors :
                        now_obj_idx = now_nei[0]
                        if obj_cluster[now_obj_idx] == None or obj_cluster[now_obj_idx] == OUT_LIER:
                            neighbors.append(now_nei)
                            obj_cluster[now_obj_idx] = cluster_id

            cluster_id += 1

data_set = []
len_data = 0
with open(input_file, 'r') as f:
    objs = f.readlines()
    for obj in objs :
        obj_idx, x, y = obj.strip().split("\t")
        data_set.append([ int(obj_idx), float(x), float(y) ])
        len_data += 1
obj_cluster = [ None for _ in range(len_data) ]
cluster_id = 0

DBSCAN(data_set, obj_cluster)

n_obj_in_clusters = [[0, i] for i in range(cluster_id)]
clusters = {i:[] for i in range(cluster_id)}

for idx, clu in enumerate(obj_cluster) :
    if clu == OUT_LIER or clu == None :
        continue
    n_obj_in_clusters[clu][0] -= 1
    clusters[clu].append(idx)

heapify(n_obj_in_clusters)
cluster_counter = 0
while n_obj_in_clusters :
    if cluster_counter == N :
        break
    count, clu_idx = heappop(n_obj_in_clusters)
    if count == 0 :
        continue
    file_name = input_file.replace(".txt","") + "_cluster_"+str(cluster_counter)+".txt"
    with open(file_name, 'w') as f :
        for obj_idx in clusters[clu_idx] :
            f.write(str(obj_idx)+"\n")
    cluster_counter += 1


