import operator
import random
import math
import collections
import numpy as np
from sklearn.neighbors import LSHForest

'''
k=10
L=20
hash_index=[]
for i in range(L):
    h = []
    for j in range(k):
        rand_num=random.randrange(len(data_matrix[0]))
        while(rand_num in hash_index):
            rand_num = random.randrange(len(data_matrix[0]))
        h.append(rand_num)
    hash_index.append(h)
for g in hash_index:
    hash_table={}
    for data in data_matrix:
        min_hash=0
        for j in range(k):
            h_index=g[j]
            d=data[h_index]
            min_hash = min(min_hash,d)
'''

def pre_processing(data_matrix,train_labels):
    hash_details = {}
    hashtable_set={}
    w=5
    vector_size=len(data_matrix[0])
    k=int(vector_size)
    for i in range(20):
        random_indices = random.sample(range(k), k)
        hashtable = {}
        data_index=0
        #r = np.random.normal(0, 1, (1,vectorSize))
        r = np.random.normal(0, 1, (1, k))
        b = np.random.uniform(0, w)
        info=[]
        info.append(r)
        info.append(b)
        info.append(w)
        info.append(random_indices)
        hash_details[i]=info
        sign={}
        sign['pos']=0
        sign['neg'] = 0
        for data in data_matrix:
            reduced_data =[data[x] for x in random_indices]
            #data=np.reshape(data,len(data),1)
            #p=np.dot(data,np.transpose(r))
            p = np.dot(reduced_data, np.transpose(r))
            if(p>=0):
                sign['pos'] += 1
            else:
                sign['neg'] += 1
            #hashIndex = int(np.floor((p/w)+b))
            hash_index = int(np.floor((p + b) / w))
            if (hash_index not in hashtable):
                element=[]
                element.append(data_index)
                #element.append(trainLabels[dataIndex])
                hashtable[hash_index]=element
            else:
                hashtable[hash_index].append(data_index)
            data_index +=1
        ord = collections.OrderedDict(sorted(hashtable.items()))
        hashtable_set[i]=ord
    return hash_details,hashtable_set

def euclidean_distance(v1, v2, length):
    distance = 0
    D = np.subtract(v1,v2)
    distance = math.sqrt(np.dot(D,np.transpose(D)))
    return distance

def query_test_vector(test_vector,data_matrix,train_labels,hash_details,hashtable_set):
    lhd=len(hash_details)
    neighbors = []
    k=5
    ltv = len(test_vector)
    hash_param=list(hash_details.values())
    hashtables=list(hashtable_set.values())
    for i in range(lhd):
        limit = k
        distance = []
        r = hash_param[i][0]
        b = hash_param[i][1]
        w = hash_param[i][2]
        random_indices = hash_param[i][3]
        hashTB=hashtables[i]
        reduced_data = [test_vector[x] for x in random_indices]
        #data = np.reshape(testVector, len(testVector), 1)
        p = np.dot(reduced_data, np.transpose(r))
        #hashIndex = int(np.floor((p / w) + b))
        hash_index = int(np.floor((p + b)/ w))
        if (hash_index not in hashTB):
            continue
        data_indices=hashTB[hash_index]

        #Take Majoirity Label from mapped labels
        '''
        neighbourLabel={}
        for j in dataIndices:
            classLabel=trainLabels[j]
            if (classLabel not in neighbourLabel):
                neighbourLabel[classLabel]=1
            else:
                neighbourLabel[classLabel] += 1
        neighbors.append(max(neighbourLabel.items(), key=operator.itemgetter(1))[0])
        '''


        #Calculate Eucledian Distance with mapped points and take closed distance point
        for j in data_indices:
            dist = euclidean_distance(test_vector, data_matrix[j], ltv)
            distance.append((train_labels[j], dist,j))
        distance.sort(key=operator.itemgetter(1))
        limit=min(k,len(distance))
        for l in range(0, limit):
            neighbors.append(distance[l])
    return neighbors

def get_neighbor_label(neighbors):
    score={}
    for i in range(len(neighbors)):
        data_index=neighbors[i][2]
        if (data_index not in score):
            score[data_index]=1
        else:
            score[data_index] += 1
    score_up = sorted(score.items(), key=operator.itemgetter(1),reverse=True)
    return score_up[0][0]

def majority_label(nearestNeighborslabels):
    label_count = {}
    for labels in nearestNeighborslabels:
        if (labels not in label_count):
            label_count[labels] = 1
        else:
            label_count[labels] += 1
    return max(label_count.items(), key=operator.itemgetter(1))[0]

def lsh(data_matrix,train_labels,test_matrix,test_labels):
    hash_details,hashtable_set=pre_processing(data_matrix,train_labels)
    prediction_level={}
    correct_prediction = 0
    label_set=set(train_labels)
    for k in label_set:
        prediction_level[k]=0
    for x in range(len(test_matrix)):
        nearestNeighbors = query_test_vector(test_matrix[x], data_matrix, train_labels, hash_details, hashtable_set)
        result=train_labels[get_neighbor_label(nearestNeighbors)]
        count=0
        if (int(test_labels[x]) == int(result)):
            correct_prediction += 1
    accuracy_mycode = (correct_prediction / len(test_matrix)) * 100.0
    print("Locality Sensitive Hashing: Accuracy without using library", accuracy_mycode)
    # endregion

    # region Sklearn Library LSH Code
    accuracy=0
    lshf = LSHForest(random_state=42)
    lshf.fit(data_matrix)
    LSHForest(min_hash_match=4, n_candidates=50, n_estimators=10,
              n_neighbors=7, radius=1.0, radius_cutoff_ratio=0.9,
              random_state=42)
    distances, indices = lshf.kneighbors(test_matrix, n_neighbors=7)
    for x in range(len(test_matrix)):
        nearest_neighbour_indices=indices[x]
        for i in nearest_neighbour_indices:
            prediction_level[train_labels[i]] += 1
        prediction_label=max(prediction_level.items(), key=operator.itemgetter(1))[0]
        if(prediction_label==test_labels[x]):
            accuracy += 1
    accuracy_lib = 0
    accuracy_lib = (accuracy / len(test_matrix)) * 100.0
    print("Locality Sensitive Hashing:Sklearn Library Accuracy ", accuracy_lib)