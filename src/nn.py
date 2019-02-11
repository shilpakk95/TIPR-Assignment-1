import math
import lsh
import sklearn
import operator
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


def euclidean_distance(v1, v2, length):
    distance = 0
    D = np.subtract(v1,v2)
    distance = math.sqrt(np.dot(D,np.transpose(D)))
    return distance


def get_neighbors(train_data,train_labels,test_row):
    neighbor=[]
    length = len(test_row)
    for i in range(len(train_data)):
        dist = euclidean_distance(test_row, train_data[i], length)
        neighbor.append((train_labels[i],dist))
    neighbor.sort(key=operator.itemgetter(1))
    return neighbor


def get_knn(distance,k):
    neighbors = []
    for i in range(0,k):
        neighbors.append(distance[i][0])
    return neighbors

def get_neighbor_labels(neighbors):
    classlab = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classlab:
            classlab[response] += 1
        else:
            classlab[response] = 1
    sortedlab = sorted(classlab.items(), key=operator.itemgetter(1), reverse=True)
    return sortedlab[0][0]

def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] is predictions[x]:
            correct=correct+1
    return (correct/float(len(test_set))) * 100.0


def nearest_neighbour(train_data,train_labels,test_data,test_labels):
    k = 7
    predictions = []
    accuracies = []
    correct_prediction = 0
    for i in range(0, len(test_data)):
        distance = get_neighbors(train_data, train_labels, test_data[i])
        knn = get_knn(distance, k)
        result = get_neighbor_labels(knn)
        predictions.append(result)
        if (test_labels[i] == result):
            correct_prediction += 1
    accuracy = (correct_prediction / len(test_data)) * 100.0
    f1_score_macro=f1_score(test_labels, predictions, average='macro')
    f1_score_micro = f1_score(test_labels, predictions, average='micro')
    print("Accuracy obtained with my code: ", accuracy)
    print("F1-Score(Macro) obtained with my code: ", f1_score_macro)
    print("F1-Score(Micro) obtained with my code: ", f1_score_micro)


    #Using Sklearn Package

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    score = model.score(test_data, test_labels)
    accuracies.append(score * 100)
    print("Test Accuracy using Sklearn Library: ", accuracies)
    print("Test F1-Score(Macro) using Sklearn Library: ",
          f1_score(test_labels, predictions, average='macro'))
    print("Test F1-Score(Micro) using Sklearn Library: ",
          f1_score(test_labels, predictions, average='micro'))

    return accuracy,f1_score_macro,f1_score_micro