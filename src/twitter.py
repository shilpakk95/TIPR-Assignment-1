import numpy as np
import sklearn
import lsh
import nn
import collections
import math
# endregion

def class_partition(train_data_set,train_labels):
    class_partition_set={}
    for i in range(len(train_data_set)):
        data_element=list(train_data_set[i])
        data_label=train_labels[i]
        if(data_label!=3):
            if(data_label not in class_partition_set):
                class_partition_set[data_label]=[]
            class_partition_set[data_label].append(data_element)
    return class_partition_set


def prior_density(train_labels):
    sample_size=len(train_labels)
    prior_dict={}
    label_size={}
    for label in train_labels:
        if label not in label_size:
            label_size[label]=1
        else:
            label_size[label] += 1
    for class_value, count in label_size.items():
        prior_dict[class_value] = count / sample_size
    return prior_dict

def twitter():
    if int((sklearn.__version__).split(".")[1]) < 18:
        from sklearn.cross_validation import train_test_split
    else:
        from sklearn.model_selection import train_test_split
    #Twitter Data Set

    ip_file_path = 'data/twitter/'
    ip_file_name = 'twitter.txt'
    out = 'twitter'
    ip_label_file_name = 'twitter_label.txt'
    test_file_name = 'twitter_test.txt'
    test_label_file_name = 'twitter_test_label.txt'
    vocabulary = {}
    input_file=open(ip_file_path + ip_file_name, 'r+')
    file_lines=input_file.readlines()
    word_matrix=[]
    for line in file_lines:
        line=line.strip()
        sentence = line.split(" ")
        word_matrix.append(sentence)
    label_matrix = np.genfromtxt(ip_file_path + ip_label_file_name, delimiter=' ')
    (train_data, test_data, train_labels, test_labels) = train_test_split(word_matrix,
                                                                      label_matrix, test_size=0.20, random_state=42)

    #prior_dict = prior_density(train_labels)
    for sentence in train_data:
        for word in sentence:
            if (word not in vocabulary):
                vocabulary[word] = 1
            else:
                vocabulary[word] = vocabulary[word] + 1
    #orderedDictionary = collections.OrderedDict(sorted(vocabulary.items(), key=lambda x: x[1]))
    updated_voc = {k: v for k, v in vocabulary.items() if v >3}
    data_matrix = np.zeros((len(train_data), len(updated_voc)))
    test_matrix = np.zeros((len(test_data), len(updated_voc)))
    i=0
    for sentence in train_data:
        for word in sentence:
            if (word not in updated_voc):
                continue
            index = list(updated_voc.keys()).index(word)
            data_matrix[i][index] = data_matrix[i][index] + 1
        i = i + 1
    i=0
    for sentence in test_data:
        for word in sentence:
            if (word not in updated_voc):
                continue
            index = list(updated_voc.keys()).index(word)
            test_matrix[i][index] = test_matrix[i][index] + 1
        i = i + 1

    class_partition_set = class_partition(data_matrix, train_labels)
    prior_prob_set = {}
    for class_value, features in class_partition_set.items():
        prior_prob_set[class_value] = len(features) / len(train_data)

    #Bayes On Twitter
    class_densities_words = {}
    for class_value, features in class_partition_set.items():
        voc_size = len(features[0])
        matrix = np.matrix(features)
        sum_elements = np.sum(matrix)
        deno = voc_size + sum_elements
        c_sum = matrix.sum(axis=0).tolist()
        densities = []
        for i in range(0, len(c_sum[0])):
            num = c_sum[0][i] + 1
            prob = (num) / deno
            densities.append(prob)
        class_densities_words[class_value] = densities
    predictions = []
    accuracy = 0
    for i in range(len(test_matrix)):
        test_vector=np.zeros((len(updated_voc)))
        #testVector = list(testData[0])
        for word in test_data[i]:
            if (word not in updated_voc):
                continue
            index = list(updated_voc.keys()).index(word)
            test_vector[index] += 1
        probabilities = {}
        for class_value, densities in class_densities_words.items():
            prob = prior_prob_set[class_value]
            for i in range(0, len(test_vector)):
                if (test_vector[i] != 0):
                    if (test_vector[i] == 1):
                        prob = prob * densities[i]
                    else:
                        prob = prob * np.power(densities[i], test_vector[i])
            probabilities[class_value] = prob
        predictions.append(max(probabilities, key=probabilities.get))
    for x in range(len(test_data)):
        if test_labels[x] == predictions[x]:
            accuracy += 1
    accuracy_mycode = (accuracy / len(test_data)) * 100.0
    print("Accuracy without using library ", accuracy_mycode)

    lsh.lsh(data_matrix,train_labels,test_matrix,test_labels)
    nn.nearest_neighbour(data_matrix,train_labels,test_matrix,test_labels)