import math
import sklearn
import numpy as np
from sklearn.metrics import f1_score


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


def mu_class(feature):
    return np.mean(feature)


def sigma_class(feature):
    return math.sqrt(np.var(feature))


def calc_mu_sigma(feature_set):
    set_mu_sigma = [(mu_class(attribute), sigma_class(attribute)) for attribute in zip(*feature_set)]
    return set_mu_sigma


def calc_gaussian_density(x, mu, sigma):
    exponent = np.exp(-(math.pow(x - mu, 2) / (2 * math.pow(sigma, 2))))
    prob = (1 / (math.sqrt(2 * math.pi) * sigma)) * exponent
    return prob


def calc_class_conditional_densities(mean_std_set, test_ip_vector):
    class_conditional_prob = {}
    for classlabel, classmean_std in mean_std_set.items():
        for i in range(len(classmean_std)):
            mu, sigma = classmean_std[i]
            if (sigma != 0):
                x = test_ip_vector[i]
                if (classlabel not in class_conditional_prob):
                    class_conditional_prob[classlabel]=calc_gaussian_density(x, mu, sigma)
                else:
                    class_conditional_prob[classlabel] *= calc_gaussian_density(x, mu, sigma)
            else:
                continue
    return class_conditional_prob


def bayes_classifier(train_data, train_labels, test_data,test_labels):
    if int((sklearn.__version__).split(".")[1]) < 18:
        from sklearn.cross_validation import train_test_split
    # otherwise we're using at lease version 0.18
    else:
        from sklearn.model_selection import train_test_split



    class_partition_set = class_partition(train_data, train_labels)
    prior_probs = {}
    for class_value, features in class_partition_set.items():
        prior_probs[class_value] = len(features) / len(train_data)
    mean_std_set = {}
    for class_value, features in class_partition_set.items():
        mean_std_set[class_value] = calc_mu_sigma(features)
    predictions = []
    accuracy = 0
    for i in range(len(test_data)):
        test_vector = test_data[i]
        probabilities = calc_class_conditional_densities(mean_std_set, test_vector)
        posterior_prob={}
        for key in probabilities:
            posterior_prob[key]=prior_probs[key]*probabilities[key]
        predicted_label = None
        prediction_prob = -1
        for class_value, probability in posterior_prob.items():
            if predicted_label is None or probability > prediction_prob:
                prediction_prob = probability
                predicted_label = class_value

        predictions.append(predicted_label)

    for x in range(len(test_data)):
        if test_labels[x] == predictions[x]:
            accuracy += 1
    accuracy_mycode = (accuracy / len(test_data)) * 100.0
    f1_score_macro = f1_score(test_labels, predictions, average='macro')
    f1_score_micro = f1_score(test_labels, predictions, average='micro')
    print("Accuracy obtained with my code: ", accuracy_mycode)
    print("F1-Score(Macro) obtained with my code: ", f1_score_macro)
    print("F1-Score(Micro) obtained with my code: ", f1_score_micro)
    return accuracy_mycode, f1_score_macro, f1_score_micro

    # Using Sklearn Package
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(data_matrix, label_matrix).predict(data_matrix)
    Total = data_matrix.shape[0]
    predicted = Total - (label_matrix != y_pred).sum()
    accuracy = (predicted / Total) * 100.0
    print("Accuracy using Library:", accuracy)
