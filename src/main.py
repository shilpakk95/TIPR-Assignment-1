import numpy as np
import math
import sklearn
import nn
import time
import lsh
import bayes
import twitter
import collections
import projections
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import operator
# endregion

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

def plot_data(x,y,x_label,y_label,title,file_name,yticky):
    max = 0
    index = 0
    for i in range(len(y)):
        if (y[i] > max):
            max = y[i]
            index = x[i]
    text = "highest"
    plt.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.yticks(yticky)
    for i in range(len(y)):
        plt.text(x[i], y[i], str(round(y[i],2)))
    plt.text(index, max/2, text)
    plt.title(title)
    plt.savefig(file_name)
    plt.close()

if __name__ == '__main__':
    Accuracy={}
    F1ScoreMacro={}
    F1ScoreMicro={}
    if int((sklearn.__version__).split(".")[1]) < 18:
        from sklearn.cross_validation import train_test_split
    else:
        from sklearn.model_selection import train_test_split
    print('Welcome to the world of high and low dimensions!')
    data_set_name = input('Enter dataset name:')
    print(str(data_set_name)," dataSet:")
    if(data_set_name.lower()=='twitter'):
        twitter.twitter()
    else:
        if (data_set_name.lower() == 'dolphin'):
            # Dolphin Data Set
            ip_file_path = 'data/dolphins/'
            ip_file_name = 'dolphins.csv'
            out='dolphins'
            ip_label_file_name = 'dolphins_label.csv'
            data_matrix = np.genfromtxt(ip_file_path + ip_file_name, delimiter=' ')
        else:
            #Pubmed Data Set
            ip_file_path = 'data/pubmed/'
            ip_file_name = 'pubmed.csv'
            out='dolphins'
            ip_label_file_name = 'pubmed_label.csv'
            data_matrix = np.genfromtxt(ip_file_path+ip_file_name, delimiter=' ')
        label_matrix = np.genfromtxt(ip_file_path+ip_label_file_name, delimiter=' ')

        c=len(data_matrix[0])
        print("Bayes Classifier:")
        K = 2
        while (K <= int(c / 2)):
            print("Reduced Dimension :",K)
            Accuracy[K]=0
            F1ScoreMacro[K] =0
            F1ScoreMicro[K] =0
            random_data_matrix=projections.random_projection(data_matrix,c,K,ip_file_path,out)

            (train_data, test_data, train_labels, test_labels) = train_test_split(random_data_matrix, label_matrix, test_size=0.25,
                                                                          random_state=42)
            accuracy_mycode, f1_score_macro, f1_score_micro = bayes.bayes_classifier(train_data, train_labels, test_data,
                                                                                     test_labels)
            Accuracy[K]=accuracy_mycode
            F1ScoreMacro[K]=f1_score_macro
            F1ScoreMicro[K]=f1_score_micro
            K = K * 2
        index=[]
        accuracy_list=[]
        for k,accuracy in Accuracy.items():
            index.append(k)
            accuracy_list.append(accuracy)
        #title="Accuracy v/s Dimension for nearest neighbor classifier for "+ str(data_set_name)+" Data Set."
        #figname="task_3_NN_Accuracy_"+str(data_set_name)+".png"
        #accuracy_mycode, f1_score_macro, f1_score_micro = nn.nearest_neighbour(train_data, train_labels, test_data,test_labels)
        #plot_data(index,accuracy_list,'Dimension','Accuracy',title,figname,np.arange(0,100,step=10))

        title = "Accuracy v/s Dimension for Bayes classifier for " + str(data_set_name.upper()) + " Data Set."
        figname="task_3_Bayes_Accuracy_"+str(data_set_name.upper())+".png"
        plot_data(index,accuracy_list,'Dimension','Accuracy',title,figname,np.arange(0,100,step=10))

        index = []
        F1MacroList = []
        for k, f1_score_macro in F1ScoreMacro.items():
            index.append(k)
            F1MacroList.append(f1_score_macro)
        #title = "F1-Score(Macro) v/s Dimension for nearest neighbor classifier for " + str(data_set_name) + " Data Set."
        #figname = "task_3_NN_F1-Score(Macro)_" + str(data_set_name) + ".png"
        title = "F1-Score(Macro) v/s Dimension for Bayes classifier for " + str(data_set_name.upper()) + " Data Set."
        figname = "task_3_Bayes_F1-Score(Macro)_" + str(data_set_name.upper()) + ".png"
        plot_data(index, F1MacroList, 'Dimension', 'F1-Score(Macro)', title, figname,np.arange(0,1,step=0.1))

        index = []
        F1MicroList = []
        for k, f1_score_micro in F1ScoreMicro.items():
            index.append(k)
            F1MicroList.append(f1_score_micro)
        #title = "F1-Score(Micro) v/s Dimension for nearest neighbor classifier for " + str(data_set_name) + " Data Set."
        #figname = "task_3_NN_F1-Score(Micro)_" + str(data_set_name) + ".png"
        title = "F1-Score(Micro) v/s Dimension for Bayes classifier for " + str(data_set_name.upper()) + " Data Set."
        figname = "task_3_Bayes_F1-Score(Micro)_" + str(data_set_name.upper()) + ".png"
        plot_data(index, F1MicroList, 'Dimension', 'F1-Score(Micro)', title, figname,np.arange(0,1,step=0.1))

        #nn
        print("Nearest neighbor classifier")
        K = 2
        while (K <= int(c / 2)):
            print("Reduced Dimension :",K)
            Accuracy[K]=0
            F1ScoreMacro[K] =0
            F1ScoreMicro[K] =0
            random_data_matrix=projections.random_projection(data_matrix,c,K,ip_file_path,out)

            (train_data, test_data, train_labels, test_labels) = train_test_split(random_data_matrix, label_matrix, test_size=0.25,
                                                                          random_state=42)
            accuracy_mycode, f1_score_macro, f1_score_micro = nn.nearest_neighbour(train_data, train_labels, test_data,
                                                                                     test_labels)
            Accuracy[K]=accuracy_mycode
            F1ScoreMacro[K]=f1_score_macro
            F1ScoreMicro[K]=f1_score_micro
            K = K * 2
        index=[]
        accuracy_list=[]
        for k,accuracy in Accuracy.items():
            index.append(k)
            accuracy_list.append(accuracy)
        title="Accuracy v/s Dimension for nearest neighbor classifier for "+ str(data_set_name)+" Data Set."
        figname="task_3_NN_Accuracy_"+str(data_set_name)+".png"
        #accuracy_mycode, f1_score_macro, f1_score_micro = nn.nearest_neighbour(train_data, train_labels, test_data,test_labels)
        plot_data(index,accuracy_list,'Dimension','Accuracy',title,figname,np.arange(0,100,step=10))

        #title = "Accuracy v/s Dimension for Bayes classifier for " + str(data_set_name.upper()) + " Data Set."
        #figname="task_3_Bayes_Accuracy_"+str(data_set_name.upper())+".png"
        #plot_data(index,accuracy_list,'Dimension','Accuracy',title,figname,np.arange(0,100,step=10))

        index = []
        F1MacroList = []
        for k, f1_score_macro in F1ScoreMacro.items():
            index.append(k)
            F1MacroList.append(f1_score_macro)
        title = "F1-Score(Macro) v/s Dimension for nearest neighbor classifier for " + str(data_set_name) + " Data Set."
        figname = "task_3_NN_F1-Score(Macro)_" + str(data_set_name) + ".png"
        #title = "F1-Score(Macro) v/s Dimension for Bayes classifier for " + str(data_set_name.upper()) + " Data Set."
        #figname = "task_3_Bayes_F1-Score(Macro)_" + str(data_set_name.upper()) + ".png"
        plot_data(index, F1MacroList, 'Dimension', 'F1-Score(Macro)', title, figname,np.arange(0,1,step=0.1))

        index = []
        F1MicroList = []
        for k, f1_score_micro in F1ScoreMicro.items():
            index.append(k)
            F1MicroList.append(f1_score_micro)
        title = "F1-Score(Micro) v/s Dimension for nearest neighbor classifier for " + str(data_set_name) + " Data Set."
        figname = "task_3_NN_F1-Score(Micro)_" + str(data_set_name) + ".png"
        #title = "F1-Score(Micro) v/s Dimension for Bayes classifier for " + str(data_set_name.upper()) + " Data Set."
        #figname = "task_3_Bayes_F1-Score(Micro)_" + str(data_set_name.upper()) + ".png"
        plot_data(index, F1MicroList, 'Dimension', 'F1-Score(Micro)', title, figname,np.arange(0,1,step=0.1))


        #prior_dict=prior_density(train_labels)
        #accuracy_mycode, f1_score_macro, f1_score_micro=nn.nearest_neighbour(train_data,train_labels,test_data,test_labels)
        lsh.lsh(train_data, train_labels, test_data, test_labels)
        #bayes.bayes_classifier(data_matrix,label_matrix)
