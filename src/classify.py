import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def classify_logistic(embed_fpath,label_fpath, train_label_fpath,test_label_fpath):
    hyper_node_embeddings = pd.read_csv(embed_fpath, sep=',', header=None).values

    f_label=open(label_fpath,"r")
    all_node_list=[]
    for line in f_label:
        node=line.split(",")[0].strip()
        all_node_list.append(node)
    f_label.close()

    f_train_label = open(train_label_fpath, "r")
    train_label_list = []
    for line in f_train_label:
        node = line.split(",")[0].strip()
        label = line.split(",")[1].strip()
        train_label_list.append([node, label])
    f_train_label.close()

    f_test_label = open(test_label_fpath, "r")
    test_label_list = []
    for line in f_test_label:
        node = line.split(",")[0].strip()
        label = line.split(",")[1].strip()
        test_label_list.append([node, label])
    f_test_label.close()
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    for i in range(len(train_label_list)):
        node=train_label_list[i][0]
        label=train_label_list[i][1]
        y_train.append(label)
        X_train.append(hyper_node_embeddings[all_node_list.index(node)])

    for i in range(len(test_label_list)):
        node=test_label_list[i][0]
        label=test_label_list[i][1]
        y_test.append(label)
        X_test.append(hyper_node_embeddings[all_node_list.index(node)])

    avg_macro, avg_micro, std_micro, std_macro, avg_weighted_f1, std_weighted_f1 = [[] for i in range(6)]
    # n_times = 100
    # macro, micro, weighted = [[] for _ in range(3)]  # reset for each percentage
    # for _ in np.arange(n_times): # don't fix the random number :P  otherwise no use of doing it 10 times
    model = LogisticRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1_macro=f1_score(y_test, y_pred, average='macro')
    f1_micro=f1_score(y_test, y_pred, average='micro')
    f1_weighted=f1_score(y_test, y_pred, average='weighted')
    accuracy=accuracy_score(y_test, y_pred)

    print('\nAccuracy :: \n', accuracy)
    print('\nMacro-F1 scores :: \n', f1_macro)
    print('\nMicro-F1 scores :: \n', f1_micro)
    print('\nWeighted-F1 scores :: \n', f1_weighted)
if __name__ == "__main__":
    classify_logistic("../data/toy_hyper_node_embeddings.csv",
                      "../data/toy_hyper_level.csv",
                      "../data/toy_hyper_train.csv",
                      "../data/toy_hyper_test.csv")

    # classify_logistic("../data/citeseer_hyper_node_embeddings.csv",
    #                   "../data/citeseer_hyper_level.csv",
    #                   "../data/citeseer_hyper_train.csv",
    #                   "../data/citeseer_hyper_test.csv")