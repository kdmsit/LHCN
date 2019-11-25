'''
Author-Kishalay Das
This is the part of code where we used multi-label hypothesis.
'''

# region Library
from __future__ import division
from __future__ import print_function
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
import plotly.graph_objects as go
import plotly
from sklearn.model_selection import train_test_split
import time
import pickle
import random
import tensorflow as tf
from utils import *
from models import GCN
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import statistics
# endregion

def createhypergraph(original_graph_edge_list):
    '''
            This procedure will create the hypergraph from actual graph edgelist file.
            '''
    # region Read Original Graph Edgelist
    f = open(original_graph_edge_list, "r")
    citing_cited_paper_list = []
    citing_paper_list = []
    for line in f:
        cited_paper_id = int(line.split(' ')[0].strip())  # \t
        citing_paper_id = int(line.split(' ')[1].strip())
        citing_cited_paper_list.append([int(cited_paper_id), int(citing_paper_id)])
        if (citing_paper_id not in citing_paper_list):
            citing_paper_list.append(citing_paper_id)
    f.close()
    # endregion

    # region Collect All Papers Cited in a particular Paper
    hyper_edge_list = []
    for citing_paper_id in citing_paper_list:
        hyper_edge = [citing_paper_id]
        for paper in citing_cited_paper_list:
            if (citing_paper_id == paper[1]):
                hyper_edge.append(paper[0])
        hyper_edge_list.append(hyper_edge)
    # endregion

    hyper_edge_dic = {}
    for i in range(len(hyper_edge_list)):
        hyper_edge_dic[i] = hyper_edge_list[i]
    pickle_out = open("hypergraph_" + str(FLAGS.dataset) + ".pickle", "wb")
    pickle.dump(hyper_edge_dic, pickle_out)
    pickle_out.close()

    # region Find Isolated Edges and their index nodes in Hyper Graph
    isolatedflag = np.zeros(len(hyper_edge_list))
    for i in range(len(isolatedflag)):
        isisolated = 1
        for j in range(len(isolatedflag)):
            if (i != j):
                commonedges = set(hyper_edge_list[i]).intersection(set(hyper_edge_list[j]))
                if (len(commonedges) != 0):
                    isisolated = 0
                    commonedges.clear()
                    break
        if (isisolated == 1):
            isolatedflag[i] = 1
    non_isolated_hyper_edge_list = []
    isolated_hyper_edge_list = []
    for i in range(len(isolatedflag)):
        if (isolatedflag[i] == 0):
            non_isolated_hyper_edge_list.append(hyper_edge_list[i])
        else:
            isolated_hyper_edge_list.append(hyper_edge_list[i])
    # endregion
    # region Count Number of hyper nodes and edges
    totalhypernodes = []
    for edge in non_isolated_hyper_edge_list:
        for node in edge:
            totalhypernodes.append(node)
    print("Number of Hyper-Nodes", len(list(dict.fromkeys(totalhypernodes))))
    print("Number of Hyper-Edges", len(hyper_edge_list))
    # endregion

    return non_isolated_hyper_edge_list

def savepickel(edgelist_path,label_path,feature_path):
    # region Save Edgelist to pickel
    f = open(edgelist_path, "r")
    citing_cited_paper_list = []
    citing_paper_list = []
    for line in f:
        cited_paper_id = int(line.split(' ')[0].strip())  # \t
        citing_paper_id = int(line.split(' ')[1].strip())
        citing_cited_paper_list.append([int(cited_paper_id), int(citing_paper_id)])
        if (citing_paper_id not in citing_paper_list):
            citing_paper_list.append(citing_paper_id)
    f.close()

    hyper_edge_list = []
    for citing_paper_id in citing_paper_list:
        hyper_edge = [citing_paper_id]
        for paper in citing_cited_paper_list:
            if (citing_paper_id == paper[1]):
                hyper_edge.append(paper[0])
        hyper_edge_list.append(hyper_edge)
    node_list=set()
    for edge in hyper_edge_list:
        for i in edge:
            node_list.add(i)
    print(len(node_list))
    hyper_incident_matrix=np.zeros((len(node_list),len(hyper_edge_list)))
    for i in range(len(hyper_edge_list)):
        for edge in hyper_edge_list[i]:
            hyper_incident_matrix[edge][i]=1
    pickle_out = open("hypergraph_incident_" + str(FLAGS.dataset) + ".pickle", "wb")
    pickle.dump(hyper_incident_matrix, pickle_out)
    pickle_out.close()
    hyper_edge_dic = {}
    for i in range(len(hyper_edge_list)):
        hyper_edge_dic[i] = hyper_edge_list[i]
    pickle_out = open("hypergraph_" + str(FLAGS.dataset) + ".pickle", "wb")
    pickle.dump(hyper_edge_dic, pickle_out)
    pickle_out.close()


    # endregion

    # region Save Labels to pickel
    labelset = pd.read_csv(label_path, header=None).values
    hyper_label_set = set()
    hyper_label_list=[]
    for label in labelset:
        hyper_label_set.add(label[0])
        hyper_label_list.append(label[0])
    pickle_out = open("labels_" + str(FLAGS.dataset) + ".pickle", "wb")
    pickle.dump(hyper_label_list, pickle_out)
    pickle_out.close()
    hyper_label_list=[]
    for label in labelset:
        one_hot_y=[0]*len(hyper_label_set)
        one_hot_y[label[0]]=1
        hyper_label_list.append(one_hot_y)
    pickle_out = open("labels_onehot" + str(FLAGS.dataset) + ".pickle", "wb")
    pickle.dump(hyper_label_list, pickle_out)
    pickle_out.close()
    # endregion

    # region Save features to pickle
    featureset = pd.read_csv(feature_path, header=None).to_numpy()
    pickle_out = open("features_" + str(FLAGS.dataset) + ".pickle", "wb")
    pickle.dump(featureset, pickle_out)
    pickle_out.close()
    # endregion

def createhypergraphun(original_graph_edge_list):
    '''
    This procedure will create the hypergraph from actual graph edgelist file.
    '''
    # region Read Original Graph Edgelist
    f = open(original_graph_edge_list, "r")
    citing_cited_paper_list = []
    citing_paper_list = []
    for line in f:
        cited_paper_id = int(line.split(' ')[0].strip())  # \t
        citing_paper_id = int(line.split(' ')[1].strip())
        citing_cited_paper_list.append([int(cited_paper_id), int(citing_paper_id)])
        if (citing_paper_id not in citing_paper_list):
            citing_paper_list.append(citing_paper_id)
    f.close()
    #endregion

    # region Collect All Papers Cited in a particular Paper
    hyper_edge_list = []
    for citing_paper_id in citing_paper_list:
        hyper_edge = [citing_paper_id]
        for paper in citing_cited_paper_list:
            if (citing_paper_id == paper[1]):
                hyper_edge.append(paper[0])
        hyper_edge_list.append(hyper_edge)
    for citing_paper_id in citing_paper_list:
        hyper_edge = []
        for paper in citing_cited_paper_list:
            if (citing_paper_id == paper[0]):
                if citing_paper_id not in hyper_edge:
                    hyper_edge.append(citing_paper_id)
                hyper_edge.append(paper[1])
        if (len(hyper_edge)!=0):
            hyper_edge_list.append(hyper_edge)
    # endregion

    # region Find Isolated Edges and their index nodes in Hyper Graph
    isolatedflag = np.zeros(len(hyper_edge_list))
    for i in range(len(isolatedflag)):
        isisolated = 1
        for j in range(len(isolatedflag)):
            if (i != j):
                commonedges = set(hyper_edge_list[i]).intersection(set(hyper_edge_list[j]))
                if (len(commonedges) != 0):
                    isisolated = 0
                    commonedges.clear()
                    break
        if (isisolated == 1):
            isolatedflag[i] = 1
    non_isolated_hyper_edge_list = []
    isolated_hyper_edge_list = []
    for i in range(len(isolatedflag)):
        if (isolatedflag[i] == 0):
            non_isolated_hyper_edge_list.append(hyper_edge_list[i])
        else:
            isolated_hyper_edge_list.append(hyper_edge_list[i])
    # endregion
    # region Count Number of hyper nodes and edges
    totalhypernodes = []
    for edge in non_isolated_hyper_edge_list:
        for node in edge:
            totalhypernodes.append(node)
    print("Number of Hyper-Nodes", len(list(dict.fromkeys(totalhypernodes))))
    print("Number of Hyper-Edges", len(hyper_edge_list))
    # endregion
    return non_isolated_hyper_edge_list

def createhypegraphlabel(hyper_edge_list,label_path):
    # region Read label file for original graph
    labelset =pd.read_csv(label_path,header=None).values
    # endregion
    hyper_node_label_list = []
    for hyper_edge in hyper_edge_list:
        for hyper_edge_node in hyper_edge:
            if(len(hyper_node_label_list)!=0):
                if hyper_edge_node not in [i[0] for i in hyper_node_label_list]:
                        hyper_node_label_list.append([hyper_edge_node, labelset[hyper_edge_node][0]])
            else:
                hyper_node_label_list.append([hyper_edge_node, labelset[hyper_edge_node][0]])
    hyper_node_label_list=sorted(hyper_node_label_list,key=lambda x: (x[0],x[1]))
    return hyper_node_label_list

def splitHyperNodes(hyper_node_label_list,testset_size):
    '''
    This procedure will split the hyper node labels into train and test
    '''
    hyper_node_list = [node for node,label in hyper_node_label_list]
    hyper_label_list = [label for node, label in hyper_node_label_list]
    #Split hyper graph nodes and labels into train and test split
    train_node, test_node, train_y, test_y = train_test_split(hyper_node_list, hyper_label_list,test_size = testset_size)  #, random_state = 42
    return list(zip(train_node,train_y)),list(zip(test_node,test_y))

def process_feature_line_graph(nodeset,content,hyper_edgelist):
    '''
    This procedure will generate feature_set of line graph from the feature set of original graph
    :param nodeset: list of nodes of original graph.
    :param content: Feature set of the original graph indexed by nodes of nodeset.
    :param hyperedges:  Set of hyperedges of the hypergraph.
    :param hyperedgelist: Dictionary with format :citing_paper_id : <cited_paper_ids in comma separated way>
    :return:line_features:Feature set of line graph
    '''
    # region gererate line features.
    '''
    This region of code will generate features for each node of line graph.
    For each line node there would be a hyper edge in hyper graph which will contain n number of nodes(n>=1)
    We will collect features of nodes of hyper edge in hyper_edge_features list.
    The we take their mean according to axis=0 and set that as the feature of line graph node.
    hyper_edge_features => features of all the nodes of a particular hyperedge.
    line_features => feature of a line graph node.
    '''
    line_features=[]
    for hypedges in hyper_edgelist:
        hyper_edge_features=[]
        for edges in hypedges:
            # i=list(nodeset).index(edges)
            hyper_edge_features.append(content[edges])
        # line_features.append(np.sum(hyper_edge_features,axis=0))
        line_features.append(np.mean(hyper_edge_features, axis=0))
    # endregion
    return line_features

def createLineGraph(hyper_edge_list,hyper_node_label_list,hyper_train,content_path):
    '''
    This procedure will create line graph corresponding to Hypergraph.
    '''

    content=pd.read_csv(content_path,header=None).values
    train_nodeset=[lis[0] for lis in hyper_train]
    train_labelset=[lis[1] for lis in hyper_train]
    nodeset=[node for node,label in hyper_node_label_list]
    #Generate the features of the line graph nodes.
    line_features=process_feature_line_graph(nodeset,content,hyper_edge_list)

    # region reorder line graph nodes
    train_node_set_line = []
    train_label_set_line = []
    test_node_set_line = []
    test_label_set_line = []
    for i in range(len(hyper_edge_list)):
        # region Majority of node label
        hyper_edge_label = []
        for hyper_edge_node in hyper_edge_list[i]:
            for j in range(len(train_labelset)):
                if (train_nodeset[j] == hyper_edge_node):
                    hyper_edge_label.append(train_labelset[j])
                    break;
        labelset=set()
        for label in hyper_node_label_list:
            labelset.add(label[1])
        y_train = [0]*len(labelset)
        if (len(hyper_edge_label) != 0):
            for pred in hyper_edge_label:
                y_train[pred] = y_train[pred] + 1
            y_train = [x / sum(y_train) for x in y_train]
            train_node_set_line.append(i)
            train_label_set_line.append(y_train)
        else:
            test_node_set_line.append(i)
            test_label_set_line.append(y_train)
        # endregion
    if (len(test_label_set_line) != 0):
        line_node_set = np.concatenate((train_node_set_line, test_node_set_line))
        line_label_set = np.vstack((train_label_set_line, test_label_set_line))
    else:
        line_node_set = train_node_set_line
        line_label_set = train_label_set_line
    # endregion


    # region line graph edgelist file generation
    '''
    This procedure will generate line graph edgelist file and edge weight list file.
    '''
    lineEdgeWeightSet=[]
    lineEdgeList=[]
    linenodelist= set()
    lineisolatelist=[]
    for i in range(len(hyper_edge_list)):
        j=i+1
        isisolated=1
        while(j<len(hyper_edge_list)):
            commonedges=set(hyper_edge_list[i]).intersection(set(hyper_edge_list[j]))
            totaldges = set(hyper_edge_list[i]).union(set(hyper_edge_list[j]))
            if(len(commonedges)!=0):
                isisolated=0
                lineEdgeWeightSet.append(len(commonedges)/len(totaldges))
                lineEdgeList.append([i,j])
                linenodelist.add(i)
                linenodelist.add(j)
            j=j+1
        if isisolated==1 and i not in linenodelist:
            lineisolatelist.append(i)
    # endregion
    return line_node_set, line_label_set, line_features, lineEdgeList, lineEdgeWeightSet, train_node_set_line.__len__()

def evaluate(features, support, labels, mask, placeholders):
    '''
    This procedure will evaluate accuracy for validation and test data set,
    :return: outs_val[0]:Loss   outs_val[1]:Accuracy    (time.time() - t_test):duration
    '''
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

def generatehyperembeddingmatrix(line_embedding,line_node_set):
    hyper_embedding_matrix=[]
    line_node_list = line_node_set
    for i in range(len(line_node_list)):
        k = np.where(line_node_list == i)
        # k = np.where(np.asarray(line_node_list) == i)
        # hyper_embedding_matrix.append(line_embedding[k[0][0]])
        hyper_embedding_matrix.append(line_embedding[k])
    return hyper_embedding_matrix

def createhypernodeembeddings(hyper_embeddings,hyper_edge_list,hyper_node_list):
    hyper_node_embeddings=[]
    no_of_nodes=len(hyper_node_list)
    embedding_dim=hyper_embeddings[0].shape[1]
    # embedding_dim = hyper_embeddings[0].shape[0]
    for node in hyper_node_list:
        node_embedding=[]
        for i in range(len(hyper_edge_list)):
            if node in hyper_edge_list[i]:
                node_embedding.append(hyper_embeddings[i]/len(hyper_edge_list[i]))
        # mean_embedding=np.mean(np.asarray(node_embedding), axis = 0)
        mean_embedding = np.sum(np.asarray(node_embedding), axis=0)
        # mean_embedding = np.max(np.asarray(node_embedding), axis=0)
        hyper_node_embeddings.append(mean_embedding)
    return np.asarray(hyper_node_embeddings).reshape(no_of_nodes,embedding_dim)

def tsneplot(hyper_node_embeddings,labels,fig_path):
    print("********************* tSNE Plot*********************")
    X = TSNE(n_components=2,perplexity=100,n_iter=5000).fit_transform(hyper_node_embeddings)
    colors = ['#FF0000', '#06D506', '#0931F7', '#00FFFF', '#FFE500', '#F700FF', '#9300FF', '#FFD700','#10DADE']
    for c in range(len(colors)):
        points = []
        for j in range(len(labels)):
            if (labels[j] == c):
                points.append(list(X[j]))
        x = []
        y = []
        for z in points:
            x.append(z[0])
            y.append(z[1])
        plt.plot(x, y, 'ro', c=colors[c], markersize=5.0, marker='.')
    plt.axis('off')
    plt.savefig(fig_path)
    plt.close()

def classify_logistic(hyper_node_embeddings,hyper_node_label_list,hyper_train,hyper_test):
    all_node_list=np.asarray(hyper_node_label_list)[:,0]
    train_label_list=hyper_train
    test_label_list=hyper_test
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    for i in range(len(train_label_list)):
        node=train_label_list[i][0]
        label=train_label_list[i][1]
        y_train.append(label)
        idx = np.where(all_node_list == node)
        X_train.append(hyper_node_embeddings[idx[0][0]])

    for i in range(len(test_label_list)):
        node=test_label_list[i][0]
        label=test_label_list[i][1]
        y_test.append(label)
        idx = np.where(all_node_list == node)
        X_test.append(hyper_node_embeddings[idx[0][0]])
    # X=np.vstack((X_train,X_test))
    # Y=np.concatenate((y_train,y_test))
    # for i in range(10):
        #     tsneplot(X,Y,"pubmed_"+str(i)+".png")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1_macro=f1_score(y_test, y_pred, average='macro')
    f1_micro=f1_score(y_test, y_pred, average='micro')
    f1_weighted=f1_score(y_test, y_pred, average='weighted')
    accuracy=accuracy_score(y_test, y_pred)
    print('\nAccuracy :: ', accuracy)
    print('\nMacro-F1 scores :: ', f1_macro)
    print('\nMicro-F1 scores :: ', f1_micro)
    print('\nWeighted-F1 scores :: ', f1_weighted)
    return accuracy

if __name__ == "__main__":
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    accuracy_list=[]
    split=[]
    time_list=[]
    dataset=['cora']        #'citeseer','pubmed'
    for data in dataset:
        # region Tf Flags
        a = datetime.datetime.now()
        # Settings
        if data != "pubmed":
            flags = tf.app.flags
            FLAGS = flags.FLAGS
            flags.DEFINE_string('dataset', data, 'Dataset string.')
            flags.DEFINE_string('model', 'gcn', 'Model string.')
            flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
            flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
            flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
            flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
            flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
            flags.DEFINE_float('weight_decay', 0,'Weight for L2 loss on embedding matrix.')
            flags.DEFINE_integer('early_stopping', 10,'Tolerance for early stopping (# of epochs).')
            flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
        else:
            flags = tf.app.flags
            FLAGS = flags.FLAGS
            flags.DEFINE_string('dataset', data,'Dataset string.')
            flags.DEFINE_string('model', 'gcn', 'Model string.')
            flags.DEFINE_float('learning_rate', 0.03, 'Initial learning rate.')
            flags.DEFINE_integer('epochs', 1700, 'Number of epochs to train.')
            flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
            flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
            flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
            flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
            flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
            flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
        # endregion

        print('dataset',FLAGS.dataset)
        print('learning_rate', FLAGS.learning_rate)
        print('epochs', FLAGS.epochs)
        print('hidden1', FLAGS.hidden1)
        print('hidden2', FLAGS.hidden2)

        # test_sizelist=[50,40,30,20,10,5]
        test_sizelist = [20]
        for i in range(len(test_sizelist)):       #len(test_sizelist)
            # print("****************** Iteration "+str(i)+" *************************")
            test_size=test_sizelist[i]
            split.append(100-test_size)
            edgelist_path="../data/"+str(FLAGS.dataset)+".edgelist"
            label_path="../data/"+str(FLAGS.dataset)+"_label.csv"
            feature_path="../data/"+str(FLAGS.dataset)+"_content.csv"
            # savepickel(edgelist_path,label_path,feature_path)
            with open("../pkl/hypergraph_" + str(FLAGS.dataset) + ".pickle", 'rb') as handle:
                hyper_edge_list = pickle.load(handle)
            with open("../pkl/hypernodelabel_" + str(FLAGS.dataset) + ".pickle", 'rb') as handle:
                hyper_node_label_list = pickle.load(handle)
            hyper_train, hyper_test=splitHyperNodes(hyper_node_label_list,test_size/100)  #test_size/100
            line_node_set, line_label_set, line_content_set, lineEdgeList, lineEdgeWeightSet, idx_train=\
                createLineGraph(hyper_edge_list,hyper_node_label_list,hyper_train,"../data/"+str(FLAGS.dataset)+"_content.csv")
            weighted_edgelist=[]
            for k in range(len(lineEdgeWeightSet)):
                wedge=lineEdgeList[k]
                wedge.append(lineEdgeWeightSet[k])
                weighted_edgelist.append(wedge)

            # region GCN
            adj, features, y_train, train_mask = load_data(line_node_set, line_label_set, line_content_set, weighted_edgelist,idx_train - 1)

            # region feature and Adj Matrix preprocessing step.
            # Some preprocessing
            features = preprocess_features(features)
            if FLAGS.model == 'gcn':
                support = [preprocess_adj(adj)]  # Normalise Adjacency Matrix
                num_supports = 1
                model_func = GCN  # Create GCN class object model_fun
            else:
                raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
            # endregion

            # Define placeholders
            placeholders = {
                'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
                'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
                'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
                'labels_mask': tf.placeholder(tf.int32),
                'dropout': tf.placeholder_with_default(0., shape=()),
                'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
            }

            # Create model
            model = model_func(placeholders, input_dim=features[2][1], logging=True)

            # Initialize session
            sess = tf.Session()
            # Init variables
            sess.run(tf.global_variables_initializer())

            cost_val = []
            embeddings = np.zeros((features[2][0], FLAGS.hidden1))
            epoclist=[]
            trainloss_list=[]
            # Train model
            for epoch in range(FLAGS.epochs):
                # if epoch % 100==0 and epoch !=0:
                #     FLAGS.learning_rate=FLAGS.learning_rate/2
                #     print(FLAGS.learning_rate)
                t = time.time()
                # region Construct Feed Dictionary
                # Construct feed dictionary.construct_feed_dict() in utils.py is called.
                feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
                # Update dropout value
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                # endregion

                # region Training step
                outs = sess.run([model.opt_op, model.loss, model.accuracy, model.activations[2]], feed_dict=feed_dict)
                embeddings = outs[3]
                # print(len(embeddings[0]))
                # endregion
                epoclist.append(epoch + 1)
                trainloss_list.append(outs[1])
                # region Print Traing and Validation results                                        "train_acc=", "{:.5f}".format(outs[2]),
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),"train_acc=", "{:.5f}".format(outs[2]),
                      "time=", "{:.5f}".format(time.time() - t))
            # endregion

            hyper_embedding=generatehyperembeddingmatrix(embeddings,line_node_set)
            hyper_node_embeddings=createhypernodeembeddings(hyper_embedding,hyper_edge_list,np.asarray(hyper_node_label_list)[:,0])
            accuracy=classify_logistic(hyper_node_embeddings,hyper_node_label_list,hyper_train,hyper_test)
            b = datetime.datetime.now()
            accuracy_list.append(accuracy*100)
        # region Clear Flags
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)
        # endregion
        b = datetime.datetime.now()
        time_list.append((b-a).seconds)
    print(time)
