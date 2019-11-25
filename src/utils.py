import numpy as np
import pickle as pkl
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# region Create Mask
def sample_mask(idx, l):
    """Create mask.
    This creates an array of size l with first idx terms = "True".
    """
    mask = np.zeros(l)
    for i in range(idx):
        mask[i] = 1
    return np.array(mask, dtype=np.bool)
# endregion

def encode_onehot(y):
    onehot_dim=int(max(y))+1
    onehot_label_matrix=np.zeros((len(y),onehot_dim))
    for i in range(len(y)):
        if(y[i]!=''):
            onehot_label_matrix[i][y[i]]=1
    return onehot_label_matrix

# region Load Dataset
def load_data(line_node_set,line_label_set, line_content_set, weighted_edgelist,idx_train ):
    # region Load Data Set
    '''
    We will load data for the dataset passed through parameter here.Used files(for cora dataset) :
    ind.cora.x  ind.cora.y  ind.cora.tx    ind.cora.ty    ind.cora.allx    ind.cora.ally    ind.cora.graph    ind.cora.test.index
    '''

    alllabellist = np.asarray(line_label_set)
    # for labels in alllabellist:
    #     print(labels)

    graph=nx.Graph()
    graph.add_weighted_edges_from(weighted_edgelist)
    # for edges in edgelist:
    #     graph.add_edge(edges[0],edges[1])
    x = line_content_set
    y = alllabellist


    features = sp.csc_matrix(x).tolil()                                # stack all features(train+test) together(Vertically Row wise).tolil() coverts the matrix to Linked list format.
    adj = nx.adjacency_matrix(graph)                                   # Generate Adjacency Matrix from Graph
    # labels = encode_onehot(y)                                                        #Stack All labels together (Vertically Row wise).2708 one hot vectors each of dimension 7.

    # region Create Mask
    '''
    Created Mask using sample_mask() function defined above.
    train_musk:Create an np.array of size |v|=2708(cora) and make idx_train entry True and others false.
    val_mask:Create an np.array of size |v|=2708(cora) and make idx_val entry True and others false.(Validation)
    test_mask:Create an np.array of size |v|=2708(cora) and make idx_test entry True and others false.
    '''
    train_mask = sample_mask(idx_train, y.shape[0])                    #Out of 2708 first 140 has true,rest all false.
    # test_mask = sample_mask(idx_train-1000, y.shape[0])                      #Out of 2708 last 1000 has true,rest all false.
    # endregion

    # region Create True Label
    '''
    Create True Label for Train,Validation and Test set.
    '''
    # y_train = np.zeros(labels.shape)                                          #numpy.zeros of size (2708,7)
    # y_test = np.zeros(labels.shape)                                         #numpy.zeros of size (2708,7)
    # y_train[train_mask, :] = labels[train_mask, :]                          #First 140(train_mask has true) will have one hot of true labels
    # y_test[test_mask, :] = labels[test_mask, :]                             #last 1000(test_mask has true) will have one hot of true labels
    # endregion
    y_train=alllabellist
    # y_test=alllabellist[1000:1510]
    return adj, features, y_train,train_mask
# endregion


# region Preprocess Features
def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation.
       Divide each row by total number of 1's in that row.
       Like First Row has 9 1's ,then make entry-0.1111111 in those 9 entries
       Second Row has 23 1's, then make entry-0.04347 in those 23 entries.
       So on..
    """
    # rowsum = np.array(features.sum(1))                      #Sum features for each Node.return np.ndarray with size (2708,1)[Sum of each 2708 nodes]
    # r_inv = np.power(rowsum, -1).flatten()                  #(1/sum) for each node and flatten it.ndarray with size (2708,)
    # r_inv[np.isinf(r_inv)] = 0.                             #Check for positive or negative infinity and make those entries as 0.
    # r_mat_inv = sp.diags(r_inv)                             #Diagonal matrix of shape (2708,2708) with (i,i) contains (1/sum) for ith node.
    # features = r_mat_inv.dot(features)                      #Dot product between Diagonal Matrix and features matrx.
    return sparse_to_tuple(features)                          #Return sparse matrix as tuple.
# endregion


# region Normalise Adjacency Matrix.
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix.
    Input to this procedure is : Adj_hat=Adj+Identity_Matrix
    It will return: Normalised_Adj=D^(-0.5).Adj_hat.D^(-0.5) [D-Degree Matrx]
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))                                                   #Dergees of each node.
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()                                   #degree^(-0.5) and flatten
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)                                           #D^(-0.5):diagonal matrix with (i,i) element contains degree(nodei)^(-0.5)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()          # Adj.D^(-0.5).Tr.D^(-0.5)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
       Adj_hat=Adj+Identity_Matrix
       Normalised_Adj=D^(-0.5).Adj_hat.D^(-0.5) [D-Degree Matrx]
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)
# endregion


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
