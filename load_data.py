import os.path as osp
import torch
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull
import scipy.io as scio
import numpy as np
import torchvision
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pickle
import itertools
import random

def load_data2(args):
    """
    parses the dataset
    """
    dataset = args.dataset
    splits = args.splits
    device = torch.device(args.device)
    path = osp.abspath(__file__)         #当前文件绝对路径
    d_path = osp.dirname(path)           #当前文件所在目录
    # f_path = osp.dirname(d_path)         #当前文件所在目录的父目录
    f_path = osp.join(d_path, ('data2'))
    
    d_path_dict = {
        'ca_cora':osp.join(osp.join(f_path, ('coauthorship')),'cora'),
        'ca_dblp':osp.join(osp.join(f_path, ('coauthorship')),'dblp'),
        'cc_cora':osp.join(osp.join(f_path, ('cocitation')),'cora'),
        'cc_citeseer':osp.join(osp.join(f_path, ('cocitation')),'citeseer'),
        'ca_pubmed':osp.join(osp.join(f_path, ('cocitation')),'pubmed')
    }

    pickle_file = osp.join(d_path_dict[dataset], "splits", str(splits) + ".pickle")

    with open(osp.join(d_path_dict[dataset], 'features.pickle'), 'rb') as handle:
        features = pickle.load(handle).todense()

    with open(osp.join(d_path_dict[dataset], 'labels.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    with open(pickle_file, 'rb') as H: 
        Splits = pickle.load(H)
        train, test = Splits['train'], Splits['test']

    with open(osp.join(d_path_dict[dataset], 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

    tmp_edge_index = []
    for key in hypergraph.keys():
        ms = hypergraph[key]
        tmp_edge_index.extend(list(itertools.permutations(ms,2)))
    
    edge_s = [ x[0] for x in tmp_edge_index]
    edge_e = [ x[1] for x in tmp_edge_index]

    edge_index = torch.LongTensor([edge_s,edge_e])

    features = torch.Tensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)

    data = {
        'fts':features,
        'edge_index':edge_index,
        'lbls':labels,
        'train_idx':train,
        'test_idx':test
    }

    return data

def load_cite(args):
    dname = args.dataset
    device = torch.device(args.device)
    path = osp.abspath(__file__)         #当前文件绝对路径
    d_path = osp.dirname(path)           #当前文件所在目录
    # f_path = osp.dirname(d_path)         #当前文件所在目录的父目录
    f_path = osp.join(d_path, ('data'))

    dataset = Planetoid(f_path,dname)      #dataset

    tmp = dataset[0].to(device)
    fts = tmp.x
    lbls = tmp.y

    if args.split_ratio < 0:
        train_idx = tmp.train_mask
        test_idx = tmp.test_mask
    else:
        nums = lbls.shape[0]
        num_train = int(nums * args.split_ratio)
        idx_list = [i for i in range(nums)]

        train_idx = random.sample(idx_list, num_train)
        test_idx = [i for i in idx_list if i not in train_idx]

        train_idx = torch.tensor(train_idx)
        test_idx = torch.tensor(test_idx)

    data = {
        'fts':fts,
        'edge_index':tmp.edge_index,
        'lbls':lbls,
        'train_idx':train_idx,
        'test_idx':test_idx
    }

    return data

def load_ft(args):
    if args.dataset == '40':
        data_dir = './data/ModelNet40_mvcnn_gvcnn.mat'
    elif args.dataset == 'NTU':
        data_dir = './data/NTU2012_mvcnn_gvcnn.mat'

    device = torch.device(args.device)
    feature_name = args.fts

    data = scio.loadmat(data_dir)
    lbls = data['Y'].astype(np.long)
    if lbls.min() == 1:
        lbls = lbls - 1
    idx = data['indices'].item()

    if feature_name == 'MVCNN':
        fts = data['X'][0].item().astype(np.float32)
        fts = torch.Tensor(fts).to(device)
    elif feature_name == 'GVCNN':
        fts = data['X'][1].item().astype(np.float32)
        fts = torch.Tensor(fts).to(device)
    else:
        fts1 = data['X'][0].item().astype(np.float32)
        fts2 = data['X'][1].item().astype(np.float32)
        fts1 = torch.Tensor(fts1).to(device)
        fts2 = torch.Tensor(fts2).to(device)

        fts = torch.cat((fts1,fts2),dim=-1)

    if args.split_ratio < 0:
        train_idx = np.where(idx == 1)[0]
        test_idx = np.where(idx == 0)[0]
    else:
        nums = lbls.shape[0]
        num_train = int(nums * args.split_ratio)
        idx_list = [i for i in range(nums)]

        train_idx = random.sample(idx_list, num_train)
        test_idx = [i for i in idx_list if i not in train_idx]

    # train_idx = np.where(idx == 1)[0]
    # test_idx = np.where(idx == 0)[0]

    lbls = torch.Tensor(lbls).squeeze().long().to(device)
    train_idx = torch.Tensor(train_idx).long().to(device)
    test_idx = torch.Tensor(test_idx).long().to(device)

    data = {
        'fts':fts,
        'lbls':lbls,
        'train_idx':train_idx,
        'test_idx':test_idx
    }

    return data

def load_data(args):
    if args.dataset in ['40','NTU']:
        return load_ft(args)
    elif args.dataset in ['Cora','Citeseer','PubMed']:
        return load_cite(args)
    elif args.dataset in ['MINIST']:
        return load_minist(args)
    elif args.dataset in ['cora']:
        return load_citation_data()

def load_minist(args):
    device = torch.device(args.device)
    dataset = torchvision.datasets.MNIST(root='./data',transform=lambda x:list(x.getdata()),download=True)
    features = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]
    features = torch.Tensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)

    train_idx = [i for i in range(50000)]
    test_idx = [i for i in range(50000,60000)]

    data = {
        'fts':features,
        'lbls':labels,
        'train_idx':train_idx,
        'test_idx':test_idx
    }

    return data


def parse_index_file(filename):
    """
    Copied from gcn
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def load_citation_data():
    """
    Copied from gcn
    citeseer/cora/pubmed with gcn split
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    cfg = {
        'citation_root':'./data/gcn',
        'activate_dataset':'cora',
        'add_self_loop': True
    }


    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(cfg['citation_root'], cfg['activate_dataset'], names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(cfg['citation_root'], cfg['activate_dataset']))
    test_idx_range = np.sort(test_idx_reorder)

    if cfg['activate_dataset'] == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    features = features.todense()

    G = nx.from_dict_of_lists(graph)
    # print("=====> ", G)
    # edge_list = G.adjacency_list()
    adjacency = G.adjacency()
    edge_list = []
    for item in adjacency:
        # print(list(item[1].keys()))
        edge_list.append(list(item[1].keys()))

    degree = [0] * len(edge_list)
    if cfg['add_self_loop']:
        for i in range(len(edge_list)):
            edge_list[i].append(i)
            degree[i] = len(edge_list[i])
    max_deg = max(degree)
    mean_deg = sum(degree) / len(degree)
    print(f'max degree: {max_deg}, mean degree:{mean_deg}')

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]     # one-hot labels
    n_sample = labels.shape[0]
    n_category = labels.shape[1]
    lbls = np.zeros((n_sample,))
    if cfg['activate_dataset'] == 'citeseer':
        n_category += 1                                         # one-hot labels all zero: new category
        for i in range(n_sample):
            try:
                lbls[i] = np.where(labels[i]==1)[0]                     # numerical labels
            except ValueError:                              # labels[i] all zeros
                lbls[i] = n_category + 1                        # new category
    else:
        for i in range(n_sample):
            lbls[i] = np.where(labels[i]==1)[0]                     # numerical labels

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))
    

    features = torch.Tensor(features)
    lbls = torch.LongTensor(lbls)

    data = {
        'fts':features,
        'lbls':lbls,
        'train_idx':idx_val,
        'test_idx':idx_test
    }

    return data

    # return features, lbls, idx_train, idx_val, idx_test, n_category, edge_list, edge_list