import math
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv, GATConv

class HConstructor(nn.Module):
    def __init__(self, num_edges, f_dim, iters = 1, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_edges = num_edges
        self.edges = None
        self.iters = iters
        self.eps = eps
        self.scale = f_dim ** -0.5
        # self.scale = 1

        self.edges_mu = nn.Parameter(torch.randn(1, f_dim))
        self.edges_logsigma = nn.Parameter(torch.zeros(1, f_dim))
        init.xavier_uniform_(self.edges_logsigma)

        self.to_q = nn.Linear(f_dim, f_dim)
        self.to_k = nn.Linear(f_dim, f_dim)
        self.to_v = nn.Linear(f_dim, f_dim)

        self.gru = nn.GRUCell(f_dim, f_dim)

        hidden_dim = max(f_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(f_dim + f_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, f_dim)
        )

        self.norm_input  = nn.LayerNorm(f_dim)
        self.norm_edgs  = nn.LayerNorm(f_dim)
        self.norm_pre_ff = nn.LayerNorm(f_dim)

    def mask_attn(self,attn,k):
        indices = torch.topk(attn,k).indices
        mask = torch.zeros(attn.shape).bool().to(attn.device)
        for i in range(attn.shape[0]):
            mask[i][indices[i]] = 1
        return attn.mul(mask)

    def ajust_edges(self,s_level,args):
        if args.stage != 'train':
            return

        if s_level > args.up_bound:
            self.num_edges = self.num_edges + 1
        elif s_level < args.low_bound:
            self.num_edges = self.num_edges - 1
            self.num_edges = max(self.num_edges,args.min_num_edges)
        else:
            return

    def forward(self, inputs, args):
        n, d, device = *inputs.shape, inputs.device
        n_s = self.num_edges
        
        if True:
        # if self.edges is None:
            mu = self.edges_mu.expand(n_s, -1)
            sigma = self.edges_logsigma.exp().expand(n_s, -1)
            edges = mu + sigma * torch.randn(mu.shape, device = device)
        else:
            edges = self.edges

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)
        k = F.relu(k)
        v = F.relu(v)

        for _ in range(self.iters):

            edges = self.norm_edgs(edges)

            #求结点相对于边的softmax
            q = self.to_q(edges)
            q = F.relu(q)

            dots = torch.einsum('ni,ij->nj', q, k.T) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)
            attn = self.mask_attn(attn,args.k_n)        #这个决定边的特征从哪些结点取

            #更新超边特征
            updates = torch.einsum('in,nf->if', attn, v)
            edges = torch.cat((edges,updates),dim=1)
            edges = self.mlp(edges)

            #按边相对于结点的softmax（更新边之后）
            q = self.to_q(inputs)
            k = self.to_k(edges)
            k = F.relu(k)
            v = F.relu(v)

            dots = torch.einsum('ni,ij->nj', q, k.T) * self.scale
            attn_v = dots.softmax(dim=1)
            attn_v = self.mask_attn(attn_v,args.k_e)    #这个决定一个结点属于多少条边
            H = attn_v
            
            #计算边的饱和度
            cc = H.ceil().abs()
            de = cc.sum(dim=0)
            empty = (de == 0).sum()
            s_level = 1 - empty/n_s

            self.ajust_edges(s_level,args)

            print("Num edges is: {}; Satuation level is: {}".format(self.num_edges,s_level))

        self.edges = edges

        return edges, H, dots

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, num_edges, bias=True):
        super(HGNN_conv, self).__init__()

        self.HConstructor = HConstructor(num_edges, in_ft)

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(in_ft, out_ft))
        self.mlp.append(nn.Linear(out_ft, out_ft))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, args):
        edges, H, H_raw = self.HConstructor(x, args)
        edges = edges.matmul(self.weight)
        if self.bias is not None:
            edges = edges + self.bias
        nodes = H.matmul(edges)
        # x = self.mlp[0](x) + self.mlp[1](nodes)
        x = x + nodes
        return x, H, H_raw

class HGNN_classifier(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(HGNN_classifier, self).__init__()
        in_dim = args.in_dim
        hid_dim = args.hid_dim 
        out_dim = args.out_dim
        num_edges = args.num_edges
        self.conv_number = args.conv_number

        self.dropout = dropout

        #self.linear_backbone = nn.Linear(in_dim,hid_dim)
      
        
        self.linear_backbone = nn.ModuleList()
        self.linear_backbone.append(nn.Linear(in_dim,hid_dim))
        self.linear_backbone.append(nn.Linear(hid_dim,hid_dim))
        self.linear_backbone.append(nn.Linear(hid_dim,hid_dim))



        self.gcn_backbone = nn.ModuleList()
        self.gcn_backbone.append(GCNConv(in_dim, hid_dim))
        self.gcn_backbone.append(GCNConv(hid_dim, hid_dim))
        

        self.convs = nn.ModuleList()
        self.transfers = nn.ModuleList()

        for i in range(self.conv_number):
            self.convs.append(HGNN_conv(hid_dim, hid_dim, num_edges))
            self.transfers.append(nn.Linear(hid_dim, hid_dim))

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_number * hid_dim, out_dim),
        )

    def forward(self, data, args):

        if args.backbone == 'linear':
            x = data['fts']
            #x = self.linear_backbone[0](x)
            x = F.relu(self.linear_backbone[0](x))
            x = F.relu(self.linear_backbone[1](x))
            x = self.linear_backbone[2](x)
        elif args.backbone == 'gcn':
            x = data['fts']
            edge_index = data['edge_index']
            x = self.gcn_backbone[0](x,edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.gcn_backbone[1](x,edge_index)

        tmp = []
        H = []
        H_raw = []
        for i in range(self.conv_number):
            x, h, h_raw = self.convs[i](x,args)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            if args.transfer == 1:
                x = self.transfers[i](x)
                x = F.relu(x)
            tmp.append(x)
            H.append(h)
            H_raw.append(h_raw)

        x = torch.cat(tmp,dim=1)

        out = self.classifier(x)
        return out, x, H, H_raw

class GCN(nn.Module):
    def __init__(self, args, layer_number=2):

        in_dim = args.in_dim
        hid_dim = args.hid_dim 
        out_dim = args.out_dim

        super(GCN, self).__init__()
        # graph convolution
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hid_dim))
        for i in range(1, layer_number):
            self.convs.append(GCNConv(hid_dim, hid_dim))

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, data, args):
        x = data['fts']
        edge_index = data['edge_index']

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        out = self.classifier(x)
        return out, x, None, None

class GAT(nn.Module):
    def __init__(self, args, layer_number=2):
        super(GAT, self).__init__()
        
        in_dim = args.in_dim
        hid_dim = args.hid_dim 
        out_dim = args.out_dim

        # graph convolution
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(in_dim, hid_dim))
        for i in range(1, layer_number):
            self.convs.append(GATConv(hid_dim, hid_dim))

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, data, args):
        x = data['fts']
        edge_index = data['edge_index']

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        out = self.classifier(x)
        return out, x, None, None

class MLP(nn.Module):
    def __init__(self, args, dropout=0.5, bias=True):

        in_dim = args.in_dim
        hid_dim = args.hid_dim 
        out_dim = args.out_dim

        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hid_dim, out_dim)
        )
  

    def forward(self,data,args):
        x = data['fts']           

        out = self.mlp(x)            

        return out, None, None, None
