from utils import setup_seed, arg_parse, visualization
from load_data import load_data
from train import train_dhl,train_gcn,train_mlp,train_gat
import json
import torch
from networks import HGNN_classifier, GCN, GAT, MLP
import torch.nn.functional as F
import time

chosse_trainer = {
    'dhl':train_dhl,
    'gcn':train_gcn,
    'MLP':train_mlp,
    'gat':train_gat
}

args = arg_parse()

setup_seed(args.seed)
data = load_data(args)

fts = data['fts']
lbls = data['lbls']

args.in_dim = fts.shape[1]
args.out_dim = lbls.max().item() + 1
args.min_num_edges = args.k_e

args_list = []

best_acc = chosse_trainer[args.model](data, args)

args.best_acc = best_acc
args_list.append(args.__dict__)

############################################## visualization
chosse_model = {
    'dhl':HGNN_classifier,
    'gcn':GCN,
    'MLP':MLP,
    'gat':GAT
}

model = chosse_model[args.model](args)
state_dict = torch.load('model.pth',map_location=args.device)
model.load_state_dict(state_dict)
model.to(args.device)


model.eval()
mask = data['test_idx']
labels = data['lbls'][mask]

out, x, H, H_raw = model(data,args)
pred = F.log_softmax(out, dim=1)

_, pred = pred[mask].max(dim=1)
correct = int(pred.eq(labels).sum().item())
acc = correct / len(labels)

print("Acc ===============> ", acc)

visualization(model, data, args, title=None)
# with open('commandline_args{}.txt'.format(args.cuda), 'w') as f:
#     json.dump([args.__dict__,args.__dict__], f, indent=2)