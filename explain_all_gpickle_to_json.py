import torch
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from dgl_gnn_models.models import GNN_factory, MLP
import argparse
from scipy import sparse as sp
import numpy as np
import os
import pickle
import torch.nn as nn
from dgl.nn import GNNExplainer  # Import GNNExplainer
import json
import os

# Configurer les arguments par défaut
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--drop', type=float, default=0., metavar='LR',
                    help='dropout (default: 0.0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dataset', type=str, default="rome", choices=["random", "rome"],
                    help='dataset')
parser.add_argument('--encoding', type=str, default="laplacian_eigenvectors",
                    choices=["one_hot", "laplacian_eigenvectors", "binary", "ones", "random", "original"],
                    help='node initial encoding')
parser.add_argument('--enc_dims', type=int, default=8, metavar='N',
                    help='input encoding size for training (default: 64)')
parser.add_argument('--loss_f', type=str, default="stress",
                    choices=["l1", "mse", "huber", "procrustes", "stress"],
                    help='loss function')
parser.add_argument('--layers', type=int, default=2,
                    help='number of model layers')
parser.add_argument('--activation', type=str, default="relu",
                    choices=["relu", "sigm", "tanh"],
                    help='activation function')
parser.add_argument('--gnn_model', type=str, default="gcn",
                    choices=["gcn", "gat", "gin", "mlp"],
                    help='model')
parser.add_argument('--hidden_size', type=int, default=100,
                    help='hidden size')
parser.add_argument('--target_type', type=str, default="stress",
                    choices=["circular", "spring", "spectral", "stress", "kamada"],
                    help='hidden size')
parser.add_argument('--wandb', type=str, default="false",
                    help='activate wandb')
parser.add_argument('--save_model', type=str, default="true",
                    help='save the best model')
parser.add_argument('--add_aesthete', type=str, default="false",
                    choices=["false", "cross", "combined"],
                    help='add aesthete for edge crossing')
parser.add_argument('--weight_l', type=float, default=1.,
                    help='weight between losses')

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, graph, features, feat=None, eweight=None):
        return self.model(graph, features)  # Appelle la méthode forward de votre modèle en passant uniquement les features

args = parser.parse_args()

# Configurations du modèle
exp_config = {
    "dataset": args.dataset,
    "device": "cuda",  # ou "cuda" si vous utilisez un GPU
    "lr": args.lr,
    "batch_size": args.batch_size,
    "encoding": args.encoding,
    "enc_dim": args.enc_dims,
    "loss_f": args.loss_f,
    "activation": args.activation,
    "gnn_model": args.gnn_model,
    "layers": args.layers,
    "hidden_size": args.hidden_size,
    "target_type": args.target_type,
    "dropout": args.drop
}

# Définir la fonction de positionnement
def positional_encoding(g, pos_enc_dim):
    n = g.number_of_nodes()
    A = g.adjacency_matrix().to_dense().numpy().astype(float)
    N = sp.diags(np.clip(g.in_degrees().numpy(), 1, None) ** -0.5, dtype=float)
    L = sp.eye(n) - N @ A @ N
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    return torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

# Créer le modèle de base
if exp_config["gnn_model"] == "mlp":
    base_model = MLP(num_layers=exp_config["layers"], input_dim=exp_config["enc_dim"], hidden_dim=exp_config["hidden_size"],
                     output_dim=2).to(exp_config["device"])
else:
    base_model = GNN_factory.createModel(name=exp_config["gnn_model"], config=exp_config).to(exp_config["device"])

# Charger les poids pré-entraînés
base_model.load_state_dict(torch.load('./saved_model/model.pth', map_location=exp_config["device"]))
base_model.eval()



# Créer le modèle avec le wrapper
model = ModelWrapper(base_model)


import os

graph_dir = './data/rome/'

# Fonction pour lire les fichiers .txt et extraire les noms des fichiers
def read_files_from_txt(directory, filenames):
    files = []
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                files.extend([line.strip() for line in f])
    return files

# Noms des fichiers .txt contenant les listes de fichiers
txt_files = ['training_list.txt', 'validation_list.txt', 'test_list.txt']

# Extraire les noms des fichiers des fichiers .txt
gpickle_files = read_files_from_txt(graph_dir, txt_files)

print(gpickle_files)







def generate_positions(model, graph, device="cuda"):
    graph = graph.to(device)
    model = model.to(device)
    with torch.no_grad():

        node_positions = model(graph.to(device), graph.ndata["feat"].to(device)).cpu().numpy()
        print(node_positions)
    return node_positions




def plot_graph(graph, positions):
    nx_graph = graph.to_networkx()
    pos_dict = {i: positions[i] for i in range(len(positions))}
    plt.figure(figsize=(8, 8))
    nx.draw(nx_graph, pos=pos_dict, with_labels=True, node_size=50, node_color='skyblue', edge_color='gray')
    plt.show()



# Ajouter GNNExplainer
def explain_node(model, graph, node_id, num_hops=3):
    explainer = GNNExplainer(model, num_hops=num_hops)
    
    new_center, sub_graph, feat_mask, edge_mask = explainer.explain_node(node_id=node_id,graph=graph, features=graph.ndata["feat"], feat =graph.ndata["feat"])

    return new_center, sub_graph, feat_mask, edge_mask

all_results = {}
number_gpickles_files = len(gpickle_files)
count_gplikles_files = 0
for gpickle_file in gpickle_files:
    count_gplikles_files = count_gplikles_files + 1
    
    with open(os.path.join(graph_dir, gpickle_file), 'rb') as f:
        graph = pickle.load(f)
    dgl_graph = dgl.from_networkx(graph)
    dgl_graph.ndata["feat"] = positional_encoding(dgl_graph, pos_enc_dim=exp_config["enc_dim"])

    print(f"Explaining nodes for graph: {gpickle_file}")
    graph_results = {}
    for node_id in range(dgl_graph.number_of_nodes()):
        print(count_gplikles_files, "/", number_gpickles_files)
        new_center, sub_graph, feat_mask, edge_mask = explain_node(model, dgl_graph, node_id, num_hops=3)
        node_result = {
            "new_center": new_center.tolist(),
            "sub_graph": sub_graph.number_of_nodes(),  # Simplify sub_graph representation
            "feat_mask": feat_mask.tolist(),
            "edge_mask": edge_mask.tolist()
        }
        graph_results[node_id] = node_result

        subgraph_node_positions = generate_positions(model=base_model, graph=sub_graph, device=exp_config["device"])
        #plot_graph(sub_graph, subgraph_node_positions)
    
    all_results[gpickle_file] = graph_results

# Write the results to a JSON file
results_file = './results/node_explanations.json'
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=4)