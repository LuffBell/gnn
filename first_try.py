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

# Créer le modèle
if exp_config["gnn_model"] == "mlp":
    model = MLP(num_layers=exp_config["layers"], input_dim=exp_config["enc_dim"], hidden_dim=exp_config["hidden_size"],
                output_dim=2).to(exp_config["device"])
else:
    model = GNN_factory.createModel(name=exp_config["gnn_model"], config=exp_config).to(exp_config["device"])

# Charger les poids pré-entraînés
model.load_state_dict(torch.load('./saved_model/model.pth', map_location=exp_config["device"]))
model.eval()

# Charger un graphe à partir de votre dataset (exemple avec un fichier gpickle)
with open('./data/random_graph/random_graph_6.gpickle', 'rb') as f:
    graph = pickle.load(f)

# Convertir le graphe NetworkX en graphe DGL
dgl_graph = dgl.from_networkx(graph)

# Ajouter les encodages de position à votre graphe
dgl_graph.ndata["feat"] = positional_encoding(dgl_graph, pos_enc_dim=exp_config["enc_dim"])

def generate_positions(model, graph, device="cpu"):
    graph = graph.to(device)
    with torch.no_grad():
        node_positions = model(graph, graph.ndata["feat"]).cpu().numpy()
    return node_positions

# Générer les positions des nœuds
node_positions = generate_positions(model, dgl_graph, exp_config["device"])

def plot_graph(graph, positions):
    nx_graph = graph.to_networkx()
    pos_dict = {i: positions[i] for i in range(len(positions))}
    plt.figure(figsize=(8, 8))
    nx.draw(nx_graph, pos=pos_dict, with_labels=True, node_size=50, node_color='skyblue', edge_color='gray')
    plt.show()

# Visualiser le graphe
plot_graph(dgl_graph, node_positions)