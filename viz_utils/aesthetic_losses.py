import torch
import torch.nn as nn
import numpy as np
from typing import List
import networkx as nx
from scipy import sparse as sp
from scipy.sparse.linalg import norm
import dgl
# from kmeans_pytorch import kmeans
# from kmeans_pytorch import kmeans_predict
from dgl.nn import AvgPooling, GNNExplainer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import pandas as pd

from dgl.nn import GraphConv, SubgraphX


def minimize_edge_variance(edge_parameters: torch.Tensor, E: np.array, norm: int = 2) -> torch.float:
    """
    Function which returns the variance of the length of the edges

    :param edge_parameters: list of vertices positions
    :param E: list of edges specified as tuple of connected nodes indices
    :param norm: norm to be used to calculate the distances (default Euclidean)
    """
    # edge ([x1, y1], [x2, y2])

    # edges = [edge_parameters[arc] for arc in E]
    #
    # edge_lengths = torch.stack([torch.dist(edge[0], edge[1], p=norm) for edge in edges])
    #
    # variance = torch.var(edge_lengths)

    # tensorial form - maybe faster
    pdist = nn.PairwiseDistance(p=2, eps=1e-16)
    tensor_edges_coordinates = edge_parameters[E[None, :]]
    edge_length = pdist(tensor_edges_coordinates[:, 0, :], tensor_edges_coordinates[:, 1, :])
    variance = torch.var(edge_length)

    return variance


def maximize_node_distances(edge_parameters: torch.Tensor) -> torch.float:
    """
    Function which return the sum of the inverse of the distances between graph nodes
    :param edge_parameters: list of vertices positions
    :return:
    """
    # distances = 0.
    # for i, edge1 in enumerate(edge_parameters):
    #     for j, edge2 in enumerate(edge_parameters):
    #         if j > i:
    #             # distances += 1 / torch.dist(edge1, edge2)
    #             distances += 1 / torch.dist(edge1, edge2)
    # return distances
    # tensorial form
    matrix_distances = torch.triu(
        torch.cdist(edge_parameters, edge_parameters))  # get upper triangle of the distance matrix
    sum_dist = torch.sum(1 / (matrix_distances[matrix_distances.nonzero(as_tuple=True)]))
    # sum_dist = 1/torch.sum(matrix_distances)
    return sum_dist


def shortest_path_computation(G):
    num_nodes = G.number_of_nodes()
    shortest_paths = dict(nx.shortest_path_length(G))

    shortest_p = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            shortest_p[i][j] = shortest_paths[i][j]

    range_graph = torch.arange(num_nodes)
    couples_indices = torch.combinations(range_graph, r=2)

    return shortest_p, couples_indices

class Stress_loss:
    def __init__(self, reduction="mean", normalize=True):
        self.loss_c = nn.MSELoss(reduction=reduction)
        self.normalize = normalize
        self.pdist = torch.nn.PairwiseDistance()

    def stress_loss(self, logits, targets, full_graph=True):
        shortest_paths, couples_indices = targets

        loss = 0.0

        if full_graph:
            sources = couples_indices[:, 0]
            dest = couples_indices[:, 1]
            coordinates_sources = logits[sources]
            coordinates_dest = logits[dest]

            targets = shortest_paths[sources, dest]
            if self.normalize:
                distances = self.pdist(coordinates_sources, coordinates_dest) * 1 / targets
            else:
                distances = self.pdist(coordinates_sources, coordinates_dest)
            loss = self.loss_c(distances, targets)

        return loss

# from openpyxl import load_workbook

# def save_metrics_to_excel(metrics, file_path, sheet_name='metricas'):
#     """
#     Salva métricas incrementalmente em um arquivo Excel.

#     Parameters:
#     - metrics: dicionário contendo as métricas a serem salvas.
#     - file_path: caminho para o arquivo Excel.
#     - sheet_name: nome da planilha (padrão é 'Sheet1').
#     """
#     try:
#         # Tentar carregar o arquivo Excel existente
#         book = load_workbook(file_path)
#         writer = pd.ExcelWriter(file_path, engine='openpyxl')
#         writer.book = book
#         writer.sheets = {ws.title: ws for ws in book.worksheets}
        
#         # Carregar a planilha existente
#         if sheet_name in writer.sheets:
#             df_existing = pd.read_excel(file_path, sheet_name=sheet_name)
#         else:
#             df_existing = pd.DataFrame()
        
#     except:
#         # Se o arquivo não existir, criar um novo DataFrame
#         writer = pd.ExcelWriter(file_path, engine='openpyxl')
#         df_existing = pd.DataFrame()
    
#     # Criar um DataFrame com as novas métricas
#     df_new = pd.DataFrame([metrics])
    
#     # Concatenar os DataFrames existente e novo
#     df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
#     # Salvar o DataFrame combinado de volta ao arquivo Excel
#     df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
#     writer.close()


class CustomLoss(nn.Module):
    def __init__(self, base_loss_fn, lambda_explainer=1.0, lambda_stress=1.0, num_clusters=3, cluster_margin=1.0):
        super(CustomLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.lambda_explainer = lambda_explainer
        self.lambda_stress = lambda_stress
        self.num_clusters = num_clusters
        self.cluster_margin = cluster_margin
        self.pdist = nn.PairwiseDistance(p=2, eps=1e-16)
        self.stress_loss_fn = StressCorrected().stress_loss
        self.explainer = GNNExplainer(self.base_loss_fn, num_hops=3)
        self.kmeans = KMeans(n_clusters=self.num_clusters)
    

    def get_node_importance_scores(self, g):
        torch.set_grad_enabled(True)
        node_feat_mask, edge_mask = self.explainer.explain_graph(g, g.ndata['feat'])
        node_importance_scores = node_feat_mask
        return node_importance_scores

    def average_pairwise_distance(self, pairwise_distances):
        return pairwise_distances.mean()

    def calculate_pairwise_distances(self, important_node_feats):
        return torch.cdist(important_node_feats, important_node_feats, p=2)

    def intra_cluster_distance(self, important_node_feats, labels):
        unique_labels = set(labels)
        intra_distances = []
        for label in unique_labels:
            cluster_feats = important_node_feats[labels == label]
            if len(cluster_feats) > 1:
                intra_distances.append(self.average_pairwise_distance(self.calculate_pairwise_distances(cluster_feats)))
        return torch.tensor(intra_distances).mean()

    def inter_cluster_distance(self, cluster_centroids):
        pairwise_distances = torch.cdist(cluster_centroids, cluster_centroids, p=2)
        num_clusters = len(cluster_centroids)
        return pairwise_distances.sum() / (num_clusters * (num_clusters - 1))

    def forward(self, logits, targets, g, epoch, data):
        # Stress loss
        stress_loss = self.stress_loss_fn(logits, targets)

        # Extract node importance scores
        node_importance_scores = self.get_node_importance_scores(g)

        # Select important nodes (e.g., top 50% most important nodes)
        threshold_value = torch.quantile(node_importance_scores, 0.5)
        important_nodes = torch.nonzero(node_importance_scores >= threshold_value).squeeze()
        important_node_feats = logits[important_nodes]

        if len(important_nodes) < self.num_clusters:
            # If there are not enough important nodes to form the clusters, avoid clustering
            return self.lambda_stress * stress_loss, None, None, None, None

        # Get embeddings of important nodes
        important_embeddings = logits[important_nodes]

        # Perform k-means clustering 
        self.kmeans.fit(
            important_embeddings.detach().cpu().numpy(),
        )
        cluster_centroids = torch.tensor(self.kmeans.cluster_centers_).to(logits.device)

        # cluster_centroids = torch.tensor(self.kmeans.cluster_centers_).to(logits.device)
        labels = self.kmeans.predict(
                important_embeddings.detach().cpu().numpy()
            )
        
        # Silhouette Score
        silhouette_avg = silhouette_score(important_node_feats.detach().cpu().numpy(), labels)
        
        # Pairwise Distance Reduction
        pairwise_distances_after = self.calculate_pairwise_distances(important_node_feats)
        avg_distance_after = self.average_pairwise_distance(pairwise_distances_after)
        
        # Intra-cluster Distance
        intra_distance = self.intra_cluster_distance(important_node_feats, labels)
        
        # Inter-cluster Distance
        inter_distance = self.inter_cluster_distance(cluster_centroids)

        # Calculate cluster separation loss
        cluster_separation_loss = 0.0
        for i in range(self.num_clusters):
            for j in range(i + 1, self.num_clusters):
                dist = self.pdist(cluster_centroids[i].unsqueeze(0), cluster_centroids[j].unsqueeze(0))
                cluster_separation_loss += torch.clamp(self.cluster_margin - dist, min=0).mean()

        cluster_separation_loss /= self.num_clusters

        # Total loss
        total_loss = self.lambda_stress * stress_loss + self.lambda_explainer * cluster_separation_loss
        return total_loss, silhouette_avg, avg_distance_after.item(), intra_distance.item(), inter_distance.item()

class CustomLoss2(nn.Module):
    def __init__(self, model, lambda_explainer=1.0, lambda_stress=1.0, margin=1.0):
        super(CustomLoss, self).__init__()
        self.model = model
        # self.base_loss_fn = base_loss_fn
        self.lambda_explainer = lambda_explainer
        self.lambda_stress = lambda_stress
        self.margin = margin
        self.pdist = nn.PairwiseDistance(p=2, eps=1e-16)
        self.stress_loss_fn = StressCorrected().stress_loss
        self.explainer = GNNExplainer(model, num_hops=3)

    def get_node_importance_scores(self, g):
        node_feat_mask, _ = self.explainer.explain_graph(g, g.ndata['feat'])
        return node_feat_mask

    def aggregate_feature_importance(self, node_feat_mask):
        # Sum the importance scores of all features for each node
        node_importance_scores = node_feat_mask.sum(dim=-1)
        return node_importance_scores

    def select_important_nodes(self, node_importance_scores, threshold=0.9):
        # Select nodes with importance scores above the threshold (e.g., top 10% most important nodes)
        threshold_value = torch.quantile(node_importance_scores, threshold)
        important_nodes = torch.nonzero(node_importance_scores >= threshold_value).squeeze()
        return important_nodes

    def forward(self, logits, targets, g):
        # Perte de stress
        stress_loss = self.stress_loss_fn(logits, targets)

        # Get node importance scores using GNNExplainer
        node_feat_mask = self.get_node_importance_scores(g)
        node_importance_scores = self.aggregate_feature_importance(node_feat_mask)
        important_nodes = self.select_important_nodes(node_importance_scores, threshold=0.9)

        # Perte de margin-based explainer
        explainer_loss = 0.0
        num_important_nodes = len(important_nodes)
        if num_important_nodes > 1:
            for i in range(num_important_nodes):
                for j in range(i + 1, num_important_nodes):
                    node_i = important_nodes[i]
                    node_j = important_nodes[j]
                    dist = self.pdist(logits[node_i].unsqueeze(0), logits[node_j].unsqueeze(0))
                    explainer_loss += torch.clamp(self.margin - dist, min=0).mean()

        explainer_loss /= num_important_nodes

        # Perte totale
        total_loss = self.lambda_stress * stress_loss + self.lambda_explainer * explainer_loss
        return total_loss

class CustomLoss2(nn.Module):
    def __init__(self, lambda_explainer=1.0, lambda_stress=1.0):
        super(CustomLoss, self).__init__()
        # self.base_loss_fn = base_loss_fn
        self.lambda_explainer = lambda_explainer
        self.lambda_stress = lambda_stress
        self.pdist = nn.PairwiseDistance(p=2, eps=1e-16)
        self.stress_loss_fn = StressCorrected().stress_loss

    def forward(self, logits, targets, g, important_nods):
        # Perte de stress
        stress_loss = self.stress_loss_fn(logits, targets)  

        # print(g)

        # Perte d'explainer
        node_masks = g.ndata['feat']
        explainer_loss = 0.0
        for node_idx in range(g.number_of_nodes()):
            node_mask = node_masks[node_idx]
            # important_nodes = torch.nonzero(node_mask > 0.5).squeeze()
            important_nodes = important_nods

            if len(important_nodes) > 1:
                for i in range(len(important_nodes)):
                    for j in range(i + 1, len(important_nodes)):
                        node_i = important_nodes[i]
                        node_j = important_nodes[j]
                        dist = self.pdist(logits[node_i].unsqueeze(0), logits[node_j].unsqueeze(0))
                        explainer_loss += dist

        explainer_loss /= g.number_of_nodes()

        # Perte totale
        total_loss = self.lambda_stress * stress_loss + self.lambda_explainer * explainer_loss
        # print(total_loss)
        return total_loss.mean()
    

class StressCorrected:
    def __init__(self, ):
        self.loss_c = nn.MSELoss()
        self.pdist = torch.nn.PairwiseDistance()

    def stress_loss(self, logits, targets, full_graph=True):
        shortest_paths, couples_indices = targets

        loss = 0.0

        if full_graph:
            sources = couples_indices[:, 0]
            dest = couples_indices[:, 1]
            coordinates_sources = logits[sources]
            coordinates_dest = logits[dest]

            delta = shortest_paths[sources, dest]

            distance = self.pdist(coordinates_sources, coordinates_dest)
            weight = 1 / (delta + 1e-7)
            loss = weight * self.loss_c(distance, delta)

        return loss.mean()


def loss_lspe(g, p, pos_enc_dim=None, lambda_loss=1.):
    # Loss B: Laplacian Eigenvector Loss --------------------------------------------

    n = g.number_of_nodes()

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt="csr")
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n) - N * A * N

    pT = torch.transpose(p, 1, 0)
    loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(g.device)), p))

    # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
    # bg = dgl.unbatch(g)
    # batch_size = len(bg)
    # P = sp.block_diag([bg[i].ndata['p'].detach().cpu() for i in range(batch_size)])
    P = sp.block_diag([p.detach().cpu()])
    PTP_In = P.T * P - sp.eye(P.shape[1])
    loss_b_2 = torch.tensor(norm(PTP_In, 'fro') ** 2).float().to(g.device)

    # loss_b = (loss_b_1 + lambda_loss * loss_b_2) / (pos_enc_dim * batch_size * n)
    loss_b = (loss_b_1 + lambda_loss * loss_b_2) / (pos_enc_dim * n)

    return loss_b
