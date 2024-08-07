import torch
import os
from dgl.dataloading import GraphDataLoader
import networkx as nx
from viz_utils.deepdraw_dataloder import Graph_sequence_from_file_dgl, collate
import argparse
import matplotlib.pyplot as plt
import wandb
from dgl_gnn_models.models import GNN_factory, MLP
from viz_utils.deepdraw_dataloder import criterion_procrustes
from viz_utils.utils import collate_stress
from viz_utils.aesthetic_losses import Stress_loss, StressCorrected, CustomLoss
from viz_utils.utils import Random_from_file_dgl, Rome_from_file_dgl
import viz_utils.utils
from viz_utils.utils import collate_stress
from copy import deepcopy
from tqdm import tqdm
from time import sleep
import torch.nn as nn
import json
# GNN Explainer
# from torch_geometric.explain import Explainer, GNNExplainer
# BRUNO
# from torch_geometric.data import Data

from dgl.nn import AvgPooling, GNNExplainer
from dgl.nn import GraphConv, SubgraphX

###NATHAN
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, graph, features, feat=None, eweight=None):
        return self.model(graph, features)  # Appelle la méthode forward de votre modèle en passant uniquement les features
    


def read_json_and_get_info(json_file_path, target_filename):
    # Lire le fichier JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Récupérer les informations relatives au fichier cible
    file_info = data.get(target_filename, None)
    
    # Vérifier si les informations ont été trouvées
    if file_info is None:
        print(f"No information found for file: {target_filename}")
        return None
    
    # Structurer les données pour faciliter les calculs
    structured_info = {
        'new_center': [],
        'sub_graph': [],
        'feat_mask': [],
        'edge_mask': []
    }

    for key, value in file_info.items():
        # Convertir les valeurs en listes de nombres
        structured_info['new_center'].append([int(x) for x in value['new_center']])
        structured_info['sub_graph'].append(int(value['sub_graph']))
        structured_info['feat_mask'].append([float(x) for x in value['feat_mask']])
        structured_info['edge_mask'].append([float(x) for x in value['edge_mask']])

    return structured_info

json_file_path = './results/node_explanations.json'
###NATHAN
def forward_plot(g, model, retina_dims, epoch, show="disk", plot_dir="gnn_plots", name="first", ):
    out = model(g, g.ndata["feat"])
    dict_positions = {i: out[i].detach().cpu().numpy() for i in range(len(out))}
    retina, ax = plt.subplots(1, 1, figsize=(retina_dims / my_dpi, retina_dims / my_dpi),
                              dpi=my_dpi)
    graph = g.cpu().to_networkx()
    H = nx.Graph(graph)
    nx.draw(H, pos=dict_positions, with_labels=False, ax=ax)
    # limits = plt.axis('on')
    # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # plt.title(f"Evolution at Epoch {epoch}")

    if show == "screen":
        plt.show()
    elif show == "disk":
        plt.savefig(os.path.join(plot_dir, f'{name}_{epoch}_new.png'), bbox_inches='tight')
        # print(dict_positions)
        plt.close()
    elif show == "wandb":
        # Log plot object
        # wandb.log({f"plot_{name}": plt}, step=epoch)
        wandb.log({f"plot_{name}": wandb.Image(plt)}, step=epoch)
        plt.close()
    else:
        raise NotImplementedError


def plot_dataloader(dataset):
    # plotting dataloaders test
    dataloader = GraphDataLoader(dataset, batch_size=1, collate_fn=collate)
    it = iter(dataloader)
    first_graph_test, targets, filename = next(it)
    sec, targets, filename = next(it)
    th, targets, filename = next(it)
    f, targets, filename = next(it)
    g, targets, filename = next(it)
    return first_graph_test, sec, th, f, g


def plot_dataloader_stress(dataset):
    # plotting dataloaders test
    dataloader = GraphDataLoader(dataset, batch_size=1, collate_fn=collate_stress)
    it = iter(dataloader)
    first_graph_test, _, _, _ = next(it)
    second, _, _, _ = next(it)
    third, _, _, _ = next(it)
    f, _, _, _ = next(it)
    g, _, _, _ = next(it)
    return first_graph_test, second, third, f, g


if __name__ == '__main__':
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

    parser.add_argument('--dataset', type=str, default="random", choices=["random", "rome"],
                        help='dataset')
    parser.add_argument('--encoding', type=str, default="laplacian_eigenvectors",
                        choices=["one_hot", "laplacian_eigenvectors", "binary", "ones", "random", "original"],
                        help='node initial encoding')
    parser.add_argument('--enc_dims', type=int, default=8, metavar='N',
                        help='input encoding size for training (default: 64)')
    parser.add_argument('--loss_f', type=str, default="stress",
                        choices=["l1", "mse", "huber", "procrustes", "stress", "custom"],
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
    args.wandb = args.wandb in {'True', 'true'}
    args.save_model = args.save_model in {'True', 'true'}

    add_aesthete = args.add_aesthete

    device = "cuda"
    dataset_folder = os.path.join("data", "rome") if args.dataset == "rome" else os.path.join("data", "random_graph")
    viz_utils.utils.set_seed(args.seed)

    tr_set = "training_list"
    test_set = "test_list"
    vl_set = "validation_list"

    if args.target_type == "stress":
        collate_fn = collate_stress
        plot_dt = plot_dataloader_stress
    else:
        collate_fn = collate
        plot_dt = plot_dataloader

    enc_digits = args.enc_dims

    exp_config = {"dataset": dataset_folder,
                  "device": device,
                  "lr": args.lr,
                  "batch_size": args.batch_size,
                  "encoding": args.encoding,
                  "enc_dim": enc_digits,
                  "loss_f": args.loss_f,
                  "activation": args.activation,
                  "gnn_model": args.gnn_model,
                  "layers": args.layers,
                  "hidden_size": args.hidden_size,
                  "target_type": args.target_type,
                  "dropout": args.drop}

    show = "wandb" if args.wandb else "disk"
    # wandb init
    if args.wandb:
        WANDB_PROJ = "test"
        WANDB_ENTITY = "brunoufpe"
        wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, config=exp_config)

    plot_dir = "gnn_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    save_dir = "saved_model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    encoding = args.encoding  # "one_hot", "laplacian_eigenvectors", "binary", "ones", "random", "original"

    # dataset loading
    target_type = args.target_type  # , "circular", "spring", "spectral", "stress"
    if args.dataset == "rome":
        graph_dataset = Rome_from_file_dgl(dataset_folder=dataset_folder, set=tr_set, encoding=encoding,
                                           enc_digits=enc_digits,
                                           target_type=target_type)

        # validation

        graph_dataset_val = Rome_from_file_dgl(dataset_folder=dataset_folder, set=vl_set, encoding=encoding,
                                               enc_digits=enc_digits,
                                               target_type=target_type)

        graph_dataset_test = Rome_from_file_dgl(dataset_folder=dataset_folder, set=test_set, encoding=encoding,
                                                enc_digits=enc_digits,
                                                target_type=target_type)

    elif args.dataset == "random":
        graph_dataset = Random_from_file_dgl(dataset_folder=dataset_folder, set=tr_set, encoding=encoding,
                                             enc_digits=enc_digits,
                                             target_type=target_type)
        graph_dataset_val = Random_from_file_dgl(dataset_folder=dataset_folder, set=vl_set, encoding=encoding,
                                                 enc_digits=enc_digits,
                                                 target_type=target_type)
        graph_dataset_test = Random_from_file_dgl(dataset_folder=dataset_folder, set=test_set, encoding=encoding,
                                                  enc_digits=enc_digits,
                                                  target_type=target_type)
        

    dataloader = GraphDataLoader(graph_dataset, batch_size=1, collate_fn=collate_fn, num_workers=4)
    print("Training set loaded...")
    dataloader_val = GraphDataLoader(graph_dataset_val, batch_size=1, collate_fn=collate_fn, num_workers=4)
    print("Validation set loaded...")

    dataloader_test = GraphDataLoader(graph_dataset_test, batch_size=1, collate_fn=collate_fn, num_workers=4)
    print("Test set loaded...")
    # plot graph

    g_train_plot = plot_dt(graph_dataset)
    g_val_plot = plot_dt(graph_dataset_val)
    g_test_plot = plot_dt(graph_dataset_test)

    final_state_dim = 2  # directly regression

    # model definition 
    model_type = args.gnn_model

    if args.gnn_model == "mlp":
        model = MLP(num_layers=args.layers, input_dim=enc_digits, hidden_dim=args.hidden_size,
                    output_dim=final_state_dim,
                    drop_rate=args.drop).to(device)
    else:
        model = GNN_factory.createModel(name=args.gnn_model, config=exp_config).to(device)
    
    if add_aesthete != "false" and args.target_type == "stress":  # crossing edge only in stress mode
        import torch.nn as nn

        activation = nn.ReLU()
        hidden_sizes = [100, 300, 10]
        # path = "model_weights_m.pth"
        path = "model_weights_m_starting_node.pth"

        config = {
            "device": device,
            "trained_model": {
                "path": "trained_models",
                "name": path,
                "hidden_dims": hidden_sizes,
                "activation": activation
            }
        }
        neural_drawer = viz_utils.utils.CrossModel(config)

    # plot config
    retina_dims = 800
    my_dpi = 96

    # optimizer and loss    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.loss_f == "l1":
        loss_comp = torch.nn.L1Loss()
    elif args.loss_f == "mse":
        loss_comp = torch.nn.MSELoss()
    elif args.loss_f == "huber":
        loss_comp = torch.nn.SmoothL1Loss()
    elif args.loss_f == "procrustes":
        loss_comp = criterion_procrustes
    elif args.loss_f == "stress":
        loss_c = StressCorrected()
        loss_comp = loss_c.stress_loss
    elif args.loss_f == "custom":
        loss_c = CustomLoss(model)
        loss_comp = loss_c.forward
    else:
        raise NotImplementedError

    # training loop
    best_acc_val = None
    patience = 30
    patience_counter = 0

    best_model_state = deepcopy(model.state_dict())
    
    if args.save_model:
        torch.save(best_model_state, os.path.join(save_dir, "start_model.pth"))

    for epoch in range(100):
        with torch.no_grad():

            model.eval()
            # train graph
            
            # GNN Explainer
            # explainer = Explainer(
            #     model=model,
            #     algorithm=GNNExplainer(epochs=200),
            #     explanation_type='model',
            #     node_mask_type='attributes',
            #     edge_mask_type='object',
            #     model_config=dict(
            #         mode='multiclass_classification',
            #         task_level='node',
            #         return_type='log_probs',  # Model returns log probabilities.
            #     ),
            # )

            # node_idx = 0
            for i, el in enumerate(g_train_plot):
                # print(type(el))
                # Extrair features dos nós
                # node_features = el.to(device).ndata['feat']

                # Extrair índices das arestas
                # edge_indices = torch.stack(el.to(device).edges(), dim=0)
                

                # explanation = explainer(node_features, edge_indices, index=10)
                # print(explanation.edge_mask, explanation.node_mask)


                forward_plot(el.to(device), model, retina_dims, epoch, show=show, plot_dir=plot_dir,
                             name=f"{i}_train", )
            for i, el in enumerate(g_val_plot):
                forward_plot(el.to(device), model, retina_dims, epoch, show=show, plot_dir=plot_dir,
                             name=f"{i}_val", )
            for i, el in enumerate(g_test_plot):
                forward_plot(el.to(device), model, retina_dims, epoch, show=show, plot_dir=plot_dir,
                             name=f"{i}_test", )

            plt.close('all')

        model.train()

        counter = 0
        # used for print
        global_train_loss = 0.0
        # used for backprop
        batch_train_loss = torch.zeros(1).to(device)
        with tqdm(dataloader, unit="batch") as tepoch:
###NATHAN
            for all in tepoch:
                print(tepoch)
                tepoch.set_description(f"Training Epoch {epoch}")
                if args.target_type == "stress":
                    g, short_p, couple_idx, filename = all
                    # forward propagation by using all nodes
                    g, short_p, couple_idx = g.to(device), short_p.to(device), couple_idx.to(device)
                    targets = [short_p, couple_idx]
                else:
                    g, targets, filename = all
                    g, targets = g.to(device), targets.to(device)
                logits = model(g, g.ndata["feat"])

                # important_nods = read_json_and_get_info(json_file_path, filename[0])['feat_mask']
###NATHAN
                # compute loss
                if add_aesthete == "false":
                    loss = loss_comp(logits, targets, g)
                elif args.target_type == "stress" and add_aesthete == "cross":
                    loss = neural_drawer.cross_loss(logits, g.edges())
                elif args.target_type == "stress" and add_aesthete == "combined":
                    # print(loss_c.stress_loss_fn)
                    loss, silhouette_score, pairwise_distance_reduction, intra_cluster_distance, inter_cluster_distance = loss_comp(logits, targets, g, epoch, 'train')
                    loss = loss + args.weight_l * neural_drawer.cross_loss(logits, g.edges())

                batch_train_loss += loss
                global_train_loss += loss
                # hack to obtain batched loss computation
                counter += 1
                if (counter % args.batch_size == 0) or (counter == len(dataloader)):
                    opt.zero_grad()
                    batch_train_loss.backward()
                    opt.step()
                    batch_train_loss = torch.zeros(1).to(device)
                    if counter == len(dataloader):
                        counter = 0
                sleep(0.01)

        epoch_train_loss = global_train_loss / len(dataloader)
        print(f"Epoch {epoch} \t Global Training loss: \t  {epoch_train_loss}")

        if args.wandb:
            # where the magic happens
            wandb.log({"epoch": epoch, "train_loss": epoch_train_loss}, step=epoch)
            wandb.log({"epoch": epoch, "train_silhouette_score": silhouette_score}, step=epoch)
            wandb.log({"epoch": epoch, "train_pairwise_distance_reduction": pairwise_distance_reduction}, step=epoch)
            wandb.log({"epoch": epoch, "train_intra_cluster_distance": intra_cluster_distance}, step=epoch)
            wandb.log({"epoch": epoch, "train_inter_cluster_distance": inter_cluster_distance}, step=epoch)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            with tqdm(dataloader_val, unit="batch") as tepochval:
                for all_val in tepochval:
                    tepochval.set_description(f"Validation Epoch {epoch}")
                    if args.target_type == "stress":
                        g_val, short_p_val, couple_idx_val, filename = all_val
                        g_val, short_p_val, couple_idx_val = g_val.to(device), short_p_val.to(
                            device), couple_idx_val.to(
                            device)
                        targets_val = [short_p_val, couple_idx_val]
                    else:
                        g_val, targets_val, filename = all_val
                        g_val, targets_val = g_val.to(device), targets_val.to(device)

                    logits_val = model(g_val, g_val.ndata["feat"])

                    if add_aesthete == "false":
                        loss = loss_comp(logits_val, targets_val, g_val)
                    elif args.target_type == "stress" and add_aesthete == "cross":
                        loss = neural_drawer.cross_loss(logits_val, g_val.edges())
                    elif args.target_type == "stress" and add_aesthete == "combined":
                        loss, silhouette_score, pairwise_distance_reduction, intra_cluster_distance, inter_cluster_distance = loss_comp(logits_val, targets_val, g_val, epoch, 'val')
                        loss = loss + args.weight_l * neural_drawer.cross_loss(logits_val, g_val.edges())

                    val_loss += loss
                    sleep(0.01)

            epoch_valid_loss = val_loss / len(dataloader_val)
            print(f"Epoch {epoch} \t Global Validation Loss: \t  {epoch_valid_loss}")

            if best_acc_val is None:
                # fist time populating test loss
                best_acc_val = epoch_valid_loss

            if epoch_valid_loss < best_acc_val:
                best_acc_val = epoch_valid_loss
                patience_counter = 0
                if args.wandb:
                    wandb.log({"best_epoch": epoch, "best_valid_loss": best_acc_val}, step=epoch)
                if args.save_model:
                    best_model_state = deepcopy(model.state_dict())
                    torch.save(best_model_state, os.path.join(save_dir, f"model_{args.gnn_model}_new.pth"))

            else:
                patience_counter += 1

            if patience_counter == patience:
                exit()

            if args.wandb:
                # where the magic happens
                wandb.log({"epoch": epoch, "valid_loss": epoch_valid_loss}, step=epoch)
                wandb.log({"epoch": epoch, "valid_silhouette_score": silhouette_score}, step=epoch)
                wandb.log({"epoch": epoch, "valid_pairwise_distance_reduction": pairwise_distance_reduction}, step=epoch)
                wandb.log({"epoch": epoch, "valid_intra_cluster_distance": intra_cluster_distance}, step=epoch)
                wandb.log({"epoch": epoch, "valid_inter_cluster_distance": inter_cluster_distance}, step=epoch)
###NATHAN
            test_loss = 0.0
            model.eval()
            with tqdm(dataloader_test, unit="batch") as tepoch_test:
                for all_test in tepoch_test:
                    tepoch_test.set_description(f"Test Epoch {epoch}")
                    if args.target_type == "stress":
                        g_test, short_p_test, couple_idx_test, filename = all_test
                        g_test, short_p_test, couple_idx_test = g_test.to(device), short_p_test.to(
                            device), couple_idx_test.to(
                            device)
                        targets_test = [short_p_test, couple_idx_test]
                    else:
                        g_test, targets_test, filename = all_test
                        g_test, targets_test = g_test.to(device), targets_test.to(device)

                    # print(read_json_and_get_info(json_file_path, filename[0])['feat_mask'])

###NATHAN

                    logits_test = model(g_test, g_test.ndata["feat"])

                    if add_aesthete == "false":
                        loss = loss_comp(logits_test, targets_test, g_test)
                    elif args.target_type == "stress" and add_aesthete == "cross":
                        loss = neural_drawer.cross_loss(logits_test, g_test.edges())
                    elif args.target_type == "stress" and add_aesthete == "combined":
                        loss, silhouette_score, pairwise_distance_reduction, intra_cluster_distance, inter_cluster_distance = loss_comp(logits_test, targets_test, g_test, epoch, 'test')
                        loss = loss + args.weight_l * neural_drawer.cross_loss(
                            logits_test
                            , g_test.edges()
                        )

                    test_loss += loss

                    sleep(0.01)

            epoch_test_loss = test_loss / len(dataloader_test)

            print(f"Epoch {epoch} \t Global Test Loss: \t  {epoch_test_loss}")
            if args.wandb:
                # where the magic happens
                wandb.log({"epoch": epoch, "test_loss": epoch_test_loss}, step=epoch)
                wandb.log({"epoch": epoch, "test_silhouette_score": silhouette_score}, step=epoch)
                wandb.log({"epoch": epoch, "test_pairwise_distance_reduction": pairwise_distance_reduction}, step=epoch)
                wandb.log({"epoch": epoch, "test_intra_cluster_distance": intra_cluster_distance}, step=epoch)
                wandb.log({"epoch": epoch, "test_inter_cluster_distance": inter_cluster_distance}, step=epoch)

                if patience_counter == 0:
                    wandb.log({"epoch": epoch, "best_test_loss": epoch_test_loss}, step=epoch)

    # saving the model
    if args.save_model:
        torch.save(best_model_state, os.path.join(save_dir, f"model_{args.gnn_model}_new.pth"))