from aeon.transformations.collection.convolution_based import MiniRocket, Rocket
from aeon.transformations.collection import PaddingTransformer

import numpy as np
import networkx as nx

from torch_geometric.data import Data, HeteroData
from torch_geometric.seed import seed_everything
from torch_geometric.nn import HANConv

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import json
from argparse import ArgumentParser
from tqdm import tqdm
import pickle

# train parameters
SEED = 2024
K = 10
K_NEIGHBORS = 5
EPOCHS = 700
HIDDEN = 128
LR = 0.001
WD = 0.001
DROPOUT = 0.6
HEADS = 8

class HeterogeneousGNN(torch.nn.Module):

  def __init__(self, data: HeteroData,
                     in_channels: int,
                     out_channels: int,
                     layers: int = 2,
                     heads: int = 4,
                     dropout: float = 0.2,
                     hidden_channels: int = 32):

    super().__init__()

    self.gnn1 = HANConv(
        in_channels, # dimensões das features
        out_channels=hidden_channels,
        heads=heads,
        metadata=data.metadata(),
        dropout=dropout
    )

    # self.gnn2 = HANConv(
    #     hidden_channels, # dimensões das features
    #     out_channels=hidden_channels,
    #     heads=heads,
    #     metadata=data.metadata(),
    #     dropout=dropout
    # )

    self.classifier = nn.Linear(len(data.metadata()[0])*hidden_channels, out_channels)

  def forward(self, X, edge_index):

    gnn_embds = self.gnn1(X, edge_index)
    # gnn_embds = self.gnn2(gnn_embds, edge_index)

    # agregando via concatenação as features enriquecidas de todos os nós
    agg_embs = torch.concat([embs for embs in gnn_embds.values()], dim=1)

    out = self.classifier(agg_embs)
    return out

def get_rocket_features(X: list) -> np.ndarray:

    univar_X_processed = []
    print(f"- Number of channels: {len(X[0])}")

    for channel in range(len(X[0])):
        print(f'- Processing channel {channel}.')
        X_curr = [np.expand_dims(example[channel], axis=0) for example in X]
        transformer = PaddingTransformer() # é necessário que todas as séries tenham o mesmo tamanho
        minirocket = MiniRocket(num_kernels=10_000, n_jobs=5, random_state=SEED)  # por padrao, MiniRocket usa ~10_000 kernels
        X_padded = transformer.fit_transform(X_curr)
        X_features = minirocket.fit_transform(X_padded)
        univar_X_processed.append(X_features)

    return univar_X_processed

def get_raw_features(X: list) -> np.ndarray:

    univar_X_processed = []
    for channel in range(len(X[0])):
        print(f'- Processing channel {channel}.')
        X_curr = [np.expand_dims(example[channel], axis=0) for example in X]
        transformer = PaddingTransformer() # é necessário que todas as séries tenham o mesmo tamanho
        X_padded = transformer.fit_transform(X_curr)
        univar_X_processed.append(X_padded.squeeze())

    return univar_X_processed

def create_data_from_adj_list(adj_list: np.array, features: np.array) -> Data:
    
    data = HeteroData()
    # data keys must be strings
    channels = [str(i) for i in range(len(features))]
    n_channels = len(channels)

    # adicionando features aos nós
    for i, feat in enumerate(channels):
        data[feat].x = torch.Tensor((features[i])).float()

    # adicionando arestas de cada grafo
    for label, adj_list in zip(channels, adj_list):

        adj_list = adj_list.toarray()
        sources = []
        targets = []

        for i in range(adj_list.shape[0]):
            neighbors = np.where(adj_list[i].astype(int) == 1)[0]
            sources.extend([i] * len(neighbors))
            targets.extend(neighbors.tolist())

        # # revertendo arestas - necessário se o método avaliado for direcionado
        # s_copy = sources.copy()
        # t_copy = targets.copy()

        # sources.extend(t_copy)
        # targets.extend(s_copy)

        edges = torch.LongTensor([sources, targets])
        data[label, f"{label}_neighbor", label].edge_index = edges
    

    # https://github.com/pyg-team/pytorch_geometric/issues/3604
    # conectando arestas de todos os tipos
    # a feature do tipo X do no BLAU tem que estar conectada com a feature do tipo Y do mesmo nó BLAU
    sources = [i for i in range(len(features[0]))]
    targets = sources.copy()

    edges = torch.LongTensor([sources, targets])

    # conectando todas as features do mesmo nó
    for feat_source in channels:
        for feat_dest in channels:
            # nao adicionar self-loop
            if feat_source == feat_dest: continue
            data[feat_source, "is_the_same", feat_dest].edge_index = edges

    return data

def train(model: nn.Module, 
          optimizer: torch.optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          data: Data, 
          X_train_ids: np.array, 
          y_train: np.array, 
          device: torch.device = torch.device('cpu')) -> float:

    model.train()
    optimizer.zero_grad()

    data = data.to(device)
    y_train = y_train.to(device)

    out = model(data.x_dict, data.edge_index_dict)
    loss = F.cross_entropy(out[X_train_ids], y_train)
    loss.backward()

    optimizer.step()
    scheduler.step()

    return float(loss)

@torch.no_grad()
def test(model: nn.Module, 
         data: Data, 
         X_test_ids: np.array, 
         y_test: np.array, 
         device: torch.device = torch.device('cpu')) -> tuple[float, dict, np.array]:

    model.eval()

    data = data.to(device)
    y_test = y_test.to(device)

    logits = model(data.x_dict, data.edge_index_dict)
    pred = logits[X_test_ids].softmax(dim=-1).argmax(dim=-1)
    loss = F.cross_entropy(logits[X_test_ids], y_test)

    y_test = y_test.cpu()
    pred = pred.cpu()

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')

    metrics = {'accuracy': acc,
               'macro_f1': f1}

    return float(loss), metrics, pred.numpy()

def evaluate_gnn_classifier(data: Data, 
                            y: np.array, 
                            k: int = 5, 
                            verbose: bool = False, 
                            device: torch.device = torch.device('cpu')) -> tuple[dict[str, list[int]], list[np.array]]:

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)

    folds_accuracy = []
    folds_f1 = []
    all_cms = []

    # X pode ser apenas um placeholder, estamos usando um cenário transdutivo
    X = np.zeros_like(y)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index] # n da para ter um np array com dimensoes diferentes :(
        y_train, y_test = y[train_index], y[test_index]

        # convertendo labels para tensores
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        # instanciando modelo
        # TODO: expor hiperparâmetros como argumentos
        model = HeterogeneousGNN(
            data,
            in_channels=data.x_dict['0'].shape[1], # dimensões das features), o canal '0' sempre existe 
            hidden_channels=HIDDEN,
            layers=1, # nao modifica nd
            dropout=DROPOUT, #0.5
            heads=HEADS,
            out_channels=n_labels
        )

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=1, epochs=EPOCHS, pct_start=0.1)

        # laço de treino
        for epoch in tqdm(range(1, EPOCHS + 1), desc=f'Fold {i + 1} epochs'):
            train(model, optimizer, scheduler, data, train_index, y_train, device)
            _, metrics, preds = test(model, data, test_index, y_test, device)
            # if verbose: print(f'- Epoch {epoch} metrics: {metrics}')

        # obtendo as métricas da ultima epoca
        acc = metrics['accuracy']
        f1 = metrics['macro_f1']
        cm = confusion_matrix(y_test, preds)

        folds_accuracy.append(acc)
        folds_f1.append(f1)
        all_cms.append(cm)

        if verbose:
            print(f"- Fold {i + 1} full report")
            print(metrics)
    
    return {'accuracy': folds_accuracy, 'macro_f1': folds_f1}, all_cms

if __name__ == '__main__':

    seed_everything(SEED)

    parser = ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to the input pickle file')
    parser.add_argument('--log', type=str, help='Path to the output log file')
    parser.add_argument('--raw', action='store_true', help='Whether use unprocessed data or not')

    args = parser.parse_args()

    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    # expecsts data['X'] to be a list of np arrays, (n_examples, channels, length)
    X = data['X']   
    y = data['y']


    n_labels = np.unique(y).shape[0]

    if args.raw:
        print('Using raw features')
        univar_X_processed = get_raw_features(X)
    else:
        print('Using MiniRocket')
        univar_X_processed = get_rocket_features(X)

    # univar_X_processed = each position of the list is a channel. if len(univar_X_processed) == 3, then we have 3 channels

    adj_lists = [kneighbors_graph(univar_X_processed[channel], n_neighbors=K_NEIGHBORS, n_jobs=-1, include_self=False) for channel in range(len(univar_X_processed))]
    print(f'- Using k={K_NEIGHBORS}')

    data = create_data_from_adj_list(adj_lists, univar_X_processed)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    print(f'Multivar')
    # data = create_data_from_distances(distances[i], univar_X_processed[i])
    metrics, all_cms = evaluate_gnn_classifier(data, y, k=K, verbose=True, device=device)
    print(f"\t- mean accuracy: {np.mean(metrics['accuracy']):.3f} +/- {np.std(metrics['accuracy']):.3f}")
    print(f"\t- mean f1: {np.mean(metrics['macro_f1']):.3f} +/- {np.std(metrics['macro_f1']):.3f}")

    with open(args.log, 'w') as f:
        json.dump(metrics, f, indent=4)





