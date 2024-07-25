import ast
import os
import pandas as pd
import numpy as np
import igraph as ig
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
import itertools
import requests
from bs4 import BeautifulSoup
import re
from transformer import add_https_prefix, get_text_from_url, generate_embeddings, save_data_to_csv
from GAT import GAT
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

torch.manual_seed(1)

# Imposta le opzioni per visualizzare più righe e colonne
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

# Define the file path of the dataset
file_path = 'dataset/polblogs.gml'

# Read the gml file
graph = ig.Graph.Read_GML(file_path)

# Build a DataFrame representation of nodes
nodes_dataframe = graph.get_vertex_dataframe()
# Drop eventual duplicated rows in the nodes dataframe
nodes_dataframe.drop_duplicates(inplace=True)

values = nodes_dataframe['value']

urls_with_prefix = [add_https_prefix(label) for label in nodes_dataframe['label']]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

if not os.path.exists('data2.csv'):
    texts = [get_text_from_url(url) for url in urls_with_prefix]
    data = []
    for text,value in zip(texts, values):
        if text is not None and text != '':
            data.append((text,value))
        else:
            data.append(("No text available!", value))
    text_embeddings, values = generate_embeddings(data, tokenizer, model)
    text_embeddings_csv = np.squeeze(text_embeddings)
    text_embeddings_str = [text_embedding.tolist() for text_embedding in text_embeddings_csv]
    save_data_to_csv(text_embeddings_str, values, 'data2.csv')
    data = pd.read_csv('data2.csv')
    emb = data['embedding']
    values = data['value']
    emb = emb.apply(ast.literal_eval)
    embeddings_array = np.array(emb.tolist())
else:
    data = pd.read_csv('data2.csv')
    emb = data['embedding']
    values = data['value']
    emb = emb.apply(ast.literal_eval)
    embeddings_array = np.array(emb.tolist())

# Build a DataFrame representation of edges
edges_dataframe = graph.get_edge_dataframe()

# Drop eventual duplicated rows in the edges dataframe
edges_dataframe.drop_duplicates(inplace=True)

# drop the attribute "label" since it's not useful for our task
nodes_dataframe = nodes_dataframe.drop(['label'], axis=1)

# Process the 'source' attribute to split by comma
nodes_dataframe['source'] = nodes_dataframe['source'].apply(lambda x: x.split(','))

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Fit and transform the 'source' column
source_encoded = mlb.fit_transform(nodes_dataframe['source'])
source_encoded_df = pd.DataFrame(source_encoded, columns=[f'source_{cls}' for cls in mlb.classes_])

# concatenate the two dataframes
nodes_dataframe = pd.concat([nodes_dataframe, source_encoded_df], axis=1)
nodes_dataframe = nodes_dataframe.drop(["source"], axis=1)

# creo il dataframe degli embeddings
embeddings_dataframe = pd.DataFrame(embeddings_array)
nodes_dataframe = pd.concat([nodes_dataframe, embeddings_dataframe, source_encoded_df], axis=1)

# split into X and y
X = nodes_dataframe.drop(["value"], axis=1).values
y = nodes_dataframe["value"].values

edges_dataframe = pd.DataFrame.transpose(edges_dataframe)
edges_np = edges_dataframe.values
X = torch.tensor(X)
y = torch.tensor(y, dtype=torch.long)

edges = torch.from_numpy(edges_np)

data = Data(x=X, edge_index=edges, y=y)

# Definisci il numero di nodi per il training
num_train_nodes = 900

# Genera una permutazione casuale degli indici dei nodi usando torch.randperm
permutation = torch.randperm(data.num_nodes) # vettore che contiene 1490 numeri interi (gli indici) da 1 a 1490

# Prendi 'num_train_nodes' nodi come nodi di training
train_indices = permutation[:num_train_nodes] # vettore che contiene i primi 900 indici (training)

# I restanti nodi come nodi di test
test_indices = permutation[num_train_nodes:] # vettore che contiene gli altri 600 indici  (test)

# Divido i 900 indici di training in 70% training (600 indici) e 30% validation (300 indici)
train_indices, val_indices = train_test_split(train_indices, test_size=0.3, random_state=1, shuffle=True)

# Crea le maschere di training e test
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool) # vettore di 1490 booleani falsi
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

train_mask[train_indices] = True            # vettore di 1490 booleani dove solo i 700 di training sono True
test_mask[test_indices] = True              # vettore di 1490 booleani dove solo i 600 di training sono True
val_mask[val_indices] = True                # vettore di 1490 booleani dove solo i 200 di training sono True

data.train_mask = train_mask
data.test_mask = test_mask
data.val_mask = val_mask

# VALIDATION (TO SELECT THE BEST HYPERPARAMETERS)
# Grid Search parameters
param_grid = {
    'hidden_dim': [10],
    'lr': [0.01],
    'head':[16],
    'optimizer':[torch.optim.Adagrad, torch.optim.Adam, torch.optim.RMSprop]
}

def translate_class(cl):
    print(cl)
    return re.sub('<class \'torch\.optim\.', '', string=cl)

def train(data, mask, model):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss

def compute_accuracy(model, mask, context):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = criterion(out[mask], data.y[mask])
        print(f'{context} loss:  {loss:.4f}')
        pred = out.argmax(dim=1)
        accuracy = accuracy_score(y_true=y[mask], y_pred=pred[mask])
    return accuracy

best_accuracy = 0
best_params = {}

# Iterate over parameter grid
for hidden_dim, lr, head, optimizer in itertools.product(param_grid['hidden_dim'], param_grid['lr'],
                                                         param_grid['head'], param_grid['optimizer']):
    gat = GAT(data.num_features, hidden_dim, head)
    optimizer = optimizer(gat.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    # alleno tutti i modelli
    for epoch in range(150):
        loss = train(data, train_mask, gat)
        if torch.isnan(loss):
            break
        losses.append(loss.detach().item())
    # plt.title(f'Training modello con hidden_dim = {hidden_dim}, LR iniziale = {lr}, heads = {head} e optimizer = {translate_class(str(type(optimizer)))}')
    plt.title(f'Training modello con optimizer = {translate_class(str(type(optimizer)))}')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.show()

    train_accuracy = compute_accuracy(gat, data.train_mask,'Training')
    val_accuracy = compute_accuracy(gat, data.val_mask, 'Validation')

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = {
            'hidden_dim': hidden_dim,
            'lr': lr,
            'head': head,
            'optimizer': type(optimizer)            # Restituisce la classe (Adam, Rmsprop, Gradient Descent, ...)
        }

print(f"Best validation accuracy: {best_accuracy:.4f}")
print(f"Best parameters:", best_params)

# Evaluate on test set with best parameters
gat = GAT(data.num_features, best_params['hidden_dim'], best_params['head'])
optimizer = best_params['optimizer'](gat.parameters(), lr=best_params['lr'])
criterion = torch.nn.CrossEntropyLoss()

losses = []
# train the best model, merging training and validation set
for epoch in range(150):
    loss = train(data, torch.logical_or(train_mask, val_mask), gat)
    if torch.isnan(loss):
        break
    losses.append(loss.detach().item())

plt.title(f'Training del best_model')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(losses)
plt.show()

# evaluate on the test set
test_acc = compute_accuracy(gat, data.test_mask, 'Test')

print(f"Test accuracy: {test_acc*100:.4f} %")
