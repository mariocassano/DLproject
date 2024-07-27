import ast
import os
import pandas as pd
import numpy as np
import igraph as ig
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
import itertools
from transformer import add_https_prefix, get_text_from_url, generate_embeddings, save_data_to_csv
from GAT import GAT
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

# seed globale
torch.manual_seed(1)

# imposta le opzioni per visualizzare più righe e colonne (20,20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

# percorso locale del dataset
file_path = 'dataset/polblogs.gml'

# lettura del grafo GML
graph = ig.Graph.Read_GML(file_path)

# salviamo in un dataframe Pandas tutti i nodi del grafo (1490 nodi)
nodes_dataframe = graph.get_vertex_dataframe()

# Rimozione di eventuali duplicati
nodes_dataframe.drop_duplicates(inplace=True)

# salviamo in una variabile tutti i valori dell'attributo target "value"
values = nodes_dataframe['value']

# trasformiamo ogni label di ogni nodo in un URL e lo salviamo in una variabile
urls_with_prefix = [add_https_prefix(label) for label in nodes_dataframe['label']]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # istanziamo il tokenizer di Bert base
model = BertModel.from_pretrained('bert-base-uncased') # istanza del modello Bert base che useremo per generare gli embeddings

# Se è presente il file csv che contiene tutti gli embeddings di tutti i nodi non effettuiamo il download
if not os.path.exists('data2.csv'):
    texts = [get_text_from_url(url) for url in urls_with_prefix] # salviamo tutti i testi per ogni nodo
    data = []
    for text,value in zip(texts, values):
        if text is not None and text != '':
            data.append((text,value))
        else:
            # anche questo testo verrà trasformato in embeddings, per ottenere dimensioni coerenti di "data"
            data.append(("No text available!", value))
    # manipolazioni dei dati necessarie per essere compatibili con il salvataggio in un file csv
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

# salviamo in un dataframe tutti gli archi del grafo (19025 archi)
edges_dataframe = graph.get_edge_dataframe()

# eliminazione di eventuali duplicati
edges_dataframe.drop_duplicates(inplace=True)

# elimina l'attributo "label"
nodes_dataframe = nodes_dataframe.drop(['label'], axis=1)

# Dividiamo l'attributo "source" se presenta due o più valori al suo interno
nodes_dataframe['source'] = nodes_dataframe['source'].apply(lambda x: x.split(','))

mlb = MultiLabelBinarizer()

# trasforma ogni possibile "source" rilevata (in tutto sono 7) come un attributo booleano multilabel
source_encoded = mlb.fit_transform(nodes_dataframe['source'])
source_encoded_df = pd.DataFrame(source_encoded, columns=[f'source_{cls}' for cls in mlb.classes_])

# aggiunta delle source "binarizzate" come feature, concatenando i dataframe ed elimino il vecchio attributo "source"
nodes_dataframe = pd.concat([nodes_dataframe, source_encoded_df], axis=1)
nodes_dataframe = nodes_dataframe.drop(["source"], axis=1)

# creo il dataframe degli embeddings e lo aggiungo come feature dei nodi, concatenando i due dataframe
embeddings_dataframe = pd.DataFrame(embeddings_array)
nodes_dataframe = pd.concat([nodes_dataframe, embeddings_dataframe, source_encoded_df], axis=1)

# split in X (feature) e y (target)
X = nodes_dataframe.drop(["value"], axis=1).values
y = nodes_dataframe["value"].values

# trasformazione del dataframe degli archi in una matrice di adiacenza compatibile con torch_geometric
edges_dataframe = pd.DataFrame.transpose(edges_dataframe)
edges_np = edges_dataframe.values
edges = torch.from_numpy(edges_np)

# trasformazione in tensori di X e y
X = torch.tensor(X)
y = torch.tensor(y, dtype=torch.long)

data = Data(x=X, edge_index=edges, y=y)

# Definizione del numero di nodi per il training
num_train_nodes = 900

# Genera una permutazione casuale degli indici dei nodi usando torch.randperm
permutation = torch.randperm(data.num_nodes) # vettore che contiene 1490 numeri interi (gli indici) da 1 a 1490

# prende 'num_train_nodes' nodi come nodi di training
train_indices = permutation[:num_train_nodes] # vettore che contiene i primi 900 indici (training)

# e i restanti nodi come nodi di test
test_indices = permutation[num_train_nodes:] # vettore che contiene gli altri 590 indici  (test)

# Divido i 900 indici di training in 80% training e 20% validation
train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=1, shuffle=True)

# Crea le maschere di training e test
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool) # vettore di 1490 booleani falsi
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

train_mask[train_indices] = True            # vettore di 1490 booleani dove solo i 700 (circa) di training sono True
test_mask[test_indices] = True              # vettore di 1490 booleani dove solo i 600 (circa) di test sono True
val_mask[val_indices] = True                # vettore di 1490 booleani dove solo i 200 (circa) di validazione sono True

data.train_mask = train_mask
data.test_mask = test_mask
data.val_mask = val_mask

# Parametri di validazione
param_grid = {
    'hidden_dim': [4, 8, 16],
    'lr': [0.01, 0.001],
    'head':[8, 12, 16],
    'optimizer':[torch.optim.Adadelta, torch.optim.Adam, torch.optim.RMSprop]
}

# per stampare il nome dell'optimizer nel grafico
def translate_optimizer(opt):
    if (type(opt) == torch.optim.Adam):
        return "Adam"
    elif (type(opt) == torch.optim.Adagrad):
        return "Adagrad"
    elif (type(opt) == torch.optim.RMSprop):
        return "RMSprop"
    elif (type(opt) == torch.optim.SGD):
        return "SGD"
    elif (type(opt) == torch.optim.Adadelta):
        return "Adadelta"

# addestra i nodi 'data' in base alla 'mask' inserita
def train(data, mask, model):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss

# effettua il test del modello e calcola l'accuracy (sia di validazione che di test)
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
validation_list = []
# definisce tutte le combinazioni di parametri possibili
for hidden_dim, lr, head, optimizer in itertools.product(param_grid['hidden_dim'], param_grid['lr'],
                                                         param_grid['head'], param_grid['optimizer']):
    gat = GAT(data.num_features, hidden_dim, head)
    optimizer = optimizer(gat.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    # addestra tutti i modelli
    for epoch in range(200):
        loss = train(data, train_mask, gat)
        if torch.isnan(loss):
            break
        losses.append(loss.detach().item())
    # illustra l'andamento della loss durante le epoche
    plt.title(f'Training modello con hidden_dim = {hidden_dim}, LR iniziale = {lr}, \n heads = {head} e optimizer = {translate_optimizer(optimizer)}')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.show()

    train_accuracy = compute_accuracy(gat, data.train_mask,'Training')
    val_accuracy = compute_accuracy(gat, data.val_mask, 'Validation')
    validation_list.append(val_accuracy)
    # seleziona il miglior modello in base alla val_accuracy più alta
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = {
            'hidden_dim': hidden_dim,
            'lr': lr,
            'head': head,
            'optimizer': type(optimizer)            # Restituisce la classe (Adam, Rmsprop, Gradient Descent, ...)
        }

print(f"Best validation accuracy: {best_accuracy*100:.4f} %")
print(f"Best parameters:", best_params)

# istanzia il modello con i parametri che hanno ottenuto i migliori risultati di accuracy
gat = GAT(data.num_features, best_params['hidden_dim'], best_params['head'])
optimizer = best_params['optimizer'](gat.parameters(), lr=best_params['lr'])
criterion = torch.nn.CrossEntropyLoss()

losses = []
# addestramento del miglior modello su training set e validation set contemporaneamente
for epoch in range(200):
    loss = train(data, torch.logical_or(train_mask, val_mask), gat)
    if torch.isnan(loss):
        break
    losses.append(loss.detach().item())

# illustra l'andamento della loss nell'addestramento del best model
plt.title(f'Training del best_model')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(losses)
plt.show()

# viene testato su dati mai visti prima (test set)
test_acc = compute_accuracy(gat, data.test_mask, 'Test')

# accuracy finale
print(f"Test accuracy: {test_acc * 100:.2f} %")