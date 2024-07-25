import ast
import itertools
import random
import numpy as np
import pandas as pd
import igraph as ig
import torch
import requests
from bs4 import BeautifulSoup
import re
from pandas import read_csv
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def add_https_prefix(url):
    if not url.startswith("http://") and not url.startswith("https://"):
        return "https://" + url
    return url

def get_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Estrarre tutto il testo
        paragraphs = soup.find_all('div')
        if paragraphs != " " or paragraphs != None:
            text = ' '.join([para.get_text() for para in paragraphs])
        return re.sub(r'\n\s*\n', '\n', text)
    except requests.exceptions.RequestException as e:
        print(f"Errore nel download del contenuto da {url}: {e}")
        return None

def save_data_to_csv(text_embeddings, values, file_path):
    df = pd.DataFrame({'embedding': text_embeddings, 'value': values[:len(text_embeddings)]})
    df.to_csv(file_path, index=False, errors='ignore')

def generate_embeddings(data, tokenizer, model):
    embeddings = []
    targets = []
    for text, target in data:
        # Tokenizza il testo utilizzando il tokenizer di BERT
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        # Ottieni gli input_ids e le attention_mask
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']

        # Passa gli embeddings al modello BERT
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # Embeddings dell'ultimo stato nascosto

        # Calcola l'embedding medio di tutti i token (escluso il padding)
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # Aggiungi l'embedding medio e la variabile target alla lista degli embeddings
        embeddings.append(mean_embeddings)
        targets.append(target)

    return torch.stack(embeddings), torch.tensor(targets, dtype=torch.float)

def train_model(text_embeddings, values, cls_model, epochs):
    optimizer = AdamW(cls_model.parameters(), lr=2e-5)
    cls_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        outputs = cls_model(inputs_embeds=text_embeddings, labels=values.unsqueeze(1))
        loss = criterion(outputs.logits, values.unsqueeze(1))
        # Backward pass
        loss.backward()
        optimizer.step()
        # print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
    return cls_model
def test(model, test_text_embeddings, test_values):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs_embeds=test_text_embeddings)
        logits = outputs.logits
        preds = torch.sigmoid(logits)
        # Calcola la loss
        test_labels = test_values.unsqueeze(1).float()
        # Converte le probabilità in classi binarie (0 o 1)
        preds = (preds > 0.5).long()
        accuracy = accuracy_score(test_values, preds)
    return accuracy

def forward(model, text_embeddings):
    with torch.no_grad():
        outputs = model(inputs_embeds=text_embeddings)
        logits = outputs.logits
        preds = torch.sigmoid(logits)
    return preds

if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    criterion = torch.nn.BCEWithLogitsLoss()

    # Define the file path of the dataset
    file_path = 'dataset/polblogs.gml'

    # Read the gml file
    graph = ig.Graph.Read_GML(file_path)

    # obtain the blogs links
    blog_labels = graph.vs["label"]
    values = graph.vs["value"]

    urls = blog_labels

    urls_with_prefix = [add_https_prefix(url) for url in urls]
    print(urls_with_prefix)

    # print(texts[8])       primo testo utilizzabile per il training

    if not os.path.exists('data.csv'):
        texts = [get_text_from_url(url) for url in urls_with_prefix]  # prova ad usare tutti i link

        data = [(text, value) for text, value in zip(texts, values) if text is not None and text != '']  # link funzionanti
        np.random.shuffle(data)
        values = [value for _, value in data]  # value dei link funzionanti
        text_embeddings, values = generate_embeddings(data)
        text_embeddings_csv = np.squeeze(text_embeddings)
        text_embeddings_str = [text_embedding.tolist() for text_embedding in text_embeddings_csv]
        save_data_to_csv(text_embeddings_str, values, 'data.csv')
    else:
        data = pd.read_csv('data.csv')
        data.sample(frac=1, random_state=1).reset_index(drop=True)
        emb = data['embedding']
        values = data['value']

        emb = emb.apply(ast.literal_eval)

        embeddings_array = np.array(emb.tolist())

        embeddings_array = embeddings_array[:, np.newaxis, :]

        text_embeddings = torch.tensor(embeddings_array, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

    training_text_embeddings, val_text_embedding, training_values, val_values = (
        train_test_split(text_embeddings, values, test_size=0.2, random_state=1))

    param_grid = BertConfig(
        hidden_size=768,
        num_hidden_layers=[6],
        num_attention_heads=[6],
        hidden_dropout_prob=[0.3],
        intermediate_size=3072,
        hidden_act=['relu'],
        initializer_range=[0.1],
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        type_vocab_size=2,
        vocab_size=30522,
        num_labels=1,
        attention_probs_dropout_prob=[0.3, 0.5]
    )

    best_accuracy = 0
    for num_hidden_layers, num_attention_heads, hidden_act, initializer_range, attention_probs_dropout_prob, hidden_dropout_prob in (
            itertools.product(param_grid.num_attention_heads, param_grid.num_attention_heads,
                              param_grid.hidden_act, param_grid.initializer_range,
                              param_grid.attention_probs_dropout_prob, param_grid.hidden_dropout_prob)):
        config = BertConfig(
            hidden_size=768,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=3072,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            initializer_range=initializer_range,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            type_vocab_size=2,
            vocab_size=30522,
            num_labels=1,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )
        cls_model = BertForSequenceClassification(config)
        optimizer = AdamW(model.parameters(), lr=2e-5)

        # alleno tutti i modelli
        cls_model = train_model(training_text_embeddings, training_values, cls_model, epochs=20)

        val_accuracy = test(cls_model, val_text_embedding, val_values)
        print(f'Validation accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = config

    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Best parameters:", best_params)

    # Evaluate on test set with best parameters
    best_model = BertForSequenceClassification(best_params)
    best_model.save_pretrained('best_model')



