from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

class AspectBasedSentimentDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=128):
        self.data = self._load_data(filename)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        encoding = self.tokenizer(item['text'], max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding['labels'] = torch.tensor(item['label'])
        return encoding

    def _load_data(self, filename):
        data = []
        with open(filename, 'r') as file:
            for line in file:
                fields = line.strip().split('\t')
                label = {'positive': 2, 'neutral': 1, 'negative': 0}[fields[0]]
                aspect_category = fields[1]
                target_term = fields[2]
                sentence = fields[4]
                text = f"{aspect_category} {target_term} [SEP] {sentence}"
                data.append({'text': text, 'label': label})
        return data

class Classifier:
    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3)

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        train_dataset = AspectBasedSentimentDataset(train_filename, self.tokenizer)
        dev_dataset = AspectBasedSentimentDataset(dev_filename, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=16)

        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        num_epochs = 5
        for epoch in range(num_epochs):
            self.model.train()
            for batch in train_loader:
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in dev_loader:
                    input_ids = batch['input_ids'].squeeze(1).to(device)
                    attention_mask = batch['attention_mask'].squeeze(1).to(device)
                    labels = batch['labels'].to(device)

                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    predictions = outputs.logits.argmax(dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total
            print(f"Epoch {epoch+1}/{num_epochs}, Dev Accuracy: {accuracy:.4f}")

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        dataset = AspectBasedSentimentDataset(data_filename, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=16)

        self.model.to(device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                batch_predictions = outputs.logits.argmax(dim=1)
                predictions.extend(batch_predictions.tolist())

        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predicted_labels = [label_map[pred] for pred in predictions]
        return predicted_labels