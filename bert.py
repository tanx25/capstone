import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset

model_name = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

data = {
    'sentence1': ["Playing basketball requires teamwork"],
    'sentence2': ["Tom plays alone at home"],
}

df = pd.DataFrame(data)

encoding = tokenizer(df['sentence1'].tolist(), df['sentence2'].tolist(), padding=True, truncation=True,
                     return_tensors="pt", max_length=128)

labels = torch.tensor([0])

dataset = TensorDataset(encoding['input_ids'], encoding['attention_mask'], labels)

train_loader = DataLoader(dataset, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    for batch in train_loader:
        b_input_ids, b_attention_mask, b_labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

with torch.no_grad():
    for batch in train_loader:
        b_input_ids, b_attention_mask, b_labels = tuple(t.to(device) for t in batch)
        outputs = model(b_input_ids, attention_mask=b_attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        if predicted.item() == 1:
            print("Follows")
        else:
            print("Not follows")
