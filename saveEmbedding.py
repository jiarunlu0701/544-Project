import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
torch.set_num_threads(12)

# 1. Load the dataset
df = pd.read_csv("finalDataset.csv")

X = df[['answer', 'ConstructName', 'QuestionText']].astype(str)
X = "answer: " + X['answer'] + " " + "ConstructName: " + X['ConstructName'] + " " + "QuestionText: " + X['QuestionText']
Y = df['MisconceptionName']

# Encode the target labels
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)  # Convert to numerical labels

# 2. Define a Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")
model = AutoModelForSequenceClassification.from_pretrained("tbs17/MathBERT", num_labels=len(label_encoder.classes_))

# 3. Create Dataset and DataLoader
dataset = TextDataset(X.tolist(), Y, tokenizer)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Set up optimizer, loss, and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 10
total_steps = len(data_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = torch.nn.CrossEntropyLoss()

# 5. Define training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train()

    losses = 0
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses += loss.item()
        correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / n_examples, losses / len(data_loader)

# 6. Training Loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_acc, train_loss = train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, len(df))
    print(f"Train loss {train_loss} accuracy {train_acc}")

# 7. Save the fine-tuned model
model.save_pretrained("finetuned_mathbert")
tokenizer.save_pretrained("finetuned_mathbert")