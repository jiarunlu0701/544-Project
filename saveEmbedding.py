import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

# Define a custom Dataset class
class SentimentAnalysisDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts.tolist()
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def prepare_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    X = df[['answer', 'ConstructName', 'QuestionText']].astype(str)
    X = "answer: " + X['answer'] + " " + "ConstructName: " + X['ConstructName'] + " " + "QuestionText: " + X[
        'QuestionText']
    Y = df['MisconceptionName']

    # Encode the labels
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    return train_test_split(X, Y, test_size=0.1, random_state=42), label_encoder


def create_dataloaders(train_x, train_y, dev_x, dev_y, tokenizer, batch_size=32):
    # Prepare datasets
    train_dataset = SentimentAnalysisDataset(train_x, train_y, tokenizer)
    dev_dataset = SentimentAnalysisDataset(dev_x, dev_y, tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size // 2, shuffle=False)

    return train_dataloader, dev_dataloader


def train_and_evaluate(train_dataloader, dev_dataloader, model, device, epochs=10, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float('inf')
    model.to(device)

    num_training_steps = epochs * len(train_dataloader)
    with tqdm(total=num_training_steps, desc="Fine-tuning") as pbar:
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0

            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.update(1)

            avg_train_loss = train_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}")

            # Evaluation phase
            model.eval()
            dev_loss = 0

            with torch.no_grad():
                for batch in dev_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)
                    dev_loss += loss.item()

            avg_dev_loss = dev_loss / len(dev_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_dev_loss}")

            # Save the best model checkpoint
            if avg_dev_loss < best_loss:
                best_loss = avg_dev_loss
                best_model = model.state_dict()
                torch.save(best_model, 'best_model_checkpoint.pth')
                print(f"Best model saved with Validation Loss: {avg_dev_loss}")


def save_embeddings(dataloader, model, device, output_file="fine_tuned_embeddings.pt"):
    model.eval()
    embeddings_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Saving Embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.hidden_states[-1][:, 0,
                         :]  # Take the [CLS] token's embedding from the last hidden layer
            embeddings_list.append(embeddings.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    torch.save(embeddings, output_file)
    print(f"Embeddings saved to {output_file}")


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data and dataloaders
    (train_x, dev_x, train_y, dev_y), label_encoder = prepare_data("finalDataset.csv")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('tbs17/MathBERT')
    model = AutoModelForSequenceClassification.from_pretrained(
        'tbs17/MathBERT',
        num_labels=len(label_encoder.classes_),
        output_hidden_states=True  # Enable output of hidden states
    )

    # Create dataloaders
    train_dataloader, dev_dataloader = create_dataloaders(train_x, train_y, dev_x, dev_y, tokenizer, batch_size=32)

    # Train and evaluate
    train_and_evaluate(train_dataloader, dev_dataloader, model, device, epochs=10, learning_rate=2e-5)

    # Save the fine-tuned embeddings
    save_embeddings(dev_dataloader, model, device, output_file="fine_tuned_embeddings.pt")