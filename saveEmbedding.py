import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Adjust based on your CPU
torch.set_num_threads(12)

df = pd.read_csv("finalDataset.csv")

X = df[['answer', 'ConstructName', 'QuestionText']].astype(str)
X = "answer: " + X['answer'] + " " + "ConstructName: " + X['ConstructName'] + " " + "QuestionText: " + X['QuestionText']
Y = df['MisconceptionName']

tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")
model = AutoModel.from_pretrained("tbs17/MathBERT")

batch_size = 32
embeddings_list = []

# Process inputs in batches
for i in range(0, len(X), batch_size):
    batch_texts = X[i:i + batch_size].tolist()

    inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling of the last hidden state as the embedding

    embeddings_list.append(embeddings)

embeddings = torch.cat(embeddings_list, dim=0)

torch.save(embeddings, "mathbert_embeddings.pt")  # Save embeddings
print("Embeddings saved successfully!")

embeddings = torch.load("mathbert_embeddings.pt")
print("Loaded embeddings shape:", embeddings.shape)

