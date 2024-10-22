import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Set the number of CPU threads (experiment with this number based on your CPU capabilities)
torch.set_num_threads(12)  # Adjust based on your CPU

df = pd.read_csv("finalDataset.csv")

X = df[['answer', 'ConstructName', 'QuestionText']].astype(str)
X = "answer: " + X['answer'] + " " + "ConstructName: " + X['ConstructName'] + " " + "QuestionText: " + X['QuestionText']
Y = df['MisconceptionName']

# Use AutoModel instead of AutoModelForMaskedLM to access hidden states
tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")
model = AutoModel.from_pretrained("tbs17/MathBERT")

# Tokenize the input texts in batches (batch size can be adjusted based on memory)
batch_size = 32  # You can adjust the batch size depending on your memory
embeddings_list = []

# Process inputs in batches
for i in range(0, len(X), batch_size):
    batch_texts = X[i:i + batch_size].tolist()

    # Tokenize the batch
    inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Generate embeddings using MathBERT
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling of the last hidden state as the embedding

    embeddings_list.append(embeddings)

# Concatenate all embeddings
embeddings = torch.cat(embeddings_list, dim=0)

# Save the embeddings and labels using torch
torch.save(embeddings, "mathbert_embeddings.pt")  # Save embeddings
print("Embeddings saved successfully!")

embeddings = torch.load("mathbert_embeddings.pt")
print("Loaded embeddings shape:", embeddings.shape)

