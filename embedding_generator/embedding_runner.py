import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from textDataset import TextDataset  # Import the custom dataset class here
import json
# Rest of your code here
if __name__ == '__main__':
    torch.set_num_threads(os.cpu_count())

    # Load your dataset
    df = pd.read_csv('misconception_mapping.csv')
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    # Extract the MisconceptionName and MisconceptionId columns
    df_text = df[['MisconceptionName', 'MisconceptionId']]

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    dataset = TextDataset(df_text['MisconceptionName'], df_text['MisconceptionId'], tokenizer)

# Set num_workers to the number of CPU cores available
    num_workers = os.cpu_count()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=num_workers)
    print('hi im here')
    # Function to generate embeddings and keep track of MisconceptionId
    def generate_embeddings(model, dataloader):
        model.eval()
        embeddings_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing Batches", unit="batch"):
                # Move inputs to device
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                
                # Retrieve misconception IDs from the batch
                misconception_ids = batch['misconception_id']
                
                # Get the model outputs
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use mean pooling to get sentence embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Append each embedding with its misconception_id
                for embedding, misconception_id in zip(embeddings.cpu(), misconception_ids):
                    embeddings_list.append({
                        'MisconceptionId': int(misconception_id),  # Convert to int
                        'Embedding': embedding.numpy().tolist()  # Convert to list
                    })
        
        return embeddings_list

    # Generate embeddings with progress tracking
    embeddings_with_ids = generate_embeddings(model, dataloader)

    # Convert embeddings to a DataFrame if needed
    embeddings_df = pd.DataFrame(embeddings_with_ids)
    # Save the DataFrame to a CSV file
    embeddings_df.to_json("embeddings_with_ids.json", orient="records")


