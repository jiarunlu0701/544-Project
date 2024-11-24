import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the tokenizer and model once for all sections
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.eval()  # Set to evaluation mode

# Shared function to generate embeddings
def get_embedding(text):
    # Tokenize and process the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        outputs = model(**inputs)
        
    # Take the mean of the last hidden state to create a single embedding vector
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Shared function to find the most similar misconception ID
def find_most_similar_id(target_embedding, ground_truth_df):
    # Calculate cosine similarity between the target embedding and each embedding in ground_truth
    similarities = cosine_similarity([target_embedding], list(ground_truth_df['Embedding']))
    
    # Find the index of the highest similarity score
    most_similar_index = np.argmax(similarities)
    
    # Retrieve the misconception_id with the highest similarity
    most_similar_id = ground_truth_df.iloc[most_similar_index]['MisconceptionId']
    return most_similar_id

# Function to calculate accuracy metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Zero-shot Section
def zero_shot_processing(ground_truth_path, target_csv, misconception_csv):
    ground_truth = pd.read_json(ground_truth_path)
    targetDf = pd.read_csv(target_csv)
    
    # Generate embeddings
    embeddings = []
    for text in tqdm(targetDf['Generated Response'], desc="Generating embeddings (Zero-shot)"):
        embeddings.append(get_embedding(text))
    targetDf['Embeddings_Generated'] = embeddings

    # Load misconception mapping
    misconception_df = pd.read_csv(misconception_csv)

    # Find most similar misconception ID
    targetDf['prediction_result'] = targetDf['Embeddings_Generated'].apply(
        lambda emb: find_most_similar_id(emb, ground_truth)
    )
    targetDf.rename(columns={'prediction_result': 'MisconceptionId'}, inplace=True)
    targetDf = targetDf.merge(misconception_df, on='MisconceptionId', how='left')

    # Metrics
    y_true = targetDf['Expected Misconception']
    y_pred = targetDf['MisconceptionName']
    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)

    print("Zero-shot Accuracy:", accuracy)
    print("Zero-shot Precision:", precision)
    print("Zero-shot Recall:", recall)
    print("Zero-shot F1 Score:", f1)

# Few-shot Section
def few_shot_processing(ground_truth_path, target_csv, misconception_csv):
    ground_truth = pd.read_json(ground_truth_path)
    targetDf = pd.read_csv(target_csv)
    
    # Generate embeddings
    embeddings = []
    for text in tqdm(targetDf['Generated Response'], desc="Generating embeddings (Few-shot)"):
        embeddings.append(get_embedding(text))
    targetDf['Embeddings_Generated'] = embeddings

    # Load misconception mapping
    misconception_df = pd.read_csv(misconception_csv)

    # Find most similar misconception ID
    targetDf['prediction_result'] = targetDf['Embeddings_Generated'].apply(
        lambda emb: find_most_similar_id(emb, ground_truth)
    )
    targetDf.rename(columns={'prediction_result': 'MisconceptionId'}, inplace=True)
    targetDf = targetDf.merge(misconception_df, on='MisconceptionId', how='left')

    # Metrics
    y_true = targetDf['Expected Misconception']
    y_pred = targetDf['MisconceptionName']
    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)

    print("Few-shot Accuracy:", accuracy)
    print("Few-shot Precision:", precision)
    print("Few-shot Recall:", recall)
    print("Few-shot F1 Score:", f1)

# Fine-tuned Section
def fine_tune_processing(ground_truth_path, target_csv, misconception_csv):
    ground_truth = pd.read_json(ground_truth_path)
    targetDf = pd.read_csv(target_csv)
    
    # Generate embeddings
    embeddings = []
    for text in tqdm(targetDf['Generated Response'], desc="Generating embeddings (Fine-tuned)"):
        embeddings.append(get_embedding(text))
    targetDf['Embeddings_Generated'] = embeddings

    # Load misconception mapping
    misconception_df = pd.read_csv(misconception_csv)

    # Find most similar misconception ID
    targetDf['prediction_result'] = targetDf['Embeddings_Generated'].apply(
        lambda emb: find_most_similar_id(emb, ground_truth)
    )
    targetDf.rename(columns={'prediction_result': 'MisconceptionId'}, inplace=True)
    targetDf = targetDf.merge(misconception_df, on='MisconceptionId', how='left')

    # Metrics
    y_true = targetDf['Expected Misconception']
    y_pred = targetDf['MisconceptionName']
    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)

    print("Fine-tuned Accuracy:", accuracy)
    print("Fine-tuned Precision:", precision)
    print("Fine-tuned Recall:", recall)
    print("Fine-tuned F1 Score:", f1)

# Main execution
if __name__ == "__main__":
    ground_truth_path = r'C:\Users\jiaru\OneDrive\Desktop\544-Project\evaluating_pipeline\groud_truth_embedding.json'
    zero_shot_csv = r'C:\Users\jiaru\OneDrive\Desktop\544-Project\evaluating_pipeline\output\zero_shot_model_responses_qwen.csv'
    few_shot_csv = r'C:\Users\jiaru\OneDrive\Desktop\544-Project\evaluating_pipeline\output\few_shot_model_responses_qwen.csv'
    fine_tune_csv = r'C:\Users\jiaru\OneDrive\Desktop\544-Project\evaluating_pipeline\output\finetune_model_responses_qwen.csv'
    misconception_csv = r'C:\Users\jiaru\OneDrive\Desktop\544-Project\embedding_generator\misconception_mapping.csv'

    print("Processing Zero-shot...")
    zero_shot_processing(ground_truth_path, zero_shot_csv, misconception_csv)

    print("Processing Few-shot...")
    few_shot_processing(ground_truth_path, few_shot_csv, misconception_csv)

    print("Processing Fine-tuned...")
    fine_tune_processing(ground_truth_path, fine_tune_csv, misconception_csv)