import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from tqdm import tqdm
from torch.cuda.amp import autocast

# Add your project directory to the system path
sys.path.append("C:/Users/jiaru/OneDrive/Desktop/544-Project")
from data_preparer import DataPreparer

# Set logging to ignore warnings
logging.set_verbosity_error()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model from the local directory
model_path = r"C:\Users\jiaru\OneDrive\Desktop\544-Project\converted_llama"
file_path = r"C:\Users\jiaru\OneDrive\Desktop\544-Project\finalDataset.csv"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Function to create a prompt based on a row of data
def generate_prompt(row):
    prompt = (
        "Instruction: Why is the given answer wrong under such circumstances?\n"
        f"answer: {row['answer']}\n"
        f"ConstructName: {row['ConstructName']}\n"
        f"QuestionText: {row['QuestionText']}"
    )
    return prompt

# Batch generation function with mixed precision
def generate_text_batch(prompts, max_length=1000):
    # Tokenize the input prompts in a batch
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    
    # Generate tokens using the model with mixed precision
    with torch.no_grad():
        with autocast():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # Decode all generated sequences in the batch
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

if __name__ == "__main__":
    # Initialize DataPreparer and load data
    data_preparer = DataPreparer(file_path=file_path)
    X_train, X_test, y_train, y_test = data_preparer.prepare_data()

    # Set batch size for processing
    batch_size = 8  # Adjust based on your GPU’s memory capacity
    results = []

    # Iterate over the test data in batches
    for i in tqdm(range(0, len(X_test), batch_size), desc="Generating prompts in batches"):
        # Select batch of indices and prepare prompts
        batch_indices = list(X_test.keys())[i:i + batch_size]
        batch_prompts = [generate_prompt(data_preparer.df.iloc[index]) for index in batch_indices]
        batch_ground_truths = [data_preparer.df.iloc[index]['MisconceptionName'] for index in batch_indices]
        
        # Generate text for the batch of prompts
        batch_generated_texts = generate_text_batch(batch_prompts, max_length=1000)
        
        # Append results for each entry in the batch
        for j, index in enumerate(batch_indices):
            results.append({
                "Prompt": batch_prompts[j],
                "Generated Response": batch_generated_texts[j],
                "Expected Misconception": batch_ground_truths[j]
            })
    
    # Save the results DataFrame as a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv("../evaluating pipeline/model_responses.csv", index=False)
    
    print("Responses saved to model_responses.csv")



#token
#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