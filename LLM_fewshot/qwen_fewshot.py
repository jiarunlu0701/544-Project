import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from tqdm import tqdm

import random
# Add your project directory to the system path
parent_str = "C:/Users/jiaru/OneDrive/Desktop/544-Project/"
sys.path.append(parent_str)
from data_preparer import DataPreparer

# Set logging to ignore warnings
logging.set_verbosity_error()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model from the local directory
# model_path = parent_str + "converted_llama"
# model_path = 'EleutherAI/llemma_7b'
model_path = "Qwen/Qwen2.5-3B-Instruct"
file_path = parent_str + "finalDataset.csv"
dir_path = parent_str + "evaluating_pipeline/output/few_shot_model_responses_qwen.csv"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16).to(device)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Function to create a prompt based on a row of data
def generate_prompt(row, X_train, y_train, data_preparer):
    # print(X_train)
    keys_list = list(X_train.keys())

    # Sample 5 indices uniformly from the list
    sampled_indices = random.sample(range(len(keys_list)), 5)

    # Get the keys corresponding to the sampled indices
    sampled_keys = [keys_list[i] for i in sampled_indices]
    # print("hi im here")
    # print(len(sampled_indices))
    example = ''
    for index in sampled_keys:
        example += f'Example{index}, Question :'
        example += data_preparer.df.iloc[index]['QuestionText']
        example += '\n'
        example += 'Answer: '
        example += data_preparer.df.iloc[index]['answer']
        example += '\n'
    prompt = (
        "Instruction: Why is the given answer wrong under such circumstances? Some of the examples are given below\n"
        f"Example from before: \n{example}\n"
        f"ConstructName: {row['ConstructName']}\n"
        f"QuestionText: {row['QuestionText']}\n"
        f"answer: {row['answer']}"
    )
    # print(prompt)
    return prompt

# Batch generation function with mixed precision
def generate_text_batch(prompts, max_length=512):
    responses = []
    for prompt in prompts:
        # Prepare the chat messages
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. Imagine now you are a math expert. I provided several examples, and try to answer it like examples. Point out the mistakes directly and do not provide reasoning steps"},
            {"role": "user", "content": prompt}
        ]
        # Convert to the Qwen-compatible format
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # Tokenize and move to device
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_length,
                pad_token_id=tokenizer.eos_token_id
            )
        # Decode the generated response
        generated_response = tokenizer.decode(
            generated_ids[0][len(model_inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        responses.append(generated_response.strip())
    return responses


if __name__ == "__main__":
    # Initialize DataPreparer and load data
    data_preparer = DataPreparer(file_path=file_path)
    X_train, X_test, y_train, y_test = data_preparer.prepare_data()

    # Set batch size for processing
    batch_size = 16  # Adjust based on your GPU’s memory capacity
    results = []

    # Iterate over the test data in batches
    for i in tqdm(range(0, len(X_test), batch_size), desc="Generating prompts in batches"):
        # Select batch of indices and prepare prompts
        batch_indices = list(X_test.keys())[i:i + batch_size]
        batch_prompts = [generate_prompt(data_preparer.df.iloc[index], X_train, y_train, data_preparer) for index in batch_indices]
        batch_ground_truths = [data_preparer.df.iloc[index]['MisconceptionName'] for index in batch_indices]
        
        # Generate text for the batch of prompts
        batch_generated_texts = generate_text_batch(batch_prompts, max_length=512)
        
        # Append results for each entry in the batch
        for j, myindex in enumerate(batch_indices):
            results.append({
                'index': myindex,
                "Prompt": batch_prompts[j],
                "Generated Response": batch_generated_texts[j],
                "Expected Misconception": batch_ground_truths[j]
            })
    
    # Save the results DataFrame as a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(dir_path, index=False)
    
    print("Responses saved to model_responses.csv")



#token
#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