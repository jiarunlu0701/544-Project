

from data_preparer import DataPreparer
from transformers import pipeline
from transformers import logging
import pandas as pd
from tqdm import tqdm
import torch
logging.set_verbosity_error()
device = 0 if torch.cuda.is_available() else -1
# Initialize the text-generation pipeline with the LLeMa model
pipe = pipeline("text-generation", model="EleutherAI/llemma_7b", device = device)


# Function to create a prompt based on a row of data
def generate_prompt(row):
    prompt = (
        "Instruction: Why is the given answer wrong under such circumstances?\n"
        f"answer: {row['answer']}\n"
        f"ConstructName: {row['ConstructName']}\n"
        f"QuestionText: {row['QuestionText']}"
    )
    return prompt


if __name__ == "__main__":
    # Initialize DataPreparer and load data
    data_preparer = DataPreparer(file_path="../finalDataSet.csv")
    X_train, X_test, y_train, y_test = data_preparer.prepare_data()

    # Iterate over the test data to generate prompts and get model responses
    results = []
    for index, value in tqdm(X_test.items(), desc="Generating prompts"):
        model_input = data_preparer.df.iloc[index]
        prompt = generate_prompt(model_input)
        ground_truth = data_preparer.df.iloc[index]['MisconceptionName']
        print('hi im here')
        response = pipe(prompt, max_length=1000, num_return_sequences=1)
        generated_text = response[0]['generated_text']
        results.append({
            "Prompt": prompt,
            "Generated Response": generated_text,
            "Expected Misconception": ground_truth  # Assuming y_test is aligned with X_test
        })
    results_df = pd.DataFrame(results)

    # Save the DataFrame as a CSV file
    results_df.to_csv("../evaluating pipeline/model_responses.csv", index=False)

    print("Responses saved to model_responses.csv")