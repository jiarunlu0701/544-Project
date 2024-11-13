import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,logging
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType

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
dir_path = parent_str + "LLM_finetune_tinhang/output_model"


class MisconceptionDataset(Dataset):
    def __init__(self, data_preparer, indices, y_train, tokenizer, max_length=512):
        self.data_preparer = data_preparer
        self.indices = indices
        self.y_train = y_train
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        row = self.data_preparer.df.iloc[index]

        # Generate the prompt based on the input data
        prompt = generate_prompt(row)

        # Get the expected misconception as the target from y_train
        misconception = self.y_train[index]

        # Tokenize the input and target texts separately
        inputs = self.tokenizer(prompt, padding="max_length", max_length=self.max_length, truncation=True,
                                return_tensors="pt")
        labels = self.tokenizer(misconception, padding="max_length", max_length=self.max_length, truncation=True,
                                return_tensors="pt")

        # Convert labels to a 1D tensor
        labels = labels.input_ids.squeeze()

        # Set tokens with padding token ID to -100 so that they’re ignored in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels,
        }


# Function to create prompt-response pairs
def generate_prompt(row):
    prompt = (
        "Instruction: Why is the given answer wrong under such circumstances?\n"
        f"answer: {row['answer']}\n"
        f"ConstructName: {row['ConstructName']}\n"
        f"QuestionText: {row['QuestionText']}"
    )
    return prompt


# Main function for fine-tuning the model
def main():
    # Load the tokenizer and model from the local directory
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model.gradient_checkpointing_enable()
    # Set pad token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
    data_preparer = DataPreparer(file_path=file_path)
    X_train, X_test, y_train, y_test = data_preparer.prepare_data()
    train_indices = list(X_train.keys())


    train_indices = train_indices[:5] # for experimentation purpose


    # Initialize the training dataset
    train_dataset = MisconceptionDataset(data_preparer, train_indices, y_train, tokenizer)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Low-rank dimension; adjust based on memory capacity and needs
        lora_alpha=16,  # Scaling parameter for LoRA layers
        lora_dropout=0.05,  # Dropout rate for LoRA layers
        target_modules=["q_proj", "v_proj"],  # Target layers for LoRA
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Set up training arguments with fp16
    training_args = TrainingArguments(
        output_dir=dir_path,
        overwrite_output_dir=True,
        num_train_epochs=3,  # Adjust based on your data and needs
        per_device_train_batch_size=4,  # Adjust based on GPU memory capacity
        fp16=True,  # Enable fp16 for reduced memory usage
        gradient_checkpointing=True,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        evaluation_strategy="no",
        weight_decay=0.01,
        learning_rate=1e-4,  # Adjust based on model and data requirements
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Start fine-tuning
    trainer.train()

    # Save the fine-tuned model with LoRA
    model.save_pretrained(dir_path)

    print("Fine-tuning with LoRA complete. Model saved.")


if __name__ == "__main__":
    main()