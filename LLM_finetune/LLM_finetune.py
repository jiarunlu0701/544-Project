import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, logging
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

# Add your project directory to the system path
parent_str = "C:/Users/jiaru/OneDrive/Desktop/544-Project/"
sys.path.append(parent_str)
from data_preparer import DataPreparer

# Set logging to ignore warnings
logging.set_verbosity_error()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model from the local directory
model_path = "Qwen/Qwen2.5-3B-Instruct"
file_path = parent_str + "finalDataset.csv"
dir_path = parent_str + "LLM_finetune_tinhang/output_model"
mydataset_path = parent_str + "LLM_finetune/LLM_finetune_tinhang/evaluation_dataset/"


def generate_chat_prompt(row, tokenizer):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": f"Why is the given answer wrong?\nAnswer: {row['answer']}\n"
                                     f"ConstructName: {row['ConstructName']}\n"
                                     f"QuestionText: {row['QuestionText']}"}
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return chat_prompt


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

        # Generate the chat-based prompt
        prompt = generate_chat_prompt(row, self.tokenizer)

        # Get the expected misconception as the target
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


def main():
    # Load the tokenizer and model
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

    # Initialize the training dataset
    train_dataset = MisconceptionDataset(data_preparer, train_indices, y_train, tokenizer)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=dir_path,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        evaluation_strategy="no",
        weight_decay=0.01,
        learning_rate=1e-4,
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
    temp = "test_x.csv"
    X_test.to_csv(mydataset_path + temp, index=False)
    temp = "test_y.csv"
    y_test.to_csv(mydataset_path + temp, index=False)


if __name__ == "__main__":
    main()
 