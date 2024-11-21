import os
import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


class Gen_cot:
    def __init__(self, model_path, prompt_file):
        self.model_path = model_path
        self.prompt_file = prompt_file
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.prompt = self.load_prompt()

    def load_model(self):
        print(f"Loading model from: {self.model_path}")
        return AutoModelForCausalLM.from_pretrained(self.model_path)

    def load_tokenizer(self):
        print(f"Loading tokenizer from: {self.model_path}")
        return AutoTokenizer.from_pretrained(self.model_path)

    def load_prompt(self):
        if not os.path.exists(self.prompt_file):
            raise FileNotFoundError(f"Prompt file {self.prompt_file} not found.")
        with open(self.prompt_file, 'r') as file:
            return file.read()

    def generate_chain_of_thought(self, input_text):

        input_prompt = f"{self.prompt}\n{input_text}\n"
        
        # Tokenize the input and generate attention_mask
        inputs = self.tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Generate model output
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Explicitly pass attention mask
            max_new_tokens=200
        )
        
        # Decode the result
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Prepare structured JSON response
        response = {
            "model": self.model_path,
            "prompt": self.prompt_file,
            "input_problem": input_text,
            "chain_of_thought": result,
        }
        return response

if __name__ == "__main__":
    try:
        # Existing model options:
        # 1. "LLM_finetune_tinhang/output_model/checkpoint-2500"
        # 2. "LLM_finetune_tinhang/output_model/checkpoint-2952"
        # 3. "Qwen/Qwen2.5-3B-Instruct"
        selected_model = "Qwen/Qwen2.5-3B-Instruct"  # Default model here

        # Ensure the Gen_cot directory exists and contains prompt files
        if not os.path.exists("Gen_COT"):
            raise FileNotFoundError("The Gen_cot directory does not exist.")

        # List available prompt files in the Gen_cot directory
        prompt_files = [f for f in os.listdir("Gen_COT/prompts") if f.endswith(".txt")]
        if not prompt_files:
            raise FileNotFoundError("No prompt files found in the Gen_COT directory.")

        # Default prompt file (set based on available files in the Gen_cot directory)
        # Example: "Gen_cot/math_prompt.txt" or any other prompt file
        selected_prompt = os.path.join("Gen_COT/prompts", prompt_files[0])  # Change index as needed

        # Show user the default configuration
        print(f"Using model: {selected_model}")
        print(f"Using prompt: {selected_prompt}")
 
        # Create generator instance
        generator = Gen_cot(model_path=selected_model, prompt_file=selected_prompt)

        # Parse the math problem from the command-line arguments, or use the default
        math_problem = (
            sys.argv[1]
            if len(sys.argv) > 1
            else "What is the sum of 123 and 456?"  # Default problem
        )

        # Generate chain of thought and get the JSON response
        response = generator.generate_chain_of_thought(math_problem)

        # Print the JSON response
        print(json.dumps(response, indent=4))

    except Exception as e:
        error_response = {"error": str(e)}
        print(json.dumps(error_response, indent=4))
