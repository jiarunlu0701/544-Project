This is the repo for an NLP project where we aim to point out the reasoning mistakes in a mathematical question and answer pair setting





Gen_cot dir: [text](Gen_cot/generator.py)

generator.py contains class Gen_cot can takes the argument to select model, input text, prompt input selection, and output. 

provide a list of model to select, local finetune model path are from  (LLM_finetune_tinhang/output_model/checkpoint-2500) and [text](LLM_finetune_tinhang/output_model/checkpoint-2952). and huggingface model_path = "Qwen/Qwen2.5-3B-Instruct". 

the output is the chain of thoughts for a math problem based on the input math problem. 

create a checking program to check if the answer is correct, if not, auto redo the response with gen cot to enhance the response.

evaluation.

