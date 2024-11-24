# Reasoning Mistake Detection in Mathematical QA Pairs

This repository is part of an NLP project that identifies reasoning mistakes in mathematical question-and-answer (QA) pairs. The system utilizes multiple strategies—**Zero-shot**, **Few-shot**, and **Fine-tuned** models—to evaluate reasoning and output a detailed chain of thought (CoT) for the given input.

## Project Overview

The aim of this project is to:
1. Detect reasoning mistakes in QA pairs for mathematical problems
2. Use chain-of-thought (CoT) prompting to provide detailed explanations of errors
3. Automate iterative responses to refine answers based on detected mistakes

## Key Components

### 1. Model Integration

- **Pretrained Hugging Face Model:**
  - Model: `Qwen/Qwen2.5-3B-Instruct`
- **Fine-tuned Models:**
  - Checkpoints:
    - `LLM_finetune_tinhang/output_model/checkpoint-2500`
    - `LLM_finetune_tinhang/output_model/checkpoint-2952`

### 2. Chain of Thought (CoT) Generation

The `Gen_cot` module (located in `Gen_cot/generator.py`) provides functionality to:
- Select models for reasoning evaluation
- Generate CoT responses for input mathematical problems
- Improve response quality iteratively by detecting errors and re-generating corrected outputs

### 3. Technical Implementation

The system uses several key libraries and frameworks:
```python
- pandas: Data manipulation and analysis
- torch: Deep learning operations
- transformers: Hugging Face's transformer models
- sklearn: Metrics calculation and similarity measures
- numpy: Numerical operations
- tqdm: Progress tracking
```

## How It Works

### Input
A mathematical problem paired with a reasoning-based answer.

### Processing Steps

1. **Embedding Generation**
   - Input is tokenized and converted into a numerical representation using the `DistilBERT` model
   - Embeddings are generated via the mean pooling of the model's last hidden state
   - Implementation includes truncation and padding to a max length of 512 tokens

2. **Reasoning Evaluation**
   - Three evaluation approaches are implemented:
     - Zero-shot analysis
     - Few-shot learning
     - Fine-tuned model evaluation
   - Each approach generates embeddings and computes similarity with ground truth data

3. **Error Correction**
   - If a mistake is detected, the system uses `Gen_cot` to refine the response iteratively
   - Misconception mapping is used to categorize and identify specific types of errors

4. **Evaluation Metrics**
   - Performance metrics are calculated for each approach:
     - **Accuracy**: Overall correctness of predictions
     - **Precision**: Accuracy of positive predictions
     - **Recall**: Ability to find all relevant cases
     - **F1 Score**: Harmonic mean of precision and recall

### Output
- A Chain of Thought explaining the detected reasoning error
- An improved answer with enhanced reasoning
- Detailed performance metrics for model evaluation

## Key Functions

### Embedding Generation
```python
def get_embedding(text):
    # Tokenize and process input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                      padding="max_length", max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Mean pooling of last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings
```

### Similarity Matching
```python
def find_most_similar_id(target_embedding, ground_truth_df):
    # Calculate cosine similarity
    similarities = cosine_similarity([target_embedding], 
                                   list(ground_truth_df['Embedding']))
    
    # Find most similar misconception
    most_similar_index = np.argmax(similarities)
    return ground_truth_df.iloc[most_similar_index]['MisconceptionId']
```

## Performance Evaluation

The system calculates comprehensive metrics for each approach:
```python
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1
```

These metrics help in comparing the effectiveness of different approaches (Zero-shot, Few-shot, and Fine-tuned) in detecting reasoning mistakes.
