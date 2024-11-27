import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def evaluate_file(file_path, output_col, ground_truth_col):

    # Load CSV file
    data = pd.read_csv(file_path)

    # Extract predictions and references
    predictions = data[output_col].astype(str).tolist()
    references = data[ground_truth_col].astype(str).tolist()

    # BLEU Score
    bleu_scores = [sentence_bleu([ref.split()], pred.split()) for pred, ref in zip(predictions, references)]

    # ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]

    # BERTScore
    P, R, F1 = score(predictions, references, lang="en", verbose=False)
    bert_scores = {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}

    # Aggregate Metrics
    metrics = {
        "Average BLEU": sum(bleu_scores) / len(bleu_scores),
        "Average ROUGE-1": sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores),
        "Average ROUGE-2": sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores),
        "Average ROUGE-L": sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores),
        "BERTScore": bert_scores
    }

    return metrics

def evaluate_multiple_files(file_paths, output_col, ground_truth_col):
    results = {}
    for file_path in file_paths:
        print(f"Evaluating file: {file_path}")
        metrics = evaluate_file(file_path, output_col, ground_truth_col)
        results[file_path] = metrics
        print(f"Metrics for {file_path}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        print("-" * 50)
    return results

# Example Usage
if __name__ == "__main__":
    # Specify your file paths
    file_paths = [
        script_dir + "/output/few_shot_model_responses_qwen.csv",
        script_dir + "/output/finetune_model_responses_qwen.csv",
        script_dir + "/output/zero_shot_model_responses_qwen_incontext.csv",
        script_dir + "/output/zero_shot_model_responses_qwen.csv"
    ]
    output_col = "Generated Response"  # Replace with the actual column name for LLM outputs
    ground_truth_col = "Expected Misconception"  # Replace with the actual column name for ground truth

    evaluate_multiple_files(file_paths, output_col, ground_truth_col)
