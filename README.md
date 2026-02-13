# Cyber Sovereignty – Spam Email Classification

Evaluation of LLMs on spam classification using single-prompt (`/api/generate`) mode.

## Datasets

| Dataset     | Samples | Ham | Spam |
|------------|---------|-----|------|
| **TREC 2007** | 3,771 | 1,248 (33%) | 2,523 (67%) |
| **Ling**      | 578   | 482 (83%)   | 96 (17%)    |

## Results

### 1. TREC 2007 Spam Corpus

| Model         | Accuracy | Macro P | Macro R | Macro F1 | Ham P/R/F1   | Spam P/R/F1   |
|---------------|----------|---------|---------|----------|--------------|---------------|
| llama3.1:8b   | **98.78%** | 0.9862 | 0.9862 | 0.9862 | 0.98 / 0.98 / 0.98 | 0.99 / 0.99 / 0.99 |
| deepseek-r1:8b| **95.78%** | 0.9447 | 0.9653 | 0.9535 | 0.90 / 0.99 / 0.94 | 0.99 / 0.94 / 0.97 |
| mistral:7b    | 87.99%  | 0.8619 | 0.8709 | 0.8661 | 0.80 / 0.84 / 0.82 | 0.92 / 0.90 / 0.91 |
| falcon3:7b    | 83.15%  | 0.8135 | 0.8505 | 0.8209 | 0.68 / 0.90 / 0.78 | 0.95 / 0.80 / 0.86 |
| gemma3:4b     | 82.92%  | 0.8975 | 0.7426 | 0.7700 | 1.00 / 0.49 / 0.65 | 0.80 / 1.00 / 0.89 |

### 2. Ling Spam Corpus

| Model         | Accuracy | Macro P | Macro R | Macro F1 | Legit P/R/F1  | Spam P/R/F1   |
|---------------|----------|---------|---------|----------|---------------|---------------|
| llama3.1:8b   | **98.62%** | 0.9615 | 0.9917 | 0.9758 | 1.00 / 0.98 / 0.99 | 0.92 / 1.00 / 0.96 |
| deepseek-r1:8b| **98.79%** | 0.9727 | 0.9844 | 0.9784 | 1.00 / 0.99 / 0.99 | 0.95 / 0.98 / 0.96 |
| mistral:7b    | 92.56%  | 0.8453 | 0.9554 | 0.8852 | 1.00 / 0.91 / 0.95 | 0.69 / 1.00 / 0.82 |
| falcon3:7b    | 92.39%  | 0.8429 | 0.9544 | 0.8829 | 1.00 / 0.91 / 0.95 | 0.69 / 1.00 / 0.81 |
| gemma3:4b     | 32.35%  | 0.5986 | 0.5944 | 0.3235 | 1.00 / 0.19 / 0.32 | 0.20 / 1.00 / 0.33 |

## Setup

```bash
pip install -r requirements.txt
```

## Run Evaluation

```bash
# TREC 2007
python eval_trec_ollama.py generate --models llama3.1:8b

# Ling
python eval_ling_ollama.py generate --models llama3.1:8b
```

## Output

- `trec2007/generate/<model>/` – `predictions.jsonl`, `eval_report.txt`
- `ling/generate/<model>/` – `predictions.jsonl`, `eval_report.txt`

## Ollama

Ensure [Ollama](https://ollama.ai) is running locally with the required models pulled (e.g. `ollama pull llama3.1:8b`).
