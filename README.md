# 🧠 CNN Text Classifier — SST-2 Sentiment Analysis

> CNN text classifier on SST-2 — explores filter sizes, pooling, and GloVe embeddings. Benchmarked against Naive Bayes, Logistic Regression, and SVM baselines.

---

## Overview

This project implements the **Kim (2014)** multi-filter Convolutional Neural Network architecture for binary sentiment classification on the **SST-2 (Stanford Sentiment Treebank)** dataset. The notebook systematically investigates what makes CNNs work for text — and where they fall short.

## Investigations

| Topic | What's explored |
|---|---|
| 🔍 Filter patterns | How convolutional filters learn to detect sentiment-bearing n-grams |
| 📏 Filter size | Impact of n-gram window size: unigram, bigram, trigram, multi-scale |
| 🏊 Pooling | Max-over-time vs. average pooling |
| 🔤 Embeddings | Random init vs. frozen GloVe vs. fine-tuned GloVe (50d) |
| 📉 Limitations | CNN accuracy degradation on longer sentences (long-range dependencies) |

## Results

| Model | Val Accuracy |
|---|---|
| Naïve Bayes (BoW) | ~81% |
| Logistic Regression (TF-IDF) | ~84% |
| Linear SVM (TF-IDF) | ~85% |
| CNN — baseline | ~84% |
| **CNN — best (GloVe fine-tuned)** | **~87%** |

## Architecture

```
Embedding → [Conv1d(k=2) ‖ Conv1d(k=3) ‖ Conv1d(k=4)]
         → MaxPool-over-time → Concat → Dropout(0.5) → Linear(2)
```

- **Vocab size:** 20,000 tokens  
- **Sequence length:** 64 tokens (padded/truncated)  
- **Filters:** 100 per filter size (300 total feature maps)  
- **Optimizer:** Adam (lr=1e-3) with ReduceLROnPlateau  

## Dataset

[SST-2](https://huggingface.co/datasets/glue/viewer/sst2) — Stanford Sentiment Treebank (binary). Part of the GLUE benchmark.

| Split | Examples |
|---|---|
| Train | 67,349 |
| Validation | 872 |

## Getting Started

### Requirements

```bash
pip install datasets transformers torch scikit-learn matplotlib seaborn gensim
```

### Run

Open and run `A1_CNN_Text_Classification.ipynb` top-to-bottom. All data is downloaded automatically via Hugging Face Datasets. GloVe vectors are fetched via `gensim.downloader` (~70MB).

Expected runtime: **~15–30 min on CPU**, faster with GPU.

## Project Structure

```
.
├── A1_CNN_Text_Classification.ipynb   # Main notebook
├── README.md                          # This file
```

## Key Findings

- **Multi-scale filters (2,3,4) outperform any single filter size** — different linguistic phenomena operate at different granularities
- **Max pooling > average pooling** for short-text sentiment — one salient phrase is often enough to determine label
- **Fine-tuned GloVe > frozen > random** — transfer learning helps even on large training sets
- **CNNs degrade on longer sentences** — structural limitation vs. attention-based models (BERT)

## References

- Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification.* EMNLP.
- Socher et al. (2013). *Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank.* EMNLP.
- Pennington et al. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP.

---

*CS455 — Natural Language Processing with Deep Learning, Spring 2026*
