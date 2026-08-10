# ML From Scratch — Research & Portfolio Repo

This repo is where I implement machine learning / deep learning / GenAI concepts **from scratch** — mainly for learning, teaching, and portfolio purposes. Each experiment is self-contained (usually a notebook) and documented below so I can pick up context quickly later, and so others can follow along.

## Repo Philosophy

- Every experiment lives in its own notebook/folder, named after what it implements (e.g. `distil_bert_pt_implementation.ipynb`).
- The goal is **understanding via reimplementation**: instead of just calling `AutoModelForSequenceClassification`, I rebuild the architecture (embeddings, attention, encoder blocks, etc.) in raw PyTorch to internalize how it actually works.
- This README is a living index. Every time I push a new experiment, I add an entry to the [Experiments Log](#experiments-log) below following the same template.

## Repo Structure

```
.
├── README.md
├── distil_bert_pt_implementation.ipynb   # Experiment 1 (see below)
└── loading_pretrained_weights_to_custom_model.ipynb # Experiment 2 (see below)
```

## Environment Setup

```bash
pip install -q transformers datasets accelerate evaluate scikit-learn huggingface_hub torch
```

- Python 3.10+
- GPU recommended (notebooks check `torch.cuda.is_available()` and fall back to CPU)
- Developed/run on Google Colab (checkpoints saved to Google Drive) — swap the checkpoint path if running locally

---

## Experiments Log

### 1. DistilBERT — Built From Scratch in PyTorch (Sentiment Classification)
**File:** `distil_bert_pt_implementation.ipynb`

A from-scratch PyTorch reimplementation of the DistilBERT architecture, trained as a 3-class sentiment classifier.

**What's learned and implemented:**

*Phase 1 — HuggingFace abstraction (no custom code):*
- `AutoTokenizer` / `AutoModelForSequenceClassification` — loaded pretrained DistilBERT with a randomly initialized classification head
- Dataset tokenization — mapped raw text to `input_ids`/`attention_mask` via the tokenizer
- `TrainingArguments` + `Trainer.train()` — fine-tuned on `cardiffnlp/tweet_eval` (3-class sentiment) using the high-level Trainer API, ~74% validation accuracy on a Colab T4
- Purpose: establish a working baseline and understand the abstraction end-to-end before reimplementing any of it manually
- `print(model)` , `model.state_dict()` , `inspect.getsource(class.__init__/class.forward)` — The base model has been properly inspected with its constructor and forward source code

*Phase 2 — DistilBERT from scratch (custom PyTorch):*
- `DistilBertEmbeddings` — word + learned positional embeddings, LayerNorm, dropout
- `SelfAttention` — single-head scaled dot-product attention (built first, for intuition)
- `MultiHeadSelfAttention` — 12-head attention (768 hidden size, 64 head-dimension)
- `FeedForward` — position-wise FFN (768 → 3072 → 768, GELU activation)
- `TransformerBlock` — attention + FFN with residual connections and Add & Norm
- `TransformerEncoder` — stack of 6 Transformer blocks executing sequentially
- `DistilBertForSentimentClassification` — full model: embeddings → encoder → pre-classifier → classifier head (combined all sub-modules into a final class)
- `NaiveClassifier` — a minimal baseline (embeddings + linear head only, no attention) used to sanity-check that attention actually helps

**Dataset:** [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (`sentiment` config) — tweets labeled `negative` / `neutral` / `positive`.

**Tokenizer:** pretrained `distilbert-base-uncased` tokenizer from Hugging Face (WordPiece vocab reused; only the model weights/architecture are trained from scratch — not the tokenizer).

**Training setup:**
- Tokenized to max length 128, padded to `max_length`
- `DataCollatorWithPadding` + PyTorch `DataLoader` (batch size 16, shuffled)
- Loss: `CrossEntropyLoss` | Optimizer: `AdamW` (lr=2e-5)
- Manual training loop (forward → loss → backward → optimizer step) over 3 epochs
- Checkpoints (model + optimizer state, epoch, losses, accuracy) saved to Google Drive and reloadable for inference/resuming

**Sanity checks / unit tests included in notebook:**
- Embedding layer output shape
- Single-head vs multi-head attention output shapes
- Full encoder forward pass
- Naive (no-attention) classifier vs full model, to see the effect of self-attention
- Manual inference function `predict_sentiment(text, model, tokenizer, device)`

**Status:** ✅ architecture implemented, trained, checkpointed, and reload-tested end-to-end.

**Possible next steps:**
- [ ] Evaluate properly on the `validation`/`test` split (accuracy/F1) rather than spot-checking
- [ ] Compare against the real pretrained `distilbert-base-uncased` fine-tuned the standard (Hugging Face `Trainer`) way
- [ ] Add attention-weight visualization
- [ ] Try loading actual pretrained DistilBERT weights into this custom architecture (weight-porting exercise)

---

<!--
Template for the next experiment — copy/paste and fill in:

### N. <Experiment Title>
**File:** `<notebook or folder name>`

Short description of what this experiment explores/implements.

**What's implemented:**
- ...

**Dataset:**

**Training setup:**

**Status:**

**Possible next steps:**
- [ ]
-->

## Notes to Future Me

- Keep each notebook runnable top-to-bottom on a fresh Colab runtime (re-check `pip install` cell and Drive mount paths).
- Prefer small, well-named modules (`nn.Module` per concept) over one giant class — makes it easier to unit-test and reuse across experiments.
- Log shapes liberally with comments (`# (B, seq_len, hidden_size)`) — future me always forgets.
