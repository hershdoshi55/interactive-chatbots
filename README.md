# Interactive Chatbots: A New Approach from Turn-Based LLMs

**Group 17** — Aryan Amberkar, Vihaansh Majithia, Hersh Doshi, Hanshul Bahl
University of Illinois Urbana-Champaign

---

## Overview

Traditional LLM chatbots operate in a strict turn-based model: the user finishes typing, then the system responds. This is unlike natural human conversation, where participants interrupt, overlap, and interject roughly **40% of the time** and interrupt approximately **once every 60 seconds** in collaborative settings.

This project implements a **3-tier interruptible chatbot** that breaks this constraint. Instead of waiting for the user to finish, the chatbot monitors each word as it's typed, decides when it has enough context to interrupt, speculates the rest of the user's message, and issues a proactive response — all before the user is done typing.

The design is inspired by *"Speculative Ad-hoc Querying"* (UT Austin / Microsoft Research / AWS) and *"Single-agent or Multi-agent Systems? Why Not Both?"* (UIUC Siebel School).

---

## Motivation

- Response delays longer than **500 ms** noticeably reduce perceived naturalness and engagement in human-agent interaction
- Allowing text-based overlap increased **user satisfaction by 23%** and reduced **perceived latency by 35%**, even when total task time remained constant
- Well-timed interruptions improved **task completion speed by up to 27%** compared to strictly turn-based systems
- StreamingLLM and related architectures demonstrate real-time token processing at **< 100 ms/token**, making proactive responses technically feasible

Current solutions (TurnGPT, Voice Activity Projection, StreamingLLM) optimize *when* to speak or *how* to stream, but not *why* or *whether* to cut in. This project addresses that gap with a learned, context-aware interrupt policy.

---

## Architecture

The pipeline has three sequential components:

```
User typing  →  [Verification Model]  →  [Speculative Model]  →  [Response Model]
                 (DistilBERT)              (T5-small)              (DeepSeek-V3.2)
                 66M parameters            60M parameters           671B parameters
                 Detect when               Guess what               Say it
```

### 1. Verification Model — Fine-tuned DistilBERT (66M parameters)

Answers: *"Is this prefix worth interrupting on?"*

**Base model specs:**
| Property | Value |
|---|---|
| Architecture | DistilBERT (`distilbert-base-uncased`) |
| Total parameters | **66 million** |
| Transformer layers | 6 (distilled from BERT's 12) |
| Hidden dimension | 768 |
| Attention heads | 12 |
| Max sequence length | 512 tokens |
| Parameter reduction vs BERT-base | **40% fewer parameters** |
| Speed vs BERT-base | **60% faster inference** |
| Performance retention vs BERT-base | **~97%** on GLUE benchmarks |

A binary classification head (768 → 2, linear, ~1,538 parameters) is added on top of the `[CLS]` token embedding to output interruptible / not-interruptible logits.

**Inference logic:**
- Processes each word prefix incrementally as the user types
- Outputs a probability score (class 1 = interruptible) via softmax over logits
- Scores are smoothed word-by-word with an **Exponential Moving Average (EMA)**:

```
smooth_i = 0.6 × score_i + 0.4 × smooth_{i-1}
```

- An interrupt fires when `smooth > 0.65`

### 2. Speculative Model — Fine-tuned T5-small (60M parameters)

Answers: *"What did the user intend to say?"*

**Base model specs:**
| Property | Value |
|---|---|
| Architecture | T5-small (`t5-small`) |
| Total parameters | **60 million** |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Hidden dimension (`d_model`) | 512 |
| Feed-forward dimension (`d_ff`) | 2,048 |
| Attention heads | 8 |
| Vocabulary size | 32,128 tokens (SentencePiece) |
| Architecture type | Encoder-decoder (seq2seq) |

Takes the interrupted prefix as input (formatted as `"prefix: {text}"`) and generates a JSON object with two fields:
- `completed_user_message` — the most likely full message
- `overprojected_user_message` — the completed message plus one extra clause for richer context

Only `completed_user_message` is used in the live pipeline. The overprojected version is available for future use.

**Example:**
```
Prefix:    "What is the capital"
Completed: "What is the capital of the United States?"
```

### 3. Response Model — DeepSeek-V3.2 (671B parameters, MoE)

Answers: *"What should the chatbot say?"*

**Model specs:**
| Property | Value |
|---|---|
| Architecture | Mixture-of-Experts (MoE) Transformer |
| Total parameters | **671 billion** |
| Active parameters per token | **~37 billion** (only active experts) |
| Access method | HuggingFace OpenAI-compatible router |
| Endpoint | `https://router.huggingface.co/v1` |
| Temperature | `0.4` |
| Response target | 1–3 sentences |

Receives the speculated full message along with the full multi-turn conversation history (`messages` list) and returns a short, polite interrupt response.

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `ALPHA` | `0.6` | EMA smoothing constant (weight on current score) |
| `THRESHOLD` | `0.65` | Interrupt trigger — smoothed probability must exceed this |
| Verifier `max_length` | `64` tokens | DistilBERT tokenizer truncation |
| Speculator input `max_length` | `96` tokens | T5 encoder input truncation |
| Speculator `max_new_tokens` | `64` | T5 decoder generation budget |
| Speculator word cap | `50` words | Post-processing truncation for runaway completions |
| Responder `temperature` | `0.4` | Controls response randomness (lower = more focused) |
| Verifier beam search | greedy (1 beam) | deterministic, fast inference |
| Speculator `do_sample` | `False` | greedy decoding |

---

## Training Data

Both local models are trained on a mix of **WikiText-2** and **SQuAD**, sourced from HuggingFace `datasets`.

| Dataset | Split | Approximate size |
|---|---|---|
| WikiText-2 (training) | train | ~2.08 million tokens, ~600 articles |
| WikiText-2 (validation) | validation | ~217,000 tokens |
| SQuAD (training) | train | 87,599 questions, 442 Wikipedia articles |
| SQuAD (validation) | validation | 10,570 questions |

WikiText sentences are filtered to ≥ 6 words (verifier) / ≥ 8 words (speculator). SQuAD questions are filtered to ≥ 4 words (verifier) / ≥ 6 words (speculator). Both datasets are shuffled and combined before sampling.

---

## Training

### Verification Model (`train_verification2.py`)

| Setting | Value |
|---|---|
| Base model | `distilbert-base-uncased` (66M parameters) |
| Task | Binary sequence classification |
| Training epochs | 2 |
| Train batch size | 16 |
| Eval batch size | 16 |
| Learning rate | `4e-5` |
| Weight decay | `0.01` |
| Max samples per split | 2,000 source sentences |
| Generated labeled examples per split | up to **~8,000** (4 labels per source sentence) |
| Positive : negative label ratio | **1 : 3** |
| Input tokenizer `max_length` | 64 tokens (padded + truncated) |
| Min source sentence length | 6 words |
| Prefix cutoff range | word 3 → word `len - 1` |
| Optimizer | AdamW (HuggingFace Trainer default) |
| LR scheduler | Linear warmup + decay |

**Label generation — 4 examples per source sentence:**

| Label | Count | Strategy | Rationale |
|---|---|---|---|
| 1 (interruptible) | 1 | Prefix from word 0 to random cutoff (3 to `len-1`) | Natural, coherent sentence beginning |
| 0 (not interruptible) | 1 | Same words, randomly shuffled | Incoherent word order despite valid vocabulary |
| 0 (not interruptible) | 1 | Middle chunk: word `k` to word `k + cutoff` | Lacks a concrete subject at position 0 |
| 0 (not interruptible) | 1 | 1–2 word prefix | Insufficient context for any interrupt |

### Speculative Model (`spec_model2.py`)

| Setting | Value |
|---|---|
| Base model | `t5-small` (60M parameters) |
| Task | Conditional text generation (prefix → JSON) |
| Training epochs | 2 |
| Train batch size | 8 |
| Eval batch size | 8 |
| Learning rate | `4e-5` |
| Weight decay | `0.01` |
| Max samples per split | 2,000 |
| Input tokenizer `max_length` | 96 tokens |
| Target tokenizer `max_length` | 128 tokens |
| Input format | `"prefix: {partial text}"` |
| Output format | `{"completed_user_message": "...", "overprojected_user_message": "..."}` |
| Min source sentence length | 8 words |
| Prefix cutoff range | word 3 → word `len - 3` |
| Optimizer | AdamW |
| LR scheduler | Linear warmup + decay |

Overprojected outputs are constructed by appending one of 5 fixed extra clauses to the completed sentence (e.g., `"with additional details for clarity."`), mirroring the over-projection concept from the UT Austin speculative querying paper.

---

## Results & Evaluation

### Verification Model Results

Evaluated on the WikiText-2 validation set. Only sentences with ≥ 10 words are used. Each sentence generates 3 labeled prefixes (1 valid, 1 shuffled, 1 mid-sentence).

| Metric | Score |
|---|---|
| **Accuracy** | **86.67%** |
| **Precision** | **77.5%** |

The model successfully learned to distinguish syntactically and semantically valid sentence beginnings from incoherent word combinations. Random word orderings and mid-sentence snippets were reliably classified as non-interruptible.

### Speculative Model Results

Evaluated using `difflib.SequenceMatcher` similarity between generated completions and ground-truth full sentences (longest common contiguous substring ratio).

| Observation | Detail |
|---|---|
| Similarity scores | Low relative to ground truth |
| Primary cause | T5-small (60M params) has limited generative capacity for open-ended completion |
| Secondary cause | Natural language completion is a many-to-one problem; exact string match is an inherently hard target |
| Model produces | Plausible, human-readable completions that are useful in context even without exact match |

Higher-capacity models (T5-base at 250M parameters or T5-large at 770M parameters) are expected to significantly close this gap.

### Live Pipeline Results (from paper)

**Example 1 — Factual question:**
```
Input:    "What is the capital of the US?"   [7 words total]
i=3  score=0.978  smooth=0.587  | What is the
i=4  score=0.979  smooth=0.822  | What is the capital
[INTERRUPT] at word 4 of 7  (57% typed, EMA=0.822)
Prefix:    What is the capital
Completed: What is the capital of the United States?
Assistant: The capital of the United States is Washington, D.C.
```

**Example 2 — Technical question:**
```
Input:    "Why is gradient descent used in neural networks?"
[INTERRUPT] at word 4  ("Why is gradient descent")
```

**Example 3 — Declarative sentence:**
```
Input:    "The lion is the king of the jungle"   [8 words total]
[INTERRUPT] at word 6 of 8  (75% typed, "The lion is the king of")
```

**Example 4 — Incoherent input (no interrupt):**
```
Input:    "Hello yes the where that what?"
Result:   No interrupt triggered  (EMA never exceeded 0.65)
```

**Example 5 — Overprojection enriching the response:**
```
Input:    "What do bears eat in North America?"   [7 words total]
i=3  score=0.966  smooth=0.580  | What do bears
i=4  score=0.900  smooth=0.820  | What do bears eat
[INTERRUPT] at word 4 of 7  (57% typed, EMA=0.820)
Completed (overprojected): "What do bears eat?, including a brief explanation."
→ The extra clause prompts the response model to produce a more detailed answer
```

### Interrupt Timing Summary

| Example input | Total words | Interrupt at word | % of message typed at interrupt |
|---|---|---|---|
| "What is the capital of the US?" | 7 | 4 | 57% |
| "Why is gradient descent used in neural networks?" | 8 | 4 | 50% |
| "The lion is the king of the jungle" | 8 | 6 | 75% |
| "What do bears eat in North America?" | 7 | 4 | 57% |
| "Hello yes the where that what?" | 6 | — | no interrupt |

On average, across valid inputs, the system interrupts after roughly **50–60% of the message is typed**, giving the chatbot a meaningful head start on generating a response before the user finishes.

---

## File Structure

```
interactive-chatbots/
├── interruptible_chatbot.py          # Full 3-model chatbot (main entry point)
├── train_verification2.py            # Fine-tune DistilBERT verifier (66M params)
├── spec_model2.py                    # Fine-tune T5-small speculator (60M params)
├── inference_verification.py         # Interactive verifier test (word-by-word EMA)
├── test_spec.py                      # Interactive speculator test
├── evaluate_interruptible_chatbot.py # Comprehensive evaluation suite (5 modes)
├── eval_texts_example.jsonl          # Example evaluation data (5 entries)
├── verification_model/               # Saved DistilBERT weights + tokenizer
└── speculative_model/                # Saved T5-small weights + tokenizer
```

---

## Setup & Usage

### Dependencies

```bash
pip install torch transformers datasets openai
```

### Environment

The response model requires a HuggingFace token with access to the inference router:

```bash
export HF_TOKEN=your_huggingface_token
```

### Quick Start

**Step 1 — Train the verification model** (DistilBERT, 66M params, ~2,000 samples, 2 epochs):
```bash
python train_verification2.py
```

**Step 2 — Train the speculative model** (T5-small, 60M params, ~2,000 samples, 2 epochs):
```bash
python spec_model2.py
```

**Step 3 — Run the interruptible chatbot:**
```bash
python interruptible_chatbot.py
```

### Chatbot Commands

| Command | Effect |
|---|---|
| *(type a message)* | Word-by-word EMA scoring; triggers interrupt when `smooth > 0.65` |
| `/send` | Skip interrupt logic; send a message directly to DeepSeek-V3.2 |
| `/reset` | Clear conversation history |
| `/quit` | Exit |

### Individual Model Testing

```bash
# Test verifier alone — no HF_TOKEN required
python inference_verification.py

# Test speculator alone — no HF_TOKEN required
python test_spec.py
```

### Evaluation Suite

```bash
# Verifier classification metrics only
python evaluate_interruptible_chatbot.py --data eval_texts_example.jsonl --mode verifier

# All metrics (requires HF_TOKEN for speculation + response modes)
python evaluate_interruptible_chatbot.py --data eval_texts_example.jsonl --mode all --model deepseek-ai/DeepSeek-V3.2
```

**Evaluation modes:**

| Mode | Metrics computed |
|---|---|
| `verifier` | Accuracy, precision, recall, F1, ROC-AUC |
| `timing` | Average trigger word index, miss rate, triggered fraction |
| `speculation` | Token F1, edit similarity (completed vs. ground truth) |
| `response` | Speculator latency (ms), responder latency (ms), pipeline latency (ms), response token length |
| `all` | All of the above |

**Eval data format (JSONL):**
```jsonl
{"text": "What is the time complexity of Dijkstra's algorithm using a binary heap?"}
{"prefix": "What is the capital", "label": 1}
```

---

## Model Size Comparison

| Model | Role | Parameters | Type | Hosted |
|---|---|---|---|---|
| DistilBERT-base-uncased | Verifier | **66M** | Encoder-only Transformer | Local (fine-tuned) |
| T5-small | Speculator | **60M** | Encoder-decoder Transformer | Local (fine-tuned) |
| DeepSeek-V3.2 | Responder | **671B** (37B active/token) | MoE Transformer | HuggingFace router |

Total locally trained parameters: **~126 million** across both fine-tuned models.

---

## Related Work

| Paper | Key Quantitative Finding | Relevance |
|---|---|---|
| *Speculative Ad-hoc Querying* — Li et al. (UT Austin / Microsoft Research / AWS) | 3 levels of speculation (L0 exact match, L1 subset, L2 directional) | Source of the speculator/over-projector concept |
| *Single-agent or Multi-agent Systems? Why Not Both?* — Gao et al. (UIUC) | MAS uses 4–220× more prefill tokens, 2–12× more decode tokens; hybrid raised accuracy 1–12% while cutting costs 20–88% | Motivation for confidence-based escalation design |
| *Efficient Streaming Language Models with Attention Sinks* — Xiao et al. | < 100 ms/token latency with attention sink cache | Establishes real-time LLM response as feasible |
| *Beyond Turn-taking* — Kim et al. | Text overlap: +23% satisfaction, −35% perceived latency | Core motivation statistic for interrupt value |
| *Eliciting Spoken Interruptions* — Edwards et al. | Interrupts ~1×/60 sec in collaborative settings | Grounding for interrupt frequency target |
| *Incremental Dialogue Management* — Kennington et al. | Survey of incremental NLU/NLG frameworks | Background on word-by-word processing |
| *Applying General Turn-Taking Models to HRI* — Skantze et al. | >500 ms delays reduce perceived naturalness | Latency threshold motivation |

---

## Limitations & Future Work

- **Speculator capacity** — T5-small (60M parameters) underperforms on open-ended generation. T5-base (250M) or T5-large (770M) would trade training time for significantly better completion quality.
- **Training data scale** — Both models are trained on only 2,000 samples per split. Scaling to the full WikiText-103 (~103M tokens) or a conversational corpus would materially improve generalization.
- **Dataset diversity** — WikiText-2 and SQuAD cover encyclopedic and factual QA patterns. Including conversational corpora (DailyDialog, MultiWOZ) would better cover informal register.
- **Class imbalance** — The 1:3 positive-to-negative training ratio for the verifier biases toward conservative interrupts; tuning this ratio or using weighted loss could improve recall.
- **Overprojection in production** — `overprojected_user_message` is generated at inference time but only used opportunistically; a routing layer that selects between `completed` and `overprojected` based on confidence would better exploit it.
- **Multi-agent escalation** — A confidence-based router (inspired by the SAS/MAS paper) could escalate borderline cases (EMA within ±0.05 of threshold) to a secondary verifier before committing to an interrupt.
- **Latency benchmarking** — Measuring wall-clock time saved per exchange (interrupt at word 4 of 7 = ~43% of typing time saved) would provide a concrete performance metric.

---
