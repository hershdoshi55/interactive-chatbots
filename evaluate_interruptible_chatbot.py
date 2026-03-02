
"""
evaluate_interruptible_chatbot.py

Evaluates metrics for a 3-tier interruptible chatbot:
  (1) Verification model (DistilBERT): when to interrupt
  (2) Speculative model (LLM): complete prefix -> completed + overprojected user message
  (3) Response model (LLM): generate interrupt response

Works in two modes:
  A) If your JSONL has {"text": "..."} lines, we generate positive/negative prefixes and run:
       - verifier classification metrics (acc/prec/recall/F1/AUC)
       - interrupt timing metrics (avg trigger word, miss rate, false interrupt rate)
       - optional speculation/response metrics (needs HF/OpenAI-style router access)
  B) If your JSONL has {"prefix": "...", "label": 0/1}, we run verifier classification metrics directly.

Run examples:
  python evaluate_interruptible_chatbot.py --data eval_texts.jsonl --mode verifier
  python evaluate_interruptible_chatbot.py --data eval_texts.jsonl --mode all --model deepseek-ai/DeepSeek-V3.2
"""

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# -------------------------
# Similarity helpers (no extra deps)
# -------------------------

def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", s.lower())

def token_f1(pred: str, gold: str) -> float:
    p = tokenize(pred)
    g = tokenize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    from collections import Counter
    cp = Counter(p)
    cg = Counter(g)
    overlap = sum((cp & cg).values())
    prec = overlap / max(1, len(p))
    rec  = overlap / max(1, len(g))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def levenshtein(a: str, b: str) -> int:
    a = normalize(a)
    b = normalize(b)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    # DP with two rows
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,      # deletion
                         cur[j - 1] + 1,   # insertion
                         prev[j - 1] + cost)  # substitution
        prev = cur
    return prev[m]

def edit_similarity(pred: str, gold: str) -> float:
    pred_n = normalize(pred)
    gold_n = normalize(gold)
    if not pred_n and not gold_n:
        return 1.0
    dist = levenshtein(pred_n, gold_n)
    denom = max(1, max(len(pred_n), len(gold_n)))
    return 1.0 - dist / denom

# -------------------------
# Metrics helpers
# -------------------------

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1

def roc_auc(scores: List[float], labels: List[int]) -> float:
    # AUC via rank statistic (equivalent to Mann–Whitney U)
    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    rank = 1
    sum_ranks_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                sum_ranks_pos += avg_rank
        rank += (j - i)
        i = j
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc

# -------------------------
# Verifier wrapper
# -------------------------

class DistilBertVerifier:
    """
    Uses your existing inference_verification.py style model folder:
      ./verification_model
    Computes score_prefix(prefix) -> probability of class 1.
    """
    def __init__(self, model_dir: str = "./verification_model"):
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def score_prefix(self, prefix: str) -> float:
        enc = self.tokenizer(prefix, return_tensors="pt", truncation=True, max_length=64)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            logits = self.model(**enc).logits
            prob = self.torch.softmax(logits, dim=-1)[0, 1].item()
        return float(prob)

# -------------------------
# Speculator + Response (OpenAI-compatible HF router)
# -------------------------

ROUTER_BASE_URL_DEFAULT = "https://router.huggingface.co/v1"

def _extract_json_object(s: str):
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start:end+1])
    except Exception:
        return None

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    return s.strip()

class Speculator:
    def __init__(self, model_name: str, base_url: str, api_key_env: str = "HF_TOKEN"):
        from openai import OpenAI
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {api_key_env} in environment.")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def speculate(self, prefix: str) -> Dict[str, str]:
        prefix = prefix.strip()
        if not prefix:
            return {"completed_user_message": "", "overprojected_user_message": ""}

        system = (
            "You are a speculator/autocomplete engine for an interruptible chatbot.\n"
            "The user is still typing. You see ONLY their current prefix.\n"
            "Your job: predict the most likely full user message.\n\n"
            "Return STRICT JSON with exactly these keys:\n"
            "  completed_user_message\n"
            "  overprojected_user_message\n\n"
            "Rules:\n"
            "- Keep completed_user_message short and natural.\n"
            "- overprojected_user_message should be the same idea, but add ONE extra clause "
            "  to make it more specific/helpful (not a whole essay).\n"
            "- Do NOT answer the question. Only rewrite/complete the user message.\n"
        )

        user = f"PREFIX:\n{prefix}\n\nReturn JSON only."

        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
        )
        latency_s = time.time() - t0

        raw = resp.choices[0].message.content or ""
        raw = strip_code_fences(raw)
        obj = _extract_json_object(raw)

        if not obj:
            return {"completed_user_message": raw.strip(),
                    "overprojected_user_message": raw.strip(),
                    "_latency_s": latency_s}

        c = str(obj.get("completed_user_message", "")).strip()
        o = str(obj.get("overprojected_user_message", "")).strip() or c
        return {"completed_user_message": c, "overprojected_user_message": o, "_latency_s": latency_s}

class Responder:
    def __init__(self, model_name: str, base_url: str, api_key_env: str = "HF_TOKEN"):
        from openai import OpenAI
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {api_key_env} in environment.")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def respond(self, speculated_message: str) -> Dict[str, str]:
        system = (
            "You are an interruptible chatbot. You are cutting in while the user is typing.\n"
            "Be short, polite, and helpful. Prefer a quick clarification question if needed.\n"
            "Keep it to 1-3 sentences.\n"
        )
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": speculated_message}],
            temperature=0.4,
        )
        latency_s = time.time() - t0
        out = (resp.choices[0].message.content or "").strip()
        return {"response": out, "_latency_s": latency_s}

# -------------------------
# Data loading + prefix generation
# -------------------------

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def generate_prefix_dataset_from_texts(rows: List[dict], max_samples: int = 2000, seed: int = 0) -> Tuple[List[str], List[int], List[str]]:
    """
    Returns: prefixes, labels, full_texts_for_positive_prefixes (same length as prefixes; "" if label=0)
    Mirrors your training logic: positive prefix, plus 3 negative variants.
    """
    random.seed(seed)
    prefixes: List[str] = []
    labels: List[int] = []
    full_for_pos: List[str] = []

    texts = [r["text"] for r in rows if "text" in r and isinstance(r["text"], str)]
    random.shuffle(texts)

    for text in texts:
        words = text.strip().split()
        if len(words) < 6:
            continue
        cutoff = random.randint(3, len(words) - 1)

        # positive
        pos = " ".join(words[:cutoff])
        prefixes.append(pos); labels.append(1); full_for_pos.append(text)

        # neg1: shuffled
        shuffled = words[:]
        random.shuffle(shuffled)
        neg1 = " ".join(shuffled[:cutoff])
        prefixes.append(neg1); labels.append(0); full_for_pos.append("")

        # neg2: middle chunk
        max_start = len(words) - cutoff - 1
        if max_start >= 1:
            start = random.randint(1, max_start)
            neg2 = " ".join(words[start:start+cutoff])
            prefixes.append(neg2); labels.append(0); full_for_pos.append("")

        # neg3: short (1-2 words)
        neg3 = " ".join(words[:random.randint(1, 2)])
        prefixes.append(neg3); labels.append(0); full_for_pos.append("")

        if len(prefixes) >= max_samples:
            break

    return prefixes, labels, full_for_pos

def first_interrupt_word(verifier: DistilBertVerifier, full_text: str, alpha: float, threshold: float) -> Optional[int]:
    words = full_text.strip().split()
    smooth = 0.0
    for i in range(1, len(words) + 1):
        prefix = " ".join(words[:i])
        score = verifier.score_prefix(prefix)
        smooth = alpha * score + (1 - alpha) * smooth
        if smooth > threshold:
            return i
    return None

# -------------------------
# Main evaluation
# -------------------------

@dataclass
class EvalConfig:
    alpha: float
    threshold: float
    model_dir: str
    base_url: str
    llm_model: str
    seed: int
    max_samples: int
    do_speculation: bool
    do_response: bool

def eval_verifier(prefixes: List[str], labels: List[int], verifier: DistilBertVerifier, threshold: float) -> dict:
    scores = [verifier.score_prefix(p) for p in prefixes]
    preds = [1 if s >= threshold else 0 for s in scores]

    tp = sum(1 for yhat, y in zip(preds, labels) if yhat == 1 and y == 1)
    fp = sum(1 for yhat, y in zip(preds, labels) if yhat == 1 and y == 0)
    fn = sum(1 for yhat, y in zip(preds, labels) if yhat == 0 and y == 1)
    tn = sum(1 for yhat, y in zip(preds, labels) if yhat == 0 and y == 0)

    precision, recall, f1 = prf(tp, fp, fn)
    acc = safe_div(tp + tn, len(labels))
    auc = roc_auc(scores, labels)

    return {
        "verifier_acc": acc,
        "verifier_precision": precision,
        "verifier_recall": recall,
        "verifier_f1": f1,
        "verifier_auc": auc,
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(labels)},
    }

def eval_timing(text_rows: List[dict], verifier: DistilBertVerifier, alpha: float, threshold: float, max_texts: int = 300) -> dict:
    texts = [r["text"] for r in text_rows if "text" in r and isinstance(r["text"], str)]
    texts = [t for t in texts if len(t.split()) >= 6][:max_texts]

    triggers = []
    misses = 0
    for t in texts:
        idx = first_interrupt_word(verifier, t, alpha, threshold)
        if idx is None:
            misses += 1
        else:
            triggers.append(idx)

    avg_trigger = sum(triggers) / len(triggers) if triggers else 0.0
    miss_rate = safe_div(misses, len(texts)) if texts else 0.0

    return {
        "timing_avg_trigger_word": avg_trigger,
        "timing_miss_rate": miss_rate,
        "timing_triggered_fraction": safe_div(len(triggers), len(texts)) if texts else 0.0,
        "timing_n_texts": len(texts),
    }

def eval_speculation(text_rows: List[dict], verifier: DistilBertVerifier, speculator: Speculator,
                     alpha: float, threshold: float, max_texts: int = 50) -> dict:
    texts = [r["text"] for r in text_rows if "text" in r and isinstance(r["text"], str)]
    texts = [t for t in texts if len(t.split()) >= 6][:max_texts]

    comp_f1 = []
    comp_edit = []
    over_f1 = []
    over_edit = []
    latencies = []
    extra_tokens = []

    for t in texts:
        idx = first_interrupt_word(verifier, t, alpha, threshold)
        if idx is None:
            continue
        prefix = " ".join(t.split()[:idx])
        out = speculator.speculate(prefix)
        c = out.get("completed_user_message", "")
        o = out.get("overprojected_user_message", "")
        latencies.append(float(out.get("_latency_s", 0.0)))

        comp_f1.append(token_f1(c, t))
        comp_edit.append(edit_similarity(c, t))
        over_f1.append(token_f1(o, t))
        over_edit.append(edit_similarity(o, t))

        extra_tokens.append(max(0, len(tokenize(o)) - len(tokenize(c))))

    def avg(xs): return sum(xs)/len(xs) if xs else 0.0
    return {
        "spec_completed_token_f1_avg": avg(comp_f1),
        "spec_completed_edit_sim_avg": avg(comp_edit),
        "spec_overproj_token_f1_avg": avg(over_f1),
        "spec_overproj_edit_sim_avg": avg(over_edit),
        "spec_overproj_extra_tokens_avg": avg(extra_tokens),
        "spec_latency_ms_avg": avg(latencies) * 1000.0,
        "spec_n": len(comp_f1),
    }

def eval_response(text_rows: List[dict], verifier: DistilBertVerifier, speculator: Speculator, responder: Responder,
                  alpha: float, threshold: float, max_texts: int = 30, use_overprojected: bool = True) -> dict:
    texts = [r["text"] for r in text_rows if "text" in r and isinstance(r["text"], str)]
    texts = [t for t in texts if len(t.split()) >= 6][:max_texts]

    resp_lats = []
    resp_lens = []
    spec_lats = []

    for t in texts:
        idx = first_interrupt_word(verifier, t, alpha, threshold)
        if idx is None:
            continue
        prefix = " ".join(t.split()[:idx])
        spec = speculator.speculate(prefix)
        spec_lats.append(float(spec.get("_latency_s", 0.0)))

        msg = spec.get("overprojected_user_message" if use_overprojected else "completed_user_message", "")

        out = responder.respond(msg)
        resp_lats.append(float(out.get("_latency_s", 0.0)))
        resp_lens.append(len(tokenize(out.get("response", ""))))

    def avg(xs): return sum(xs)/len(xs) if xs else 0.0
    return {
        "resp_latency_ms_avg": avg(resp_lats) * 1000.0,
        "resp_length_tokens_avg": avg(resp_lens),
        "pipeline_latency_ms_avg": (avg(spec_lats) + avg(resp_lats)) * 1000.0,
        "resp_n": len(resp_lens),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL. Either {'text':...} or {'prefix':..., 'label':0/1}.")
    ap.add_argument("--mode", choices=["verifier", "timing", "speculation", "response", "all"], default="all")
    ap.add_argument("--model_dir", default="./verification_model")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base_url", default=ROUTER_BASE_URL_DEFAULT)
    ap.add_argument("--llm_model", default="deepseek-ai/DeepSeek-V3.2")
    ap.add_argument("--no_speculation", action="store_true")
    ap.add_argument("--no_response", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.data)

    verifier = DistilBertVerifier(model_dir=args.model_dir)

    report = {
        "config": {
            "alpha": args.alpha,
            "threshold": args.threshold,
            "model_dir": args.model_dir,
            "llm_model": args.llm_model,
            "base_url": args.base_url,
        }
    }

    # Decide dataset type
    has_prefix_labels = all(("prefix" in r and "label" in r) for r in rows if isinstance(r, dict))
    has_texts = any(("text" in r) for r in rows if isinstance(r, dict))

    if args.mode in ("verifier", "all"):
        if has_prefix_labels:
            prefixes = [r["prefix"] for r in rows]
            labels = [int(r["label"]) for r in rows]
            report.update(eval_verifier(prefixes, labels, verifier, args.threshold))
        elif has_texts:
            prefixes, labels, _ = generate_prefix_dataset_from_texts(rows, max_samples=args.max_samples, seed=args.seed)
            report.update(eval_verifier(prefixes, labels, verifier, args.threshold))
        else:
            raise RuntimeError("Data must contain either (prefix,label) or text.")

    if args.mode in ("timing", "all") and has_texts:
        report.update(eval_timing(rows, verifier, args.alpha, args.threshold))

    do_spec = (not args.no_speculation) and (args.mode in ("speculation", "all")) and has_texts
    do_resp = (not args.no_response) and (args.mode in ("response", "all")) and has_texts

    speculator = None
    responder = None

    if do_spec or do_resp:
        speculator = Speculator(model_name=args.llm_model, base_url=args.base_url)

    if do_spec:
        report.update(eval_speculation(rows, verifier, speculator, args.alpha, args.threshold))

    if do_resp:
        responder = Responder(model_name=args.llm_model, base_url=args.base_url)
        report.update(eval_response(rows, verifier, speculator, responder, args.alpha, args.threshold))

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
