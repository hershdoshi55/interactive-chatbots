"""
interruptible_chatbot.py

Full 3-model interruptible chatbot pipeline:
  1. Verifier (DistilBERT): scores each word prefix; EMA triggers an interrupt
  2. Speculator (T5): completes the interrupted prefix into a full message
  3. Responder (DeepSeek via HF router): generates the assistant's reply

Commands:
  /send  — skip interrupt logic, send current input directly to Responder
  /reset — clear conversation history
  /quit  — exit

Requires:
  - ./verification_model  (from train_verification2.py)
  - ./speculative_model   (from spec_model2.py)
  - HF_TOKEN env var      (for HuggingFace router access)
"""

import json
import os
import re

import torch
from openai import OpenAI
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

# -------------------------
# Constants
# -------------------------

ALPHA = 0.6
THRESHOLD = 0.65
MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
ROUTER_URL = "https://router.huggingface.co/v1"

SYSTEM_PROMPT = (
    "You are an interruptible chatbot. You are cutting in while the user is still typing.\n"
    "Be short, polite, and helpful. Keep your response to 1-3 sentences."
)

# -------------------------
# Model loaders
# -------------------------

def load_verifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained("./verification_model")
    model = DistilBertForSequenceClassification.from_pretrained("./verification_model")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def load_speculator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5TokenizerFast.from_pretrained("./speculative_model")
    model = T5ForConditionalGeneration.from_pretrained("./speculative_model")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def load_responder():
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("HF_TOKEN environment variable is not set.")
    return OpenAI(base_url=ROUTER_URL, api_key=api_key)

# -------------------------
# Inference helpers
# -------------------------

def score_prefix(prefix, model, tokenizer, device):
    enc = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=64)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
    return prob


def autocomplete(prefix, model, tokenizer, device):
    """Run the T5 speculator and return completed_user_message."""
    input_text = f"prefix: {prefix}"
    enc = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=96)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **enc,
            max_new_tokens=64,
            num_beams=1,
            do_sample=False,
            early_stopping=True,
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    try:
        obj = json.loads(decoded)
        completed = obj.get("completed_user_message", "(missing)")
    except Exception:
        if '"completed_user_message":' in decoded:
            completed = decoded.split('"completed_user_message":')[-1].split('"')[1]
        else:
            completed = decoded

    # truncate runaway output
    completed = " ".join(completed.split()[:50])
    return completed


def get_response(client, messages):
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.4,
    )
    return (resp.choices[0].message.content or "").strip()

# -------------------------
# Main chatbot loop
# -------------------------

def main():
    print("Loading verifier model...")
    v_model, v_tok, v_dev = load_verifier()

    print("Loading speculator model...")
    s_model, s_tok, s_dev = load_speculator()

    print("Loading responder (HF router)...")
    client = load_responder()

    print("\nInterruptible Chatbot ready.")
    print("Commands: /send  /reset  /quit\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye.")
            break

        if user_input == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[History cleared]")
            continue

        if user_input == "/send":
            user_input = input("Message > ").strip()
            if not user_input:
                continue
            messages.append({"role": "user", "content": user_input})
            response = get_response(client, messages)
            messages.append({"role": "assistant", "content": response})
            print(f"Assistant: {response}\n")
            continue

        # Word-by-word EMA interrupt loop
        words = user_input.split()
        smooth = 0.0
        interrupted = False

        for i in range(1, len(words) + 1):
            prefix = " ".join(words[:i])
            score = score_prefix(prefix, v_model, v_tok, v_dev)
            smooth = ALPHA * score + (1 - ALPHA) * smooth
            print(f"  i={i} score={score:.3f} smooth={smooth:.3f} | {prefix}")

            if smooth > THRESHOLD:
                print(f"\n[INTERRUPT] at word {i} (EMA={smooth:.3f})")
                completed = autocomplete(prefix, s_model, s_tok, s_dev)
                print(f"  Prefix:    {prefix}")
                print(f"  Completed: {completed}")

                messages.append({"role": "user", "content": completed})
                response = get_response(client, messages)
                messages.append({"role": "assistant", "content": response})
                print(f"\nAssistant: {response}\n")

                interrupted = True
                break

        if not interrupted:
            print("[No interrupt — sending full message]")
            messages.append({"role": "user", "content": user_input})
            response = get_response(client, messages)
            messages.append({"role": "assistant", "content": response})
            print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
