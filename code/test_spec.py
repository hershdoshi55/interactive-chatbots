import torch
import json
from transformers import T5TokenizerFast, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
tokenizer = T5TokenizerFast.from_pretrained("./speculative_model")
model = T5ForConditionalGeneration.from_pretrained("./speculative_model")
model.to(device)
model.eval()


def autocomplete(prefix):
    """Return only completed_user_message from speculative model."""
    input_text = f"prefix: {prefix}"

    enc = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=96,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    output_ids = model.generate(
        **enc,
        max_new_tokens=64,
        num_beams=1,
        do_sample=False,
        early_stopping=True
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Try to parse JSON and extract only completed_user_message
    try:
        obj = json.loads(decoded)
        completed = obj.get("completed_user_message", "(missing)")
    except Exception:
        # fallback if JSON is invalid
        completed = decoded.split('"completed_user_message":')[-1].split('"')[1] if '"completed_user_message":' in decoded else decoded

    # truncate runaway text if needed
    completed = ' '.join(completed.split()[:50])
    return completed


def main():
    print("Speculative Autocomplete Test (only completed_user_message)")
    print("Type an incomplete message.")
    while True:
        prefix = input("\nPrefix > ").strip()
        completed = autocomplete(prefix)
        print("\ncompleted:")
        print(completed)


if __name__ == "__main__":
    main()
