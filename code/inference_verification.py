import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# a smoothing constant 
ALPHA = 0.6
# the score to interrupt
THRESHOLD = 0.65 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model
tokenizer = DistilBertTokenizerFast.from_pretrained("./verification_model")
model = DistilBertForSequenceClassification.from_pretrained("./verification_model")
model.to(device)
model.eval()

# get the score of the prefix
def score_prefix(prefix):
    enc = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=64)
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits
    prob = torch.softmax(logits, dim=-1)[0,1].item()
    return prob

def main():
    print("Verification model (type sentence)")
    while True:
        sentence = input("\nSentence > ").strip()
        words = sentence.split()
        smooth = 0.0
        triggered = False

        for i in range(1, len(words)+1):
            prefix = " ".join(words[:i]) # continue to add to prefix
            score = score_prefix(prefix) # get the score
            smooth = ALPHA * score + (1 - ALPHA) * smooth # apply smoothing 

            # if greater than interrupt
            if smooth > THRESHOLD:
                print(f"Interrupt at word {i} ('{prefix}')")
                triggered = True
                break

        if not triggered:
            print("No interrupt for this sentence.")

if __name__ == "__main__":
    main()
