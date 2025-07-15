import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Choose model
model_id = "gpt2-large"

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# Load tokenizer and model
print(f"🔄 Loading model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print(f"✅ Loaded {model_id} on {device}")

# Chat loop
while True:
    prompt = input("🧠 You: ")
    if prompt.lower() in ["exit", "quit", "q"]:
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n🤖 AI: {response}\n")
