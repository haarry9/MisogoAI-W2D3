# Tiny SFT: Polite LLM Fine-Tuning

This project fine-tunes a base LLM (LLaMA 3 8B) using a small supervised dataset via LoRA and PEFT.

---

### 🚀 Setup

```bash
pip install transformers peft accelerate datasets bitsandbytes
```
🧠 Training
```
python train.py
```
- Uses dataset.json for training

- Saves model in sft-model/

- Trains for 3–5 epochs

- Uses LoRA for efficiency

### 📊 Evaluation
Compare responses before and after using before_after.md. You can test new prompts using:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct", device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, "./sft-model")
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
pipe("<|user|>Translate hello to French.<|assistant|>", max_new_tokens=30)
```

### 🗃️ Files
- dataset.json — prompt/response pairs

- train.py — fine-tuning script

- before_after.md — eval results

- README.md — this guide