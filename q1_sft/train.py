# train.py

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset, Dataset
import torch
import json

# Load base model and tokenizer
model_name = "Meta-Llama-3-8B-Instruct"  # or smaller if you lack RAM
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Load dataset
with open("dataset.json") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

def tokenize(sample):
    return tokenizer(sample["text"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize)

# Apply LoRA
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training config
training_args = TrainingArguments(
    output_dir="./sft-output",
    per_device_train_batch_size=1,
    num_train_epochs=4,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    report_to="none"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained("./sft-model")
