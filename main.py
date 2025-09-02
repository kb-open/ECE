# Load Model with LoRA + BitsAndBytes
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from ece_regularization import ECEConfig, ECERegularizer
from ece_trainer import ECETrainer

model_name = "meta-llama/Llama-2-7b-hf"

# Load in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LoRA adapters
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# Define ECE regularizer
ece = ECERegularizer(ECEConfig(lambda_ece=0.05, mode="logdet"))

# Optimizer (8-bit AdamW)
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4)

# Setup TrainingArguments + ECETrainer
from datasets import load_dataset

# Example dataset (replace with legal/finance/medical tasks)
dataset = load_dataset("imdb")
tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized["train"].select(range(1000))   # subset for demo
eval_dataset = tokenized["test"].select(range(500))

# Training arguments
args = TrainingArguments(
    output_dir="./ece-lora-results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=200,
    report_to="wandb",   # or "tensorboard"
    fp16=True,
)

# Initialize trainer
trainer = ECETrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, None),   # use our custom optimizer
    ece=ece,
)

# Train
trainer.train()