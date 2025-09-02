from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import torch
import torch.nn as nn

from ece_regularization import ECEConfig, ECERegularizer, training_step

# 1. Load base model in 8-bit
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Apply LoRA adapters
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# 3. Define ECE
ece = ECERegularizer(ECEConfig(lambda_ece=0.05, mode="logdet"))

# 4. Optimizer (8-bit AdamW)
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4)

# 5. Training loop example
for step, batch in enumerate(train_loader):
    batch = {k: v.to(model.device) for k, v in batch.items()}

    logs = training_step(
        model=model,
        batch=batch,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        ece=ece
    )

    if step % 50 == 0:
        print(logs)
