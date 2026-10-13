
from __future__ import annotations

import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


BASE_MODEL = "YichuanMa/LoGos-7B"

TRAIN_FILE = "data/final/logo_format/train.json"
VAL_FILE = "data/final/logo_format/val.json"
OUTPUT_DIR = "outputs/logo-qlora"

MAX_LENGTH = 3072
NUM_EPOCHS = 2
LEARNING_RATE = 5e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
SEED = 42


dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    },
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


def format_example(example):
    return (
        "### Instruction:\n"
        f"{example['input']}\n\n"
        "### Response:\n"
        f"{example['output']}"
        f"{tokenizer.eos_token}"
    )


compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=compute_dtype,
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=MAX_LENGTH,
    packing=False,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    max_grad_norm=1.0,
    bf16=compute_dtype == torch.bfloat16,
    fp16=compute_dtype == torch.float16,
    logging_steps=10,
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=400,
    save_strategy="steps",
    save_steps=400,
    save_total_limit=2,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    formatting_func=format_example,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()

final_dir = os.path.join(OUTPUT_DIR, "final")
trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

print(f"Saved adapter to: {final_dir}")
