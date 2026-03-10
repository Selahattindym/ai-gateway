import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "outputs/tinyllama-lora"

def formatting_func(example):
    return f"""### Talimat:
{example["instruction"]}

### Cevap:
{example["output"]}"""

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.config.use_cache = False

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    training_args = TrainingArguments( output_dir=OUTPUT_DIR, per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=2e-4, 
        num_train_epochs=3, logging_steps=1, save_strategy="epoch", fp16=False, bf16=False, report_to="none"

    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        args=training_args
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"LoRA eğitimi bitti. Çıktı klasörü: {OUTPUT_DIR}")

if __name__ == "__main__": main()


