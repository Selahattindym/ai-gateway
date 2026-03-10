import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "outputs/tinyllama-lora"
MERGED_PATH = "outputs/tinyllama-merged"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)

merged_model = model.merge_and_unload()

merged_model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)

print("Model merge edildi → outputs/tinyllama-merged")
