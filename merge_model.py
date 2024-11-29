import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base_model_name_or_path = "Saxo/Linkbricks-Horizon-AI-Korean-Advanced-8B"
lora_model_path = "./1_3"
cache_dir="./cache"

model = AutoModelForCausalLM.from_pretrained(
    base_model_name_or_path,
    # load_in_4bit=True,    -> kernel died
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir,
    device_map="cuda"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
model = PeftModel.from_pretrained(model, lora_model_path)

model = model.merge_and_unload()
save_directory = "./merged_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

