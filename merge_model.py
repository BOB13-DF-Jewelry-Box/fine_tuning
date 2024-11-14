from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

cache_dir='./cache'
adapter_path = "./adapter"
output_path = "./merged_model"
model_name = 'Saxo/Linkbricks-Horizon-AI-Korean-Advanced-8B'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir
    ).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
    )

model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)