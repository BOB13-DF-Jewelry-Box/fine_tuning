from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import PeftModel
import torch

cache_dir='./cache'
adapter_path = "./adapter"
output_path = "./merged_model"
model_name = 'Saxo/Linkbricks-Horizon-AI-Korean-Advanced-8B'

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config
    )

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
    )

model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
