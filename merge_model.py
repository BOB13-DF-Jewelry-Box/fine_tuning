from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def save_merged_model_as_gguf(base_model_path, lora_adapter_path, output_path, cache_dir):
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, cache_dir=cache_dir)
    model = PeftModel.from_pretrained(model, lora_adapter_path, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, cache_dir=cache_dir)
    
    model = model.merge_and_unload()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    model_name = 'Saxo/Linkbricks-Horizon-AI-Korean-Advanced-8B'
    lora_adapter_path = './adapter'   # adapter 경로
    output_path = './merged_model'
    cache_dir = './cache'
    save_merged_model_as_gguf(model_name, lora_adapter_path, output_path, cache_dir=cache_dir)
