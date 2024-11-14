### Parameter
- r = 8
- lora_alpha = 16
- dropout = 0.1
- epochs = 5
- learning rate = 2e-6

### Adapter + basemodel => gguf
'''
git clone https://github.com/ggerganov/llama.cpp.git
python merge_model.py
python .\llama.cpp\convert_hf_to_gguf.py .\merged_model --outfile [name.gguf] --outtype q8_0
'''

### Ollama (gguf, Modelfile 필요)
 ollama create [Name] -f Modelfile
6. ollama run [Name]

### Modelfile 만들기
- https://github.com/ollama/ollama/blob/main/docs/modelfile.md
