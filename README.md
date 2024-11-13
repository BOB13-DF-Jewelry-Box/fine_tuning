### Parameter
- r = 8
- lora_alpha = 16
- dropout = 0.1
- epochs = 5
- learning rate = 2e-6

### fine tuning 모델 만들기
1. git clone https://github.com/ggerganov/llama.cpp.git
2. `python merge_model.py`
3. `python .\llama.cpp\convert_hf_to_gguf.py .\merged_model --outfile [name.gguf] --outtype q8_0`
4. ollama create [Name] -f Modelfile
5. ollama run [Name]

### Modelfile 만들기## Parameter
- r = 8
- lora_alpha = 16
- dropout = 0.1
- epochs = 5
- learning rate = 2e-6

### fine tuning 모델 만들기
1. git clone https://github.com/ggerganov/llama.cpp.git
2. python merge_model.py
3. python .\llama.cpp\convert_hf_to_gguf.py .\merged_model --outfile [name.gguf] --outtype q8_0
4. ollama create [Name] -f Modelfile
5. ollama run [Name]

### Modelfile 만들기
- https://github.com/ollama/ollama/blob/main/docs/modelfile.md
