### Parameter
- r = 8
- lora_alpha = 16
- dropout = 0.1
- epochs = 5
- learning rate = 2e-6

### Adapter + basemodel => gguf
**adapter 폴더를 fine_tuning 폴더 내부에 넣고 실행(CUDA 필요)**
```python
git clone https://github.com/ggerganov/llama.cpp.git
python merge_model.py
python .\llama.cpp\convert_hf_to_gguf.py .\merged_model --outfile [name.gguf] --outtype q8_0
```

### Ollama (gguf, Modelfile 필요)
```python
ollama create [Name] -f Modelfile
ollama run [Name]
```

### Modelfile 만들기
- https://github.com/ollama/ollama/blob/main/docs/modelfile.md
