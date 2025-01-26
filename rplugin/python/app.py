import subprocess
import time
import ollama

model_name = "llama3.2:3b"

try:
    ollama.pull(model_name)
except Exception as e:
    print(f"Error: {e}")
