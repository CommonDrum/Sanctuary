from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import time
from typing import Optional

app = FastAPI()

class ModelRequest(BaseModel):
    model: str = "llama3.2:3b"

class GenerateRequest(BaseModel):
    model: str = "llama3.2:3b"
    prompt: str

OLLAMA_BASE_URL = "http://ollama:11434/api"

@app.post("/pull")
async def pull_model(request: ModelRequest):
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/pull",
            json={"model": request.model}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Model pull failed")
        return {"status": "success", "message": f"Model {request.model} pulled successfully"}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/generate",
            json={
                "model": request.model,
                "prompt": request.prompt
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Text generation failed")
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)