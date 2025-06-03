import requests
from typing import Any, Dict

class OllamaProvider:
    def __init__(self, model: str = "qwen2.5"):
        self.model = model
        self.base_url = "http://localhost:11434/api"  # Default Ollama API endpoint

    async def generate(self, prompt: str) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()