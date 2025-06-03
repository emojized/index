import aiohttp
from typing import Any, Dict

class OllamaProvider:
    def __init__(
        self,
        model: str = "qwen2.5",
        enable_thinking: bool = False,
        thinking_token_budget: int = 2048
    ):
        self.model = model
        self.enable_thinking = enable_thinking
        self.thinking_token_budget = thinking_token_budget
        self.base_url = "http://localhost:11434/api"

    async def call(self, prompt: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            ) as response:
                response.raise_for_status()
                return await response.json()