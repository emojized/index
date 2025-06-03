import aiohttp
from typing import Any, Dict, List

# These are just examples — you may already have these defined elsewhere
class TextContent:
    def __init__(self, text: str):
        self.text = text

class Message:
    def __init__(self, role: str, content: Any):
        self.role = role
        self.content = content

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

    async def call(self, messages: List[Message]) -> Dict[str, Any]:
        # Only serialize message data at the last moment
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": msg.role,
                    "content": self._extract_content(msg.content)
                }
                for msg in messages
            ],
            "stream": False
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()

    def _extract_content(self, content: Any) -> str:
        """Recursively extract text from various content types"""
        if isinstance(content, TextContent):
            return content.text
        elif isinstance(content, list):
            return " ".join([self._extract_content(item) for item in content])
        elif hasattr(content, "text"):
            return content.text  # Fallback for duck-typed objects
        else:
            return str(content)