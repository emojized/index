import aiohttp
from typing import Any, List, Union

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

    async def call(self, messages: Union[List[Message], str]) -> Dict[str, Any]:
        """
        Accepts either a list of Message objects or a single string prompt.
        Converts everything into a flat prompt and sends it via Ollama's /generate endpoint.
        Returns raw response JSON.
        """

        # Extract full prompt from messages or use raw string
        if isinstance(messages, str):
            full_prompt = messages
        else:
            full_prompt = self._extract_prompt_from_messages(messages)

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/generate",
                json=payload
            ) as response:
                try:
                    response.raise_for_status()
                    return await response.json()
                except Exception as e:
                    raise RuntimeError(f"Ollama API Error: {e}, Response: {await response.text()}")

    def _extract_prompt_from_messages(self, messages: List[Message]) -> str:
        """Extracts the last user message content or falls back to first message"""
        for msg in reversed(messages):
            if msg.role == "user":
                return self._extract_content(msg.content)
        
        # Fallback: Use first message if no user message found
        if messages:
            return self._extract_content(messages[0].content)
        
        return ""

    def _extract_content(self, content: Any) -> str:
        """Recursively extract text from various content types"""
        if isinstance(content, TextContent):
            return content.text
        elif isinstance(content, list):
            return " ".join([self._extract_content(item) for item in content])
        elif hasattr(content, "text"):
            return content.text  # Fallback for duck-typed objects
        elif isinstance(content, dict) and "text" in content:
            return content["text"]
        elif isinstance(content, str):
            return content
        else:
            return str(content)