import aiohttp
from typing import Any, List, Union, Dict

# These classes may already exist elsewhere in your project
class TextContent:
    def __init__(self, text: str):
        self.text = text

class Message:
    def __init__(self, role: str, content: Any, name: str | None = None):
        self.role = role
        self.content = content
        self.name = name

    def to_dict(self):
        result = {"role": self.role, "content": self._serialize_content(self.content)}
        if self.name:
            result["name"] = self.name
        return result

    @staticmethod
    def _serialize_content(content: Any) -> Union[str, list]:
        if isinstance(content, TextContent):
            return content.text
        elif isinstance(content, list):
            return [Message._serialize_content(item) for item in content]
        elif hasattr(content, "text"):
            return content.text
        elif isinstance(content, dict) and "text" in content:
            return content["text"]
        elif isinstance(content, str):
            return content
        else:
            return str(content)

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

    async def call(self, messages: List[Union[Message, Dict]]) -> Dict[str, Any]:
        """
        Accepts a list of Message objects or dicts.
        Filters out non-prompt stream messages before calling Ollama /generate.
        Returns raw response from Ollama.
        """

        # Filter messages to extract only valid prompt input
        full_prompt = ""
        for msg in messages:
            # Skip internal stream control messages
            if isinstance(msg, dict) and "type" in msg:
                continue

            # Handle Message objects
            if isinstance(msg, Message):
                if msg.role == "user":
                    full_prompt = self._extract_content(msg.content)
                    break
            # Handle raw dictionaries
            elif isinstance(msg, dict):
                if msg.get("role") == "user":
                    full_prompt = self._extract_content(msg.get("content", ""))
                    break

        if not full_prompt:
            full_prompt = "Hello"

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

    def _extract_content(self, content: Any) -> str:
        """Recursively extract text from various content types"""
        if isinstance(content, TextContent):
            return content.text
        elif isinstance(content, list):
            return " ".join([self._extract_content(item) for item in content])
        elif hasattr(content, "text"):
            return content.text
        elif isinstance(content, dict) and "text" in content:
            return content["text"]
        elif isinstance(content, str):
            return content
        else:
            return str(content)