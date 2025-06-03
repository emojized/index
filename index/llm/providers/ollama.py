import aiohttp
from typing import Any, List, Union, Dict

class TextContent:
    def __init__(self, text: str):
        self.text = text

class Message:
    def __init__(self, role: str, content: Any):
        self.role = role
        self.content = content

    def to_dict(self):
        return {
            "role": self.role,
            "content": self._serialize_content(self.content)
        }

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
        model: str = "llama3.2",
        enable_thinking: bool = False,
        thinking_token_budget: int = 2048
    ):
        self.model = model
        self.enable_thinking = enable_thinking
        self.thinking_token_budget = thinking_token_budget
        self.base_url = "http://localhost:11434/api"

    async def call(self, messages: List[Union[Message, Dict]]) -> Dict[str, Any]:
        """
        Accepts list of Message objects or dictionaries.
        Filters out internal stream control messages before calling Ollama /chat endpoint.
        """

        # Filter out stream control messages before processing
        payload_messages = []
        for msg in messages:
            # Skip internal stream control messages like {"type": "stream", ...}
            if isinstance(msg, dict) and "type" in msg:
                continue

            # Convert Message objects to dict
            if isinstance(msg, Message):
                payload_messages.append(msg.to_dict())
            elif isinstance(msg, dict):
                payload_messages.append({
                    "role": msg.get("role", "user"),
                    "content": self._extract_content(msg.get("content", ""))
                })
            else:
                # Fallback: wrap raw strings in user message
                payload_messages.append({
                    "role": "user",
                    "content": str(msg)
                })

        # Ensure at least one valid message was found
        if not payload_messages:
            payload_messages = [{"role": "user", "content": "Hello"}]

        payload = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat",
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