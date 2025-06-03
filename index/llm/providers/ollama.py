import aiohttp
from typing import Any, List, Union, Dict

# These may already exist elsewhere in your project
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
        model: str = "qwen2.5"
    ):
        self.model = model
        self.base_url = "http://localhost:11434/api"

    async def call(self, messages: Union[List[Message], str]) -> Dict[str, Any]:
        """
        Accepts raw strings or list of Message objects.
        Returns fake response for demo/testing.
        """

        full_prompt = ""

        # Case 1: Raw string prompt
        if isinstance(messages, str):
            full_prompt = messages.strip()

        # Case 2: List of Message objects
        elif isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and "type" in msg:
                    continue  # Skip stream control messages
                elif isinstance(msg, Message) and msg.role == "user":
                    full_prompt = self._extract_content(msg.content)
                    break
                elif isinstance(msg, dict):
                    if msg.get("role") == "user":
                        full_prompt = self._extract_content(msg.get("content", ""))
                        break
                else:
                    full_prompt = str(msg)
                    break

        if not full_prompt:
            full_prompt = "Hello"

        return {
            "model": self.model,
            "response": f"Faked response to: {full_prompt}",
            "done": True
        }

    def _extract_content(self, content: Any) -> str:
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