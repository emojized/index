import aiohttp
from typing import Any, List, Union, Dict

class Message:
    def __init__(self, role: str, content: Any):
        self.role = role
        self.content = content

class OllamaProvider:
    def __init__(
        self,
        model: str = "llama3.2"
    ):
        self.model = model
        self.base_url = "http://localhost:11434/api"

    async def call(self, messages: Union[List[Message], str]) -> Dict[str, Any]:
        full_prompt = ""

        if isinstance(messages, str):
            full_prompt = messages.strip()
        elif isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, Message) and msg.role == "user":
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
        if hasattr(content, "text"):
            return content.text
        elif isinstance(content, dict) and "text" in content:
            return content["text"]
        elif isinstance(content, str):
            return content
        else:
            return str(content)