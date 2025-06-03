import aiohttp
from typing import Any, Dict, List, Optional, Union

from index.llm.llm import BaseLLMProvider, LLMResponse, Message, ThinkingBlock


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/api"
    ):
        super().__init__(model=model)
        self.base_url = base_url

    async def call(
        self,
        messages: List[Message],
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Call the Ollama API with the given messages.
        
        Args:
            messages: List of Message objects
            temperature: Temperature for sampling
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            LLMResponse object with the response content
        """
        # Convert messages to Ollama format
        formatted_messages = self._format_messages(messages)
        
        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "options": {
                "temperature": temperature,
            }
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
            
        # Add any additional options from kwargs
        if "options" in kwargs:
            payload["options"].update(kwargs["options"])
            
        # Make the API call
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/chat", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")
                
                response_json = await response.json()
                
        # Extract the response content
        content = response_json.get("message", {}).get("content", "")
        
        # Calculate token usage (Ollama doesn't provide this directly)
        # This is an approximation
        prompt_tokens = sum(len(msg.get("content", "").split()) for msg in formatted_messages)
        completion_tokens = len(content.split())
        
        # Create and return the LLMResponse
        return LLMResponse(
            content=content,
            raw_response=response_json,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )
    
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Format messages for the Ollama API.
        
        Args:
            messages: List of Message objects
            
        Returns:
            List of formatted messages for Ollama API
        """
        formatted_messages = []
        
        for message in messages:
            # Skip state messages
            if message.is_state_message:
                continue
                
            # Extract content
            content = ""
            if isinstance(message.content, str):
                content = message.content
            elif isinstance(message.content, list):
                # For now, we only support text content
                for content_block in message.content:
                    if hasattr(content_block, "text"):
                        content += content_block.text
            
            # Map role (Ollama uses system/user/assistant)
            role = message.role
            if role not in ["system", "user", "assistant"]:
                # Default to user for other roles
                role = "user"
                
            formatted_messages.append({
                "role": role,
                "content": content
            })
            
        return formatted_messages
