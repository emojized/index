import aiohttp
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from index.llm.llm import BaseLLMProvider, LLMResponse, Message, ThinkingBlock

logger = logging.getLogger(__name__)

class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/api",
        enable_thinking: bool = False,
        **kwargs
    ):
        super().__init__(model=model)
        self.base_url = base_url
        self.enable_thinking = enable_thinking

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
        # Check if this is an agent prompt that requires special handling
        is_agent_prompt = self._is_agent_prompt(messages)
        
        # Add a system message to enforce JSON output format if this is an agent prompt
        formatted_messages = self._format_messages(messages, is_agent_prompt)
        
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
            async with session.post(f"{self.base_url}/chat", json=payload, params={"stream": "false"}) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")
                
                # Handle both streaming and non-streaming responses
                content_type = response.headers.get('Content-Type', '')
                if 'application/x-ndjson' in content_type:
                    # Handle streaming response
                    full_response = ""
                    response_json = {}
                    async for line in response.content:
                        if not line.strip():
                            continue
                        try:
                            chunk = line.decode('utf-8').strip()
                            chunk_json = json.loads(chunk)
                            if 'message' in chunk_json and 'content' in chunk_json['message']:
                                full_response += chunk_json['message']['content']
                            response_json = chunk_json  # Keep the last chunk for metadata
                        except Exception as e:
                            logger.error(f"Error parsing chunk: {e}")
                    content = full_response
                else:
                    # Handle regular JSON response
                    response_json = await response.json()
                    content = response_json.get("message", {}).get("content", "")
        
        # Post-process the content if this is an agent prompt
        if is_agent_prompt:
            content = self._post_process_agent_response(content)
        
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
    
    def _is_agent_prompt(self, messages: List[Message]) -> bool:
        """
        Determine if this is an agent prompt that requires special JSON formatting.
        
        Args:
            messages: List of Message objects
            
        Returns:
            True if this is an agent prompt, False otherwise
        """
        # Check if any message contains the agent prompt signature
        for message in messages:
            content = ""
            if isinstance(message.content, str):
                content = message.content
            elif isinstance(message.content, list):
                for content_block in message.content:
                    if hasattr(content_block, "text"):
                        content += content_block.text
            
            # Look for agent prompt signatures
            if "<action_descriptions>" in content and "<o>" in content:
                return True
            if "Your response must always be in the following JSON format" in content:
                return True
        
        return False
    
    def _format_messages(self, messages: List[Message], is_agent_prompt: bool = False) -> List[Dict[str, Any]]:
        """
        Format messages for the Ollama API.
        
        Args:
            messages: List of Message objects
            is_agent_prompt: Whether this is an agent prompt that requires special handling
            
        Returns:
            List of formatted messages for Ollama API
        """
        formatted_messages = []
        
        # Add a system message to enforce JSON output format if this is an agent prompt
        if is_agent_prompt:
            formatted_messages.append({
                "role": "system",
                "content": """You are a browser agent that strictly follows instructions and outputs valid JSON.
Your responses must always be valid JSON objects wrapped in <o> tags.
Always use the exact format specified in the user's instructions.
Never include any explanations or text outside the <o> tags.
Ensure your JSON is properly formatted with all required fields."""
            })
        
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
    
    def _post_process_agent_response(self, content: str) -> str:
        """
        Post-process the agent response to ensure it's properly formatted.
        
        Args:
            content: The raw content from the LLM
            
        Returns:
            Properly formatted content
        """
        # Check if the content already has <o> tags
        if "<o>" in content and "</o>" in content:
            # Extract content between <o> tags
            pattern = r"<o>(.*?)</o>"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
                # Validate that it's proper JSON
                try:
                    json.loads(json_content)
                    return f"<o>{json_content}</o>"
                except json.JSONDecodeError:
                    # If not valid JSON, continue with further processing
                    pass
        
        # Try to find a JSON object in the content
        try:
            # Look for patterns that might indicate JSON content
            json_pattern = r'\{\s*"thought"\s*:.*"action"\s*:.*\}'
            match = re.search(json_pattern, content, re.DOTALL)
            if match:
                json_content = match.group(0).strip()
                # Validate that it's proper JSON
                try:
                    json.loads(json_content)
                    return f"<o>{json_content}</o>"
                except json.JSONDecodeError:
                    # If not valid JSON, continue with further processing
                    pass
            
            # If we couldn't find a valid JSON object, try to fix common issues
            # Remove any text before the first { and after the last }
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                json_content = content[first_brace:last_brace+1].strip()
                # Validate that it's proper JSON
                try:
                    json.loads(json_content)
                    return f"<o>{json_content}</o>"
                except json.JSONDecodeError:
                    # If still not valid JSON, return the best attempt wrapped in tags
                    return f"<o>{json_content}</o>"
        except Exception as e:
            logger.error(f"Error post-processing agent response: {e}")
        
        # If all else fails, return the original content wrapped in tags
        return f"<o>{content}</o>"
