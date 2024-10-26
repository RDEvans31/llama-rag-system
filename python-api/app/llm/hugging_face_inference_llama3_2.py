import os
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import httpx
import json
import requests

from pydantic import BaseModel

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message
    logprobs: Optional[dict] = None
    finish_reason: str

class ResponseBody(BaseModel):
    object: str
    id: str
    created: int
    model: str
    system_fingerprint: str
    choices: List[Choice]

# Make the generate function async
def generate(input: str):
    url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct/v1/chat/completions"

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": input
            }
        ],
        "max_tokens": 1000,
        "stream": False
    })

    headers = {
        'x-wait-for-model': 'true',
        'Content-Type': 'application/json',
        'Authorization': '••••••'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    valid_response = ResponseBody(**response.json())


    return response

async def agenerate(input: str):
    url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct/v1/chat/completions"

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": input
            }
        ],
        "max_tokens": 500,
        "stream": False
    })

    headers = {
        'x-wait-for-model': 'true',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {HUGGINGFACEHUB_API_TOKEN}'
    }

    # Use httpx instead of requests and await the result
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=payload)
        valid_response = ResponseBody(**response.json())

    
    return valid_response

class HuggingFaceInference(LLM):

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model's response to the prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        response = generate(prompt)
        
        return response
    
    async def _acall(self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        response: ResponseBody = await agenerate(prompt)
        
        return response.choices[0].message.content

    # async def _astream(
    #     self,
    #     prompt: str,
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> Iterator[GenerationChunk]:
    #     """Stream the LLM on the given prompt.

    #     This method should be overridden by subclasses that support streaming.

    #     If not implemented, the default behavior of calls to stream will be to
    #     fallback to the non-streaming version of the model and return
    #     the output as a single chunk.

    #     Args:
    #         prompt: The prompt to generate from.
    #         stop: Stop words to use when generating. Model output is cut off at the
    #             first occurrence of any of these substrings.
    #         run_manager: Callback manager for the run.
    #         **kwargs: Arbitrary additional keyword arguments. These are usually passed
    #             to the model provider API call.

    #     Returns:
    #         An iterator of GenerationChunks.
    #     """
    #     response = await generate(prompt)
        
    #     for char in prompt[: self.n]:
    #         chunk = GenerationChunk(text=char)
    #         if run_manager:
    #             run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    #         yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "HuggingFaceLlama3.2ChatCompletion",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"