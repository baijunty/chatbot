import json
from .common import plugin_config, http_invoke
from nonebot import logger

class LLM:
    def __init__(self, *args):
        pass

    async def chat(self, message: list):
        pass

    async def generate(self, prompt: str):
        pass

    async def embeddings(self, prompt: str):
        pass


class Ollama(LLM):
    def __init__(self, *args):
        pass

    async def chat(self, message: list, model: str = plugin_config.assistant_chat_model):
        logger.info(f'调用模型{model} {[{msg['role']:msg['content']} for msg in message]} from {plugin_config.assistant_llm_base_url}')
        return await http_invoke(f'{plugin_config.assistant_llm_base_url}/api/chat', data=json.dumps({
            "model": model, "messages": message, "stream": False},ensure_ascii=False))

    async def generate(self, prompt: str, model: str = plugin_config.assistant_chat_model):
        return await http_invoke(f'{plugin_config.assistant_llm_base_url}/api/generate', data=json.dumps({
            "prompt": prompt, "model": model, "stream": False},ensure_ascii=False))


    async def embeddings(self, prompt: str, model: str = plugin_config.assistant_embeddings_model):
        return await http_invoke(f'{plugin_config.assistant_llm_base_url}/api/embed', data=json.dumps({
            "input": prompt, "model": model, "stream": False},ensure_ascii=False))



class OpenAI(LLM):
    def __init__(self, *args):
        self.api_key = plugin_config.assistant_openai_api_key

    async def chat(self, message: list, model: str = plugin_config.assistant_chat_model):
        return await http_invoke(f'{plugin_config.assistant_llm_base_url}/v1/chat/completions', headers={'Authorization': f'Bearer {self.api_key}'}, data=json.dumps({
            "model": model, "messages": message, "stream": False},ensure_ascii=False))


    async def generate(self, prompt: str, model: str = plugin_config.assistant_chat_model):
        return await http_invoke(f'{plugin_config.assistant_llm_base_url}/v1/completions', headers={'Authorization': f'Bearer {self.api_key}'}, data=json.dumps({
            "model": model, "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                         {"role": "user", "content": prompt}], "stream": False},ensure_ascii=False))


    async def embeddings(self, prompt: str, model: str = plugin_config.assistant_embeddings_model):
        return await http_invoke(f'{plugin_config.assistant_llm_base_url}/v1/embeddings', headers={'Authorization': f'Bearer {self.api_key}'}, data=json.dumps({
            "input": prompt, "model": model, "stream": False, "encoding_format": "float"},ensure_ascii=False))

