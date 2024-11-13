from pydantic import BaseModel, field_validator

class Config(BaseModel):
    assistant_command_priority: int = 10
    assistant_plugin_enabled: bool = True
    assistant_plugin_weather_api_key: str = ''
    assistant_llm_base_url:str=''
    assistant_chat_model:str=''
    assistant_image_model:str=''
    assistant_code_model:str=''
    assistant_embeddings_model:str=''
    assistant_openai_api_key: str = ''
    assistant_llm_provider: str=''
    @field_validator("assistant_command_priority")
    @classmethod
    def check_priority(cls, v: int) -> int:
        if v >= 1:
            return v
        raise ValueError("weather command priority must greater than 1")
    
    def save_to_file(self):
        with open('config.json', 'w',encoding='utf-8') as file:
            file.write(self.model_dump_json())

    @property
    def llm(self):
        from .llm import Ollama,OpenAI
        if self.assistant_llm_provider=='ollama':
            return Ollama()
        return OpenAI()